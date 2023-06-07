import math
import numpy as np
import os
import hydra
from omegaconf import OmegaConf

from isaacgym import gymutil, gymtorch, gymapi
# from isaacgym.torch_utils import *
from isaacgymenvs.tasks.base.vec_task import VecTask

import torch
from torch.distributions.exponential import Exponential
from dct_transform import BatchDCT
from scipy.spatial.transform import Rotation as R
import torch_dct


class BaseEnv(VecTask):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless,
                 virtual_screen_capture=False, force_render=False):
        self.cfg = OmegaConf.create(cfg)
        # self.ori_obs = self.cfg.ori_obs
        cfg["env"]["numObservations"] = 3
        # cfg["env"]["numObservations"] = 12 - 3
        # self.fixed_init = self.cfg.fixed_init
        super().__init__(config=cfg, rl_device=rl_device, sim_device=sim_device,
                         graphics_device_id=graphics_device_id, headless=headless,
                         virtual_screen_capture=virtual_screen_capture, force_render=force_render)
        self.max_episode_length = self.cfg.env.maxEpisodeLength
        self.action_speed_scale = self.cfg.env.actionSpeedScale

        self.dct_handler = BatchDCT(self.cfg.dct.order, self.cfg.dct.dim_local)
        self.half_dim = int((self.dct_handler.dim_local - 1) / 2)
        # self.dct_step = self.cfg.dct.step

        self._prepare_tensors()

        # # set goal
        # # TODO: set goals for orientations
        # # self.goal_pos = torch.tensor(self.cfg.goal.pos, device=self.device)  # (3,)
        #
        # # set reward
        # self.trans_diff_factor = self.cfg.reward.trans_diff_factor
        # self.trans_delta_diff_factor = self.cfg.reward.trans_delta_diff_factor
        # self.reach_threshold = self.cfg.reward.reach_threshold
        # self.reach_bonus = self.cfg.reward.reach_bonus

        # reset
        self.reset_idx()

    def _prepare_tensors(self):
        self.root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        # self.force_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        vec_root_tensor = gymtorch.wrap_tensor(self.root_tensor)
        vec_dof_tensor = gymtorch.wrap_tensor(self.dof_state_tensor)
        # vec_force_tensor = gymtorch.wrap_tensor(self.force_tensor)

        num_objects = 1

        self.root_states = vec_root_tensor.view(self.num_envs, self.robot_side + num_objects, 13)
        self.object_positions_gt = self.root_states[..., -1, 0:3]
        self.object_orientations = self.root_states[..., -1, 3:7]
        self.object_linvels = self.root_states[..., -1, 7:10]
        self.object_angvels = self.root_states[..., -1, 10:13]

        self.dof_states = vec_dof_tensor.view(self.num_envs, self.robot_side, self.robot_side, 2)
        self.dof_positions = self.dof_states[..., 0]
        self.dof_velocities = self.dof_states[..., 1]

        # self.forces = vec_force_tensor[:, 0:3].view(self.num_envs, self.robot_side, self.robot_side, 3)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        # self.gym.refresh_force_sensor_tensor(self.sim)

        self.initial_dof_states = self.dof_states.clone()
        self.initial_root_states = self.root_states.clone()

        self.dof_position_targets = torch.zeros((self.num_envs, self.robot_side, self.robot_side), dtype=torch.float32,
                                                device=self.device, requires_grad=False)
        self.all_actor_indices = torch.arange(self.num_envs * (self.robot_side + num_objects),
                                              dtype=torch.int32,
                                              device=self.device).view(self.num_envs, self.robot_side + num_objects)
        # self.all_object_actor_indices = torch.arange(start=self.robot_side,
        #                                              end=self.num_envs * (self.robot_side + 1),
        #                                              step=self.robot_side + 1,
        #                                              dtype=torch.int3num_objects,
        #                                              device=self.device).view(self.num_envs, 1)
        self.all_robot_actor_indices = []
        for i in range(self.num_envs):
            self.all_robot_actor_indices.append(torch.arange(start=i * (self.robot_side + num_objects),
                                                             end=(i + 1) * (self.robot_side + num_objects) - num_objects,
                                                             dtype=torch.int32,
                                                             device=self.device))
        self.all_robot_actor_indices = torch.stack(self.all_robot_actor_indices, dim=0)
        self.all_robot_dof_indices = torch.arange(self.num_envs * self.robot_side ** 2,
                                                  dtype=torch.int32,
                                                  device=self.device).view(self.num_envs, self.robot_side ** 2)

    def allocate_buffers(self):
        super().allocate_buffers()
        self.obj_pos_buf = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float32)
        self.obj_height_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.obj_euler_buf = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float32)
        self.obj_quat_buf = torch.zeros(self.num_envs, 4, device=self.device, dtype=torch.float32)
        self.obj_rotvec_buf = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float32)

        self.local_centroid_buf = torch.zeros(self.num_envs, 2, device=self.device, dtype=torch.int32)
        self.height_diff_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.euler_diff_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.ori_diff_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)

    def create_sim(self):
        self.dt = self.sim_params.dt
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81

        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)

        self._create_asset()
        self._create_ground_plane()
        self._create_envs()

    def _create_asset(self):
        # cfg = self.cfg["env"]["assets"]
        cfg = self.cfg.env.assets

        robot_root = os.path.join(os.path.expanduser("~"), cfg.robot.root)
        robot_file = cfg.robot.file
        asset_options_robot = gymapi.AssetOptions()
        asset_options_robot.fix_base_link = True
        print("Loading asset '%s' from '%s'" % (robot_file, robot_root))
        print("NOTE: Due to the DOF limit in PhysX, we slice the array robot in rows.")
        self.asset_robot = self.gym.load_asset(self.sim, robot_root, robot_file, asset_options_robot)
        self.robot_side = self.gym.get_asset_dof_count(self.asset_robot)
        self.asset_dof_props = self.gym.get_asset_dof_properties(self.asset_robot)
        self.dof_lower_limit = self.asset_dof_props['lower'][0]
        self.dof_upper_limit = self.asset_dof_props['upper'][0]
        self.dof_middle = (self.dof_lower_limit + self.dof_upper_limit) / 2
        self.dof_range = self.dof_upper_limit - self.dof_lower_limit

        self.init_asset_dof_states = np.zeros(self.robot_side, dtype=gymapi.DofState.dtype)
        # self.init_asset_dof_states['pos'][:] = self.dof_middle
        self.init_asset_dof_states['pos'][:] = self.dof_lower_limit

        # object assets
        object_root = os.path.join(os.path.expanduser("~"), 'urdf')
        self.asset_object_list = []
        asset_options_object = gymapi.AssetOptions()
        asset_options_object.override_com = True
        asset_options_object.override_inertia = True
        asset_options_object.vhacd_enabled = True
        self.asset_object = self.gym.load_asset(self.sim, 'urdf', 'box-8cm.urdf', asset_options_object)

        self.object_half_extend = 0.04

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.distance = 0.1
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self):
        cfg = self.cfg.env

        self.robot_row_gap = cfg.robot.row_gap
        self.robot_size = self.robot_row_gap * self.robot_side  # size per side

        num_envs_per_row = int(np.sqrt(self.num_envs))
        spacing = self.robot_side * self.robot_row_gap
        env_lower = gymapi.Vec3(-spacing / 2, -spacing / 2, -spacing / 2)
        env_upper = gymapi.Vec3(3 * spacing / 2, 3 * spacing / 2, spacing)

        self.envs = []
        self.robot_row_handles = []
        self.object_handles = []
        self.vis_handles = []
        self.object_base_pos = torch.tensor([cfg.object.x, cfg.object.y, cfg.object.z + self.object_half_extend],
                                            device=self.device)
        self.object_init_z = self.dof_upper_limit + self.object_half_extend + 0.04
        self.fixed_object_init_pos = np.array([self.robot_size / 2, self.robot_size / 2, self.object_init_z])
        self.fixed_object_init_ori = np.array([0, 0, 0, 1])

        self.goal_height = self.dof_upper_limit + self.object_half_extend + 0.04
        self.goal_euler = torch.tensor([np.pi / 2, 0, 0], device=self.device, dtype=torch.float32)
        self.goal_ori = R.from_euler('xyz', self.goal_euler)

        print("Creating %d environments." % self.num_envs)
        for i in range(self.num_envs):
            env = self.gym.create_env(self.sim, env_lower, env_upper, num_envs_per_row)
            self.envs.append(env)

            # Create robot. Here, we slice the array robot in rows.
            for j in range(self.robot_side):
                pose_robot = gymapi.Transform()
                pose_robot.p = gymapi.Vec3(self.robot_row_gap * j, 0, -0.02)  # compensate for the ee height
                pose_robot.r = gymapi.Quat(0, 0.0, 0.0, 1)

                robot_row_handle = self.gym.create_actor(env=env, asset=self.asset_robot, pose=pose_robot,
                                                         name="robot_" + str(i) + "_" + str(j),
                                                         group=i,
                                                         filter=1)  # We exclude the collisions within the robots.
                self.robot_row_handles.append(robot_row_handle)

                # Set DOF properties. NOTE: force = posError * stiffness + velError * damping
                props = self.gym.get_actor_dof_properties(env, robot_row_handle)
                props["driveMode"][:] = gymapi.DOF_MODE_POS
                props["velocity"][:] = cfg.robot.velocity
                props["stiffness"][:] = cfg.robot.stiffness
                props["damping"][:] = cfg.robot.damping
                props["effort"][:] = cfg.robot.effort
                self.gym.set_actor_dof_properties(env, robot_row_handle, props)
                self.gym.set_actor_dof_states(env, robot_row_handle, self.init_asset_dof_states, gymapi.STATE_ALL)

            pose_object = gymapi.Transform()
            # pose_object.p = gymapi.Vec3(self.robot_size / 2, self.robot_size / 2, self.object_init_z)
            pose_object.p = gymapi.Vec3(self.fixed_object_init_pos[0],
                                        self.fixed_object_init_pos[1],
                                        self.fixed_object_init_pos[2])
            pose_object.r = gymapi.Quat(self.fixed_object_init_ori[0],
                                        self.fixed_object_init_ori[1],
                                        self.fixed_object_init_ori[2],
                                        self.fixed_object_init_ori[3])

            # Create object
            object_handle = self.gym.create_actor(env=env, asset=self.asset_object, pose=pose_object,
                                                  name="object" + str(i),
                                                  group=i,
                                                  filter=0)
            self.object_handles.append(object_handle)

        # # Global rigid body ids
        # self.global_vis_id = self.robot_side + self.robot_side ** 2 * 3
        # self.global_obj_id = self.global_vis_id + 1

        # Verify the gaps in row and in column are the same.
        rb_states = self.gym.get_actor_rigid_body_states(self.envs[0], 0, gymapi.STATE_POS)
        self.base_unit_pos = torch.tensor(rb_states[1][0][0].item())  # [3, ]
        next_unit_pos = torch.tensor(rb_states[4][0][0].item())
        assert next_unit_pos[1] - self.base_unit_pos[1] == self.robot_row_gap

    def _normalize_object_position(self, pos):
        # FIXME: bug in z
        x = 2 * pos[:, 0] / self.robot_size - 1
        y = 2 * pos[:, 1] / self.robot_size - 1
        z = 2 * (pos[:, 2] - self.object_half_extend - self.dof_range / 2) / self.dof_range
        return torch.vstack([x, y, z]).T

    def _normalize_diff_position(self, pos):
        x = pos[:, 0] / self.robot_size
        y = pos[:, 1] / self.robot_size
        z = pos[:, 2] / self.dof_range
        return torch.vstack([x, y, z]).T

    def compute_observations(self, env_ids=None):
        """
        Update object position buffer, observation buffer, and local centroid buffer.
        NOTE: Should be called after ALL the tensors are refreshed.
        """
        if env_ids is None:
            env_ids = list(range(self.num_envs))

        self.obj_pos_buf[env_ids] = self.object_positions_gt[env_ids].clone()
        self.obj_height_buf[env_ids] = self.obj_pos_buf[env_ids, -1].clone()
        obj_oris = R.from_quat(self.object_orientations[env_ids].clone())
        self.obj_euler_buf[env_ids] = torch.tensor(obj_oris.as_euler("xyz", degrees=False), dtype=torch.float32)
        self.obj_quat_buf[env_ids] = self.object_orientations[env_ids].clone()
        self.obj_rotvec_buf[env_ids] = torch.tensor(obj_oris.as_rotvec(), dtype=torch.float32)

        # Update centroids
        self.local_centroid_buf[env_ids] = torch.round((self.obj_pos_buf[env_ids] - self.base_unit_pos)
                                                       / self.robot_row_gap)[:, :2].int()

    def compute_reward(self):
        raise NotImplementedError
        # # TODO: check correctness
        # trans_diff = torch.linalg.norm(self.object_positions_gt - self.goal_pos_buf, dim=-1)  # [num_envs]
        # bonus = torch.where(trans_diff < torch.tensor([self.reach_threshold] * self.num_envs),
        #                     torch.tensor([self.reach_bonus] * self.num_envs), torch.tensor([0] * self.num_envs))
        # self.rew_buf = (self.translation_diff_buf - trans_diff) * self.trans_delta_diff_factor + bonus
        # self.translation_diff_buf = trans_diff

    def compute_reset(self):
        timeout = torch.where(self.progress_buf >= torch.tensor([self.max_episode_length] * self.num_envs),
                              torch.ones(self.num_envs), torch.zeros(self.num_envs))
        local_centroids = torch.round((self.object_positions_gt - self.base_unit_pos) / self.robot_row_gap)[:, :2].int()
        x_out = torch.logical_or(local_centroids[:, 0] < self.half_dim,
                                 local_centroids[:, 0] >= self.robot_side - self.half_dim)
        y_out = torch.logical_or(local_centroids[:, 1] < self.half_dim,
                                 local_centroids[:, 1] >= self.robot_side - self.half_dim)
        self.reset_buf = torch.logical_or(torch.logical_or(x_out, y_out), timeout)

    def reset_idx(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs)

        # reset object positions
        low_pos = torch.tensor([self.robot_size * 0.2, self.robot_size * 0.2, self.dof_lower_limit + self.object_half_extend])
        high_pos = torch.tensor([self.robot_size * 0.8, self.robot_size * 0.8, self.dof_upper_limit + self.object_half_extend])
        xyz_dist = torch.distributions.uniform.Uniform(low_pos, high_pos)
        object_pos = xyz_dist.sample((len(env_ids),)).float()
        object_pos[:, 2] = self.object_init_z   # Note: z has to be high enough to avoid collision

        # reset root state for robots and objects in selected envs
        self.root_states[env_ids] = self.initial_root_states[env_ids]

        # # self.root_states[env_ids, -2, 0:3] = goal_pos
        # self.root_states[env_ids, -1, 0:3] = self.fixed_object_init_pos
        # self.root_states[env_ids, -1, 3:7] = self.fixed_object_init_ori

        actor_indices = self.all_actor_indices[env_ids].flatten()
        self.gym.set_actor_root_state_tensor_indexed(self.sim, self.root_tensor, gymtorch.unwrap_tensor(actor_indices),
                                                     len(actor_indices))

        # reset dof states for robots in selected envs
        self.dof_states[env_ids] = self.initial_dof_states[env_ids]
        robot_actor_indices = self.all_robot_actor_indices[env_ids].flatten()
        self.gym.set_dof_state_tensor_indexed(self.sim, self.dof_state_tensor, gymtorch.unwrap_tensor(robot_actor_indices),
                                              len(robot_actor_indices))
        self.dof_position_targets[env_ids] = torch.tensor(self.dof_lower_limit)

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

        # reset object position, local centroid, observation, & translation diff buffer
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        # self.gym.refresh_force_sensor_tensor(self.sim)
        self.compute_observations(env_ids)
        # self.translation_diff_buf[env_ids] = torch.linalg.norm(self.object_positions_gt[env_ids] - self.goal_pos_buf[env_ids], dim=-1)
        self.height_diff_buf[env_ids] = np.abs(self.dof_lower_limit + self.object_half_extend - self.goal_height)
        self.euler_diff_buf[env_ids] = torch.linalg.norm(self.obj_euler_buf[env_ids] - self.goal_euler, dim=-1)

        self.obj_height_buf[env_ids] = self.object_positions_gt[env_ids, 2]

        ori = R.from_quat(self.object_orientations[env_ids])
        ori_diff = ori * R.inv(self.goal_ori)
        self.ori_diff_buf[env_ids] = torch.linalg.norm(torch.tensor(ori_diff.as_rotvec()), dim=-1).float()

    def action_handler(self, _action):
        raise NotImplementedError

    def pre_physics_step(self, _actions):
        # resets
        reset_env_ids = self.reset_buf.nonzero().squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        # transform action on frequencies to DOF shifts
        actions = _actions.to(self.device)      # [num_envs, dim_freq (6 / 25)]
        # local_dof_shifts = self.dct_handler.idct(actions)  # [num_envs, dim_local, dim_local]
        local_dof_shifts = self.action_handler(actions)  # [num_envs, dim_local, dim_local]
        # TODO: batch padding
        for env_id, dof_shift in enumerate(local_dof_shifts):
            if env_id not in reset_env_ids:
                # handling out of border
                x = min(max(self.local_centroid_buf[env_id][0], self.half_dim), self.robot_side - self.half_dim - 1)
                y = min(max(self.local_centroid_buf[env_id][1], self.half_dim), self.robot_side - self.half_dim - 1)

                dof_target_local = self.dof_position_targets[env_id][x - self.half_dim:x + self.half_dim + 1,
                                                                     y - self.half_dim:y + self.half_dim + 1]
                local_target = dof_target_local + dof_shift * self.dt * self.action_speed_scale
                self.dof_position_targets[env_id] = torch.tensor(self.dof_lower_limit)
                self.dof_position_targets[env_id][x - self.half_dim:x + self.half_dim + 1,
                                                  y - self.half_dim:y + self.half_dim + 1] = local_target

        # update position targets from actions
        # self.dof_position_targets += self.dt * self.action_speed_scale * dof_shifts
        self.dof_position_targets = torch.clamp(self.dof_position_targets, self.dof_lower_limit, self.dof_upper_limit)

        # reset position targets for reset envs
        self.dof_position_targets[reset_env_ids] = torch.tensor(self.dof_lower_limit)

        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_position_targets))

    def post_physics_step(self):
        self.progress_buf += 1

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        # self.gym.refresh_force_sensor_tensor(self.sim)

        self.compute_observations()
        self.compute_reward()
        self.compute_reset()

        # # Immediate resets to reduce computational cost.
        # reset_env_ids = self.reset_buf.nonzero().squeeze(-1)
        # if len(reset_env_ids) > 0:
        #     self.reset_idx(reset_env_ids)


class Lift(BaseEnv):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless,
                 virtual_screen_capture=False, force_render=False):
        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)

    def compute_reward(self):
        height_diff = np.abs(self.obj_height_buf - self.goal_height)
        self.rew_buf = (self.height_diff_buf - height_diff) * 100
        self.rew_buf = (self.height_diff_buf - height_diff) * 100
        self.height_diff_buf = height_diff

    def compute_observations(self, env_ids=None):
        if env_ids is None:
            env_ids = list(range(self.num_envs))
        super().compute_observations(env_ids)
        self.obs_buf[env_ids] = self._normalize_object_position(self.obj_pos_buf[env_ids])


class LiftFreq6(Lift):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture=False, force_render=False):
        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)

    def action_handler(self, _action):
        # action: [num_envs, 6]
        return self.dct_handler.idct(_action)   # [num_envs, 5, 5]


class LiftFreq25(Lift):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture=False, force_render=False):
        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)

    def action_handler(self, _action):
        # action: [num_envs, 25]
        return torch_dct.idct_2d(_action.reshape(_action.shape[0], 5, 5))


class LiftSpatial(Lift):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture=False, force_render=False):
        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)

    def action_handler(self, _action):
        # action: [num_envs, 25]
        return _action.reshape(_action.shape[0], 5, 5)


class Tilt(BaseEnv):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless,
                 virtual_screen_capture=False, force_render=False):
        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)

    def compute_reward(self):
        # euler_diff = torch.linalg.norm(self.obj_euler_buf - self.goal_euler, dim=-1)
        # self.rew_buf = (self.euler_diff_buf - euler_diff) * 10
        # self.row_diff_buf = euler_diff

        ori = R.from_quat(self.object_orientations)
        ori_diff = ori * R.inv(self.goal_ori)
        ori_diff = torch.linalg.norm(torch.tensor(ori_diff.as_rotvec()), dim=-1).float()
        self.rew_buf = (self.ori_diff_buf - ori_diff) * 10
        self.ori_diff_buf = ori_diff

    def compute_observations(self, env_ids=None):
        if env_ids is None:
            env_ids = list(range(self.num_envs))
        super().compute_observations(env_ids)
        self.obs_buf[env_ids] = self.obj_rotvec_buf[env_ids]


class TiltFreq6(Tilt):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture=False, force_render=False):
        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)

    def action_handler(self, _action):
        # action: [num_envs, 6]
        return self.dct_handler.idct(_action)   # [num_envs, 5, 5]


class TiltFreq25(Tilt):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture=False, force_render=False):
        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)

    def action_handler(self, _action):
        # action: [num_envs, 25]
        return torch_dct.idct_2d(_action.reshape(_action.shape[0], 5, 5))


class TiltSpatial(Tilt):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture=False, force_render=False):
        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)

    def action_handler(self, _action):
        # action: [num_envs, 25]
        return _action.reshape(_action.shape[0], 5, 5)


@hydra.main(version_base=None, config_path='waste/config', config_name='test_isaac_vectask')
def main(cfg):
    envs = LiftFreq6(cfg=cfg, rl_device='cuda:0', sim_device='cuda:0', graphics_device_id=0, headless=False)

    while True:
        pass


if __name__ == '__main__':
    main()