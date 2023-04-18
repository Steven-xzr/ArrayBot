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


class ArrayRobot(VecTask):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless,
                 virtual_screen_capture=False, force_render=False):
        self.cfg = OmegaConf.create(cfg)
        self.ori_obs = self.cfg.ori_obs
        cfg["env"]["numObservations"] = 16 if self.ori_obs else 12
        # cfg["env"]["numObservations"] = 12 - 3
        self.fixed_init = self.cfg.fixed_init
        super().__init__(config=cfg, rl_device=rl_device, sim_device=sim_device,
                         graphics_device_id=graphics_device_id, headless=headless,
                         virtual_screen_capture=virtual_screen_capture, force_render=force_render)
        self.max_episode_length = self.cfg.env.maxEpisodeLength
        self.action_speed_scale = self.cfg.env.actionSpeedScale

        self.dct_handler = BatchDCT(self.cfg.dct.order, self.cfg.dct.dim_local)
        self.half_dim = int((self.dct_handler.dim_local - 1) / 2)
        # self.dct_step = self.cfg.dct.step

        self._prepare_tensors()

        # set goal
        # TODO: set goals for orientations
        self.goal_pos = torch.tensor(self.cfg.goal.pos, device=self.device)  # (3,)

        # set reward
        self.trans_diff_factor = self.cfg.reward.trans_diff_factor
        self.trans_delta_diff_factor = self.cfg.reward.trans_delta_diff_factor
        self.reach_threshold = self.cfg.reward.reach_threshold
        self.reach_bonus = self.cfg.reward.reach_bonus

    def _prepare_tensors(self):
        self.root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        vec_root_tensor = gymtorch.wrap_tensor(self.root_tensor)
        vec_dof_tensor = gymtorch.wrap_tensor(self.dof_state_tensor)

        self.root_states = vec_root_tensor.view(self.num_envs, self.robot_side + 1, 13)
        self.object_positions = self.root_states[..., -1, 0:3]
        self.object_orientations = self.root_states[..., -1, 3:7]
        self.object_linvels = self.root_states[..., -1, 7:10]
        self.object_angvels = self.root_states[..., -1, 10:13]

        self.dof_states = vec_dof_tensor.view(self.num_envs, self.robot_side, self.robot_side, 2)
        self.dof_positions = self.dof_states[..., 0]
        self.dof_velocities = self.dof_states[..., 1]

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)

        self.initial_dof_states = self.dof_states.clone()
        self.initial_root_states = self.root_states.clone()

        self.dof_position_targets = torch.zeros((self.num_envs, self.robot_side, self.robot_side), dtype=torch.float32,
                                                device=self.device, requires_grad=False)
        self.all_actor_indices = torch.arange(self.num_envs * (self.robot_side + 1),
                                              dtype=torch.int32,
                                              device=self.device).view(self.num_envs, self.robot_side + 1)
        self.all_object_actor_indices = torch.arange(start=self.robot_side,
                                                     end=self.num_envs * (self.robot_side + 1),
                                                     step=self.robot_side + 1,
                                                     dtype=torch.int32,
                                                     device=self.device).view(self.num_envs, 1)
        self.all_robot_actor_indices = []
        for i in range(self.num_envs):
            self.all_robot_actor_indices.append(torch.arange(start=i * (self.robot_side + 1),
                                                             end=(i + 1) * (self.robot_side + 1) - 1,
                                                             dtype=torch.int32,
                                                             device=self.device))
        self.all_robot_actor_indices = torch.stack(self.all_robot_actor_indices, dim=0)
        self.all_robot_dof_indices = torch.arange(self.num_envs * self.robot_side ** 2,
                                                  dtype=torch.int32,
                                                  device=self.device).view(self.num_envs, self.robot_side ** 2)

    def allocate_buffers(self):
        super().allocate_buffers()
        self.local_centroid_buf = torch.zeros(self.num_envs, 2, device=self.device, dtype=torch.int32)
        self.translation_diff_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)

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
        self.init_asset_dof_states['pos'][:] = self.dof_middle

        object_root = os.path.join(os.path.expanduser("~"), cfg.object.root)
        object_file = cfg.object.file
        print("Loading asset '%s' from '%s'" % (object_file, object_root))
        self.asset_object = self.gym.load_asset(self.sim, object_root, object_file, gymapi.AssetOptions())
        self.object_half_extend = cfg.object.half_extend

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
        self.object_base_pos = torch.tensor([cfg.object.x, cfg.object.y, cfg.object.z + self.object_half_extend],
                                            device=self.device)

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
            pose_object.p = gymapi.Vec3(cfg.object.x, cfg.object.y, self.object_half_extend + cfg.object.z)
            pose_object.r = gymapi.Quat(0, 0.0, 0.0, 1)
            object_handle = self.gym.create_actor(env=env, asset=self.asset_object, pose=pose_object,
                                                  name="object" + str(i),
                                                  group=i,
                                                  filter=0)
            self.object_handles.append(object_handle)

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

    def compute_observations(self):
        # assert len(self.reset_buf.nonzero().squeeze(-1)) == 0
        # object states
        obj_pos = self._normalize_object_position(self.object_positions)  # [num_envs, 3]
        if self.ori_obs:
            obj_ori = self.object_orientations  # [num_envs, 4]
        obj_diff_pos = self._normalize_diff_position(self.object_positions - self.goal_pos)  # [num_envs, 3]

        # robot states
        # TODO: batch local selection
        local_tensors = []
        for env_id, global_tensor in enumerate(self.dof_positions):
            # handling out of border
            x = min(max(self.local_centroid_buf[env_id][0], self.half_dim), self.robot_side - self.half_dim - 1)
            y = min(max(self.local_centroid_buf[env_id][1], self.half_dim), self.robot_side - self.half_dim - 1)
            local_tensors.append(
                global_tensor[x - self.half_dim:x + self.half_dim + 1, y - self.half_dim:y + self.half_dim + 1])
        robot_local = torch.stack(local_tensors, dim=0)     # [num_envs, local_side, local_side]
        robot_dct = self.dct_handler.dct(robot_local)  # [num_envs, dct_handler.n_freq]

        # self.obs_buf = torch.cat([obj_pos, obj_ori, obj_diff_pos, robot_dct], dim=-1)
        self.obs_buf = torch.cat([obj_pos, obj_diff_pos, robot_dct], dim=-1)
        # self.obs_buf = torch.cat([obj_pos, robot_dct], dim=-1)
        return self.obs_buf

    def compute_reward(self):
        # TODO: check correctness
        trans_diff = torch.linalg.norm(self.object_positions - self.goal_pos, dim=-1)  # [num_envs]
        bonus = torch.where(trans_diff < torch.tensor([self.reach_threshold] * self.num_envs),
                            torch.tensor([self.reach_bonus] * self.num_envs), torch.tensor([0] * self.num_envs))
        self.rew_buf = (self.translation_diff_buf - trans_diff) * self.trans_delta_diff_factor + bonus
        self.translation_diff_buf = trans_diff

    def compute_reset(self):
        # TODO: check correctness
        timeout = torch.where(self.progress_buf >= torch.tensor([self.max_episode_length] * self.num_envs),
                              torch.ones(self.num_envs), torch.zeros(self.num_envs))
        x_out = torch.logical_or(self.local_centroid_buf[:, 0] < self.half_dim,
                                 self.local_centroid_buf[:, 0] >= self.robot_side - self.half_dim)
        y_out = torch.logical_or(self.local_centroid_buf[:, 1] < self.half_dim,
                                 self.local_centroid_buf[:, 1] >= self.robot_side - self.half_dim)
        self.reset_buf = torch.logical_or(torch.logical_or(x_out, y_out), timeout)

    def reset_idx(self, env_ids):
        # xyz_dist = Exponential(torch.tensor([5.0, 5.0, 5.0]))
        if not self.fixed_init:
            end_pos = torch.tensor([self.goal_pos[0] * 0.8, self.goal_pos[1] * 0.8, self.object_base_pos[2] * 1.1])
            xyz_dist = torch.distributions.uniform.Uniform(self.object_base_pos, end_pos)
            object_pos = xyz_dist.sample((len(env_ids),)) * (self.goal_pos - self.object_base_pos) + self.object_base_pos

        # reset root state for robots and objects in selected envs
        self.root_states[env_ids] = self.initial_root_states[env_ids]
        if not self.fixed_init:
            self.object_positions[env_ids][0:2] = object_pos[0:2]   # Note: z has to be high enough to avoid collision
        actor_indices = self.all_actor_indices[env_ids].flatten()
        self.gym.set_actor_root_state_tensor_indexed(self.sim, self.root_tensor, gymtorch.unwrap_tensor(actor_indices),
                                                     len(actor_indices))

        # reset dof states for robots in selected envs
        self.dof_states[env_ids] = self.initial_dof_states[env_ids]
        robot_actor_indices = self.all_robot_actor_indices[env_ids].flatten()
        self.gym.set_dof_state_tensor_indexed(self.sim, self.dof_state_tensor, gymtorch.unwrap_tensor(robot_actor_indices),
                                              len(robot_actor_indices))

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

        # reset object centroid & translation diff buffer
        self._update_centroids(env_ids)
        self.translation_diff_buf[env_ids] = torch.linalg.norm(self.object_positions[env_ids] - self.goal_pos, dim=-1)

    def pre_physics_step(self, _actions):
        # resets
        reset_env_ids = self.reset_buf.nonzero().squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        # transform action on frequencies to DOF shifts
        actions = _actions.to(self.device)      # [num_envs, dim_freq (6)]
        local_dof_shifts = self.dct_handler.idct(actions)  # [num_envs, dim_local, dim_local]
        # TODO: batch padding
        for env_id, dof_shift in enumerate(local_dof_shifts):
            if env_id not in reset_env_ids:
                centroid = self.local_centroid_buf[env_id]  # [2]
                dof_target_local = self.dof_position_targets[env_id][centroid[0] - self.half_dim:centroid[0] + self.half_dim + 1,
                                                                     centroid[1] - self.half_dim:centroid[1] + self.half_dim + 1]
                local_target = dof_target_local + dof_shift * self.dt * self.action_speed_scale
                self.dof_position_targets[env_id] = self.dof_middle
                self.dof_position_targets[env_id][centroid[0] - self.half_dim:centroid[0] + self.half_dim + 1,
                                                  centroid[1] - self.half_dim:centroid[1] + self.half_dim + 1] = local_target

        # update position targets from actions
        # self.dof_position_targets += self.dt * self.action_speed_scale * dof_shifts
        self.dof_position_targets = torch.clamp(self.dof_position_targets, self.dof_lower_limit, self.dof_upper_limit)

        # reset position targets for reset envs
        self.dof_position_targets[reset_env_ids] = self.dof_middle

        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_position_targets))

    def post_physics_step(self):
        self.progress_buf += 1

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)

        self._update_centroids()
        self.compute_reward()
        self.compute_reset()
        # # Immediate resets. Always return the observation after reset when an episode is done.
        # reset_env_ids = self.reset_buf.nonzero().squeeze(-1)
        # if len(reset_env_ids) > 0:
        #     self.reset_idx(reset_env_ids)
        self.compute_observations()

    def _update_centroids(self, env_ids=None):
        """
        Update local centroid buffer in selected envs
        """
        if env_ids is None:
            env_ids = list(range(self.num_envs))
        self.local_centroid_buf[env_ids] = torch.round((self.object_positions[env_ids] - self.base_unit_pos)
                                                       / self.robot_row_gap)[:, :2].int()


@hydra.main(version_base=None, config_path='config', config_name='test_isaac_vectask')
def main(cfg):
    envs = ArrayRobot(cfg=cfg, rl_device='cuda:0', sim_device='cuda:0', graphics_device_id=0, headless=False)

    while True:
        pass


if __name__ == '__main__':
    main()