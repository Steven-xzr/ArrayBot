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
        # self.ori_obs = self.cfg.ori_obs
        cfg["env"]["numObservations"] = 9 + 6
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

        # set goal
        # TODO: set goals for orientations
        # self.goal_pos = torch.tensor(self.cfg.goal.pos, device=self.device)  # (3,)

        # set reward
        self.trans_diff_factor = self.cfg.reward.trans_diff_factor
        self.trans_delta_diff_factor = self.cfg.reward.trans_delta_diff_factor
        self.reach_threshold = self.cfg.reward.reach_threshold
        self.reach_bonus = self.cfg.reward.reach_bonus

        # reset
        self.reset_idx()

    def _prepare_tensors(self):
        self.root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.force_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        vec_root_tensor = gymtorch.wrap_tensor(self.root_tensor)
        vec_dof_tensor = gymtorch.wrap_tensor(self.dof_state_tensor)
        vec_force_tensor = gymtorch.wrap_tensor(self.force_tensor)

        num_objects = 2

        self.root_states = vec_root_tensor.view(self.num_envs, self.robot_side + num_objects, 13)
        self.object_positions_gt = self.root_states[..., -1, 0:3]
        self.object_orientations = self.root_states[..., -1, 3:7]
        self.object_linvels = self.root_states[..., -1, 7:10]
        self.object_angvels = self.root_states[..., -1, 10:13]

        self.dof_states = vec_dof_tensor.view(self.num_envs, self.robot_side, self.robot_side, 2)
        self.dof_positions = self.dof_states[..., 0]
        self.dof_velocities = self.dof_states[..., 1]

        self.forces = vec_force_tensor[:, 0:3].view(self.num_envs, self.robot_side, self.robot_side, 3)

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
        self.goal_pos_buf = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float32)
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

        # force sensors are installed between the cuboid links and cylinder links
        rigid_body_count = self.gym.get_asset_rigid_body_count(self.asset_robot)
        rigid_body_dict = self.gym.get_asset_rigid_body_dict(self.asset_robot)

        self.x_indices, self.y_indices = torch.meshgrid(torch.arange(self.robot_side), torch.arange(self.robot_side))

        self.force_threshold = cfg.force_threshold

        sphere_indices = [3 + i * 3 for i in range(self.robot_side)]
        cylinder_indices = [2 + i * 3 for i in range(self.robot_side)]
        cuboid_indices = [1 + i * 3 for i in range(self.robot_side)]

        sensor_pose = gymapi.Transform(gymapi.Vec3(0, 0, 0))       # TODO: check correctness
        sensor_props = gymapi.ForceSensorProperties()
        sensor_props.enable_forward_dynamics_forces = False
        sensor_props.enable_constraint_solver_forces = True
        sensor_props.use_world_frame = False

        # for cylinder_index in cylinder_indices:
        #     self.gym.create_asset_force_sensor(self.asset_robot, cylinder_index, sensor_pose, sensor_props)
        for sphere_index in sphere_indices:
            self.gym.create_asset_force_sensor(self.asset_robot, sphere_index, sensor_pose, sensor_props)
        # for cuboid_index in cuboid_indices:
        #     self.gym.create_asset_force_sensor(self.asset_robot, cuboid_index, sensor_pose, sensor_props)

        # object assets
        object_root = os.path.join(os.path.expanduser("~"), cfg.object.root)
        self.asset_object_list = []
        asset_options_object = gymapi.AssetOptions()
        asset_options_object.override_com = True
        asset_options_object.override_inertia = True
        asset_options_object.vhacd_enabled = True

        self.asset_vis_list = []
        asset_options_vis = gymapi.AssetOptions()
        asset_options_vis.disable_gravity = True
        asset_options_vis.fix_base_link = True

        # object_file = cfg.object.file
        # print("Loading asset '%s' from '%s'" % (object_file, object_root))
        # self.asset_object = self.gym.load_asset(self.sim, object_root, object_file, gymapi.AssetOptions())
        # self.asset_ball = self.gym.load_asset(self.sim, object_root, 'ball.urdf', gymapi.AssetOptions())
        # self.asset_cube = self.gym.load_asset(self.sim, object_root, 'box.urdf', gymapi.AssetOptions())

        self.object_half_extend = cfg.object.half_extend
        difficulties = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

        slice_object_file = 'A00_0_rescaled.urdf'
        # slice_object_file = 'box.urdf'
        asset_object = self.gym.load_asset(self.sim, object_root, slice_object_file, asset_options_object)
        asset_vis = self.gym.load_asset(self.sim, object_root, slice_object_file, asset_options_vis)

        for env_idx in range(self.num_envs):
            self.asset_object_list.append(asset_object)
            self.asset_vis_list.append(asset_vis)
            continue

            # env_idx = 0
            if 'train' in object_root:
                difficulty = env_idx // 16
                complexity = (env_idx - difficulty * 16) // 2
                idx = env_idx % 2
                object_file = difficulties[difficulty] + '0' + str(complexity) + '_' + str(idx) + '_rescaled.urdf'
                if os.path.exists(os.path.join(object_root, object_file)):
                    print("Loading asset '%s' from '%s'" % (object_file, object_root))
                    self.asset_object_list.append(self.gym.load_asset(self.sim, object_root, object_file, asset_options_object))
                    self.asset_vis_list.append(self.gym.load_asset(self.sim, object_root, object_file, asset_options_vis))
                else:
                    print("Target asset not found!")
                    asset_idx = np.random.randint(0, len(self.asset_object_list))
                    self.asset_object_list.append(self.asset_object_list[asset_idx])
                    self.asset_vis_list.append(self.asset_vis_list[asset_idx])
            elif 'eval' in object_root:
                difficulty = env_idx // 4
                complexity = env_idx % 4
                object_file = difficulties[difficulty] + str(complexity) + '_rescaled.urdf'
                print("Loading asset '%s' from '%s'" % (object_file, object_root))
                self.asset_object_list.append(self.gym.load_asset(self.sim, object_root, object_file, asset_options_object))
                self.asset_vis_list.append(self.gym.load_asset(self.sim, object_root, object_file, asset_options_vis))
            else:
                raise NotImplementedError

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
            pose_object.p = gymapi.Vec3(cfg.object.x, cfg.object.y, self.object_init_z)
            pose_object.r = gymapi.Quat(0, 0.0, 0.0, 1)
            color_green = gymapi.Vec3(0.1, 1.0, 0.1)

            # Create goal visualizations
            asset_vis = self.asset_vis_list[i]
            vis_handle = self.gym.create_actor(env=env, asset=asset_vis, pose=pose_object,
                                               name="vis" + str(i),
                                               group=self.num_envs,
                                               filter=1)
            self.gym.set_rigid_body_color(env, vis_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color_green)
            rigid_body_props = self.gym.get_actor_rigid_body_properties(env, vis_handle)
            rigid_body_props[0].mass = 1e-9
            self.gym.set_actor_rigid_body_properties(env, vis_handle, rigid_body_props)
            self.vis_handles.append(vis_handle)

            # Create object
            asset_object = self.asset_object_list[i]
            object_handle = self.gym.create_actor(env=env, asset=asset_object, pose=pose_object,
                                                  name="object" + str(i),
                                                  group=i,
                                                  filter=0)
            self.object_handles.append(object_handle)

        # Global rigid body ids
        self.global_vis_id = self.robot_side + self.robot_side ** 2 * 3
        self.global_obj_id = self.global_vis_id + 1

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

        # Update object position via contact points
        mean_corr_list = []
        for env_id in env_ids:
            contacts = self.gym.get_env_rigid_contacts(self.envs[env_id])
            if len(contacts) == 0:
                mean_corr_list.append(torch.tensor([self.robot_size / 2,
                                                    self.robot_size / 2,
                                                    self.object_half_extend + self.dof_range / 2]).float())
                continue
            contact_points = []
            for contact in contacts:
                ee_id = contact["body1"]
                assert ee_id != self.global_vis_id and ee_id != self.global_obj_id
                xx = ee_id // (self.robot_side * 3 + 1)
                yy = (ee_id - xx * (self.robot_side * 3 + 1) - 1) // 3
                x = xx * self.robot_row_gap + self.base_unit_pos[0]
                y = yy * self.robot_row_gap + self.base_unit_pos[1]
                z = self.dof_positions[env_id][xx, yy].float() + self.object_half_extend
                contact_points.append(torch.tensor([x, y, z]))
            mean_contact = torch.stack(contact_points).mean(dim=0)
            mean_corr_list.append(mean_contact)
        self.obj_pos_buf[env_ids] = torch.stack(mean_corr_list)

        # TODO: add disturbance
        # disturbance = torch.rand(object_xy.shape) * 0.1 - 0.05

        # Update centroids
        self.local_centroid_buf[env_ids] = torch.round((self.obj_pos_buf[env_ids] - self.base_unit_pos)
                                                       / self.robot_row_gap)[:, :2].int()

        # Object states
        obj_pos = self._normalize_object_position(self.obj_pos_buf[env_ids])  # [num_envs, 3]
        # if self.ori_obs:
        #     obj_ori = self.object_orientations  # [num_envs, 4]
        goal_pos = self._normalize_object_position(self.goal_pos_buf[env_ids])
        obj_diff_pos = self._normalize_diff_position(self.obj_pos_buf[env_ids] - self.goal_pos_buf[env_ids])  # [num_envs, 3]

        # Robot states
        local_tensors = []
        for env_id in env_ids:
            global_tensor = self.dof_positions[env_id]  # [num_side, num_side]
            # handling out of border
            x = min(max(self.local_centroid_buf[env_id][0], self.half_dim), self.robot_side - self.half_dim - 1)
            y = min(max(self.local_centroid_buf[env_id][1], self.half_dim), self.robot_side - self.half_dim - 1)
            local_tensors.append(
                global_tensor[x - self.half_dim:x + self.half_dim + 1, y - self.half_dim:y + self.half_dim + 1])
        robot_local = torch.stack(local_tensors, dim=0)     # [num_envs, local_side, local_side]
        robot_dct = self.dct_handler.dct(robot_local)  # [num_envs, dct_handler.n_freq]

        self.obs_buf[env_ids] = torch.cat([obj_pos, goal_pos, obj_diff_pos, robot_dct], dim=-1)
        return self.obs_buf

    def compute_reward(self):
        # TODO: check correctness
        trans_diff = torch.linalg.norm(self.object_positions_gt - self.goal_pos_buf, dim=-1)  # [num_envs]
        bonus = torch.where(trans_diff < torch.tensor([self.reach_threshold] * self.num_envs),
                            torch.tensor([self.reach_bonus] * self.num_envs), torch.tensor([0] * self.num_envs))
        self.rew_buf = (self.translation_diff_buf - trans_diff) * self.trans_delta_diff_factor + bonus
        self.translation_diff_buf = trans_diff

    def compute_reset(self):
        # TODO: check correctness
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
        goal_pos = xyz_dist.sample((len(env_ids),)).float()
        self.goal_pos_buf[env_ids] = goal_pos
        object_pos = xyz_dist.sample((len(env_ids),)).float()
        object_pos[:, 2] = self.object_init_z   # Note: z has to be high enough to avoid collision

        # reset root state for robots and objects in selected envs
        self.root_states[env_ids] = self.initial_root_states[env_ids]

        self.root_states[env_ids, -2, 0:3] = goal_pos
        self.root_states[env_ids, -1, 0:3] = object_pos

        actor_indices = self.all_actor_indices[env_ids].flatten()
        self.gym.set_actor_root_state_tensor_indexed(self.sim, self.root_tensor, gymtorch.unwrap_tensor(actor_indices),
                                                     len(actor_indices))

        # reset dof states for robots in selected envs
        self.dof_states[env_ids] = self.initial_dof_states[env_ids]
        robot_actor_indices = self.all_robot_actor_indices[env_ids].flatten()
        self.gym.set_dof_state_tensor_indexed(self.sim, self.dof_state_tensor, gymtorch.unwrap_tensor(robot_actor_indices),
                                              len(robot_actor_indices))
        self.dof_position_targets[env_ids] = self.dof_middle

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

        # reset object position, local centroid, observation, & translation diff buffer
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        # self.gym.refresh_force_sensor_tensor(self.sim)
        self.compute_observations(env_ids)
        self.translation_diff_buf[env_ids] = torch.linalg.norm(self.object_positions_gt[env_ids] - self.goal_pos_buf[env_ids], dim=-1)

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
                # handling out of border
                x = min(max(self.local_centroid_buf[env_id][0], self.half_dim), self.robot_side - self.half_dim - 1)
                y = min(max(self.local_centroid_buf[env_id][1], self.half_dim), self.robot_side - self.half_dim - 1)

                dof_target_local = self.dof_position_targets[env_id][x - self.half_dim:x + self.half_dim + 1,
                                                                     y - self.half_dim:y + self.half_dim + 1]
                local_target = dof_target_local + dof_shift * self.dt * self.action_speed_scale
                self.dof_position_targets[env_id] = self.dof_middle
                self.dof_position_targets[env_id][x - self.half_dim:x + self.half_dim + 1,
                                                  y - self.half_dim:y + self.half_dim + 1] = local_target

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
        # self.gym.refresh_force_sensor_tensor(self.sim)

        self.compute_observations()
        self.compute_reward()
        self.compute_reset()

        # # Immediate resets to reduce computational cost.
        # reset_env_ids = self.reset_buf.nonzero().squeeze(-1)
        # if len(reset_env_ids) > 0:
        #     self.reset_idx(reset_env_ids)


@hydra.main(version_base=None, config_path='waste/config', config_name='test_isaac_vectask')
def main(cfg):
    envs = ArrayRobot(cfg=cfg, rl_device='cuda:0', sim_device='cuda:0', graphics_device_id=0, headless=False)

    while True:
        pass


if __name__ == '__main__':
    main()