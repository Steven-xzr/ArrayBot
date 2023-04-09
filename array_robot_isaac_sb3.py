import hydra
import os
import sys
import numpy as np
import gym
from gym import spaces
import warnings
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Type, Union

from isaacgym import gymapi, gymutil, gymtorch
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

import torch
from dct_transform import BatchDCT


VecEnvIndices = Union[None, int, Iterable[int]]
VecEnvObs = Union[np.ndarray, Dict[str, np.ndarray], Tuple[np.ndarray, ...]]
VecEnvStepReturn = Tuple[VecEnvObs, np.ndarray, np.ndarray, List[Dict]]


class ArrayRobot(VecEnv):
    """
    The VecTask class in IsaacGymEnv does not support discrete action space.
    We wrap the Isaac Gym accelerated environment with the sb3 Base VecEnv class.
    sb3 defines numpy-format observation and action, requiring frequent transitions between numpy and pytorch.
    A more efficient way is to implement all the procedures in pytorch tensors.
    TODO: a fully pytorch implementation
    """
    def __init__(self, cfg):
        self.dct_handler = BatchDCT(cfg.dct.order, cfg.dim_local)
        self.half_dim = int((self.dct_handler.dim_local - 1) / 2)
        self.dct_step = cfg.dct.step
        self.dim_obs = cfg.dim_obs_obj + self.dct_handler.n_freq
        self.dim_action = self.dct_handler.n_freq * 2 + 1
        action_space = spaces.Discrete(self.dim_action)
        observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.dim_obs,))
        super().__init__(cfg.num_envs, observation_space, action_space)
        self.max_episode_steps = cfg.max_episode_steps

        # revised from VecTask
        self._init_sim(cfg.sim)
        self._add_viewer()
        self._set_envs(cfg.env)
        self._prepare_tensors()
        self._allocate_buffers()
        self._set_goal(cfg.goal)
        self._set_reward(cfg.reward)

    def _init_sim(self, cfg):
        """
        Initialize self.sim
        """
        self.gym = gymapi.acquire_gym()
        self.headless = cfg.headless
        self.sim_device = cfg.sim_device
        self.data_device = "cpu" if not cfg.use_gpu_pipeline else self.sim_device
        self.control_freq = cfg.control_freq

        # It seems Isaac only supports cuda:0
        sim_device_id = 0 if self.sim_device == "cpu" or self.sim_device == "cuda" else int(self.sim_device.split(":")[1])
        graphics_device_id = -1 if cfg.headless else sim_device_id

        sim_params = gymapi.SimParams()
        sim_params.dt = cfg.dt
        sim_params.substeps = cfg.substeps
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)

        # PhysX related
        sim_params.physx.use_gpu = cfg.sim_device != "cpu"
        sim_params.physx.num_threads = cfg.physx.num_threads
        sim_params.physx.solver_type = cfg.physx.solver_type
        sim_params.physx.num_position_iterations = cfg.physx.num_position_iterations
        sim_params.physx.num_velocity_iterations = cfg.physx.num_velocity_iterations
        sim_params.physx.contact_offset = cfg.physx.contact_offset
        sim_params.physx.rest_offset = cfg.physx.rest_offset

        # TODO: handle other sim params

        sim_params.use_gpu_pipeline = False
        physics_engine = gymapi.SIM_PHYSX
        print("WARNING: Forcing PhysX and CPU pipeline.")   # Otherwise, multiple bugs

        self.sim = self.gym.create_sim(sim_device_id, graphics_device_id, physics_engine, sim_params)

        # add ground plane
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.distance = 0.1
        self.gym.add_ground(self.sim, plane_params)

    def _add_viewer(self):
        # create viewer
        self.viewer = None
        if not self.headless:
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            if self.viewer is None:
                warnings.warn("Failed to create viewer!!!")
            self.enable_viewer_sync = True

            # position the camera
            cam_pos = gymapi.Vec3(-0.6, -0.6, 1.5)
            cam_target = gymapi.Vec3(0.6, 0.6, 0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def _set_envs(self, cfg):
        self._load_assets(cfg.assets)

        self.robot_row_gap = cfg.robot.row_gap
        self.robot_size = self.robot_row_gap * self.robot_side    # size per side

        num_envs_per_row = int(np.sqrt(self.num_envs))
        spacing = self.robot_side * self.robot_row_gap
        env_lower = gymapi.Vec3(-spacing / 2, -spacing / 2, -spacing / 2)
        env_upper = gymapi.Vec3(3 * spacing / 2, 3 * spacing / 2, spacing)

        self.envs = []
        self.robot_row_handles = []
        self.object_handles = []

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

            # TODO: randomly initialized object poses
            pose_object = gymapi.Transform()
            pose_object.p = gymapi.Vec3(cfg.object.x, cfg.object.y, self.object_half_extend + cfg.object.z)
            pose_object.r = gymapi.Quat(0, 0.0, 0.0, 1)
            object_handle = self.gym.create_actor(env=env, asset=self.asset_object, pose=pose_object,
                                                  name="object" + str(i),
                                                  group=i,
                                                  filter=0)
            self.object_handles.append(object_handle)

        rb_states = self.gym.get_actor_rigid_body_states(self.envs[0], 0, gymapi.STATE_POS)
        self.base_unit_pos = torch.tensor(rb_states[1][0][0].item())   # [3, ]
        next_unit_pos = torch.tensor(rb_states[4][0][0].item())
        assert next_unit_pos[1] - self.base_unit_pos[1] == self.robot_row_gap

    def _prepare_tensors(self):
        """
        Isaac Gym tensor API
        """
        self.gym.prepare_sim(self.sim)
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

        self.dof_position_targets = torch.zeros((self.num_envs, self.robot_side ** 2), dtype=torch.float32,
                                                device=self.data_device, requires_grad=False)
        self.all_actor_indices = torch.arange(self.num_envs * (self.robot_side + 1),
                                              dtype=torch.int32,
                                              device=self.data_device).view(self.num_envs, self.robot_side + 1)
        self.all_object_actor_indices = torch.arange(start=self.robot_side,
                                                     end=self.num_envs * (self.robot_side + 1),
                                                     step=self.robot_side + 1,
                                                     dtype=torch.int32,
                                                     device=self.data_device).view(self.num_envs, 1)
        self.all_robot_dof_indices = torch.arange(self.num_envs * self.robot_side ** 2,
                                                  dtype=torch.int32,
                                                  device=self.data_device).view(self.num_envs, self.robot_side ** 2)

    def _load_assets(self, cfg):
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
        self.init_asset_dof_states = np.zeros(self.robot_side, dtype=gymapi.DofState.dtype)
        self.init_asset_dof_states['pos'][:] = self.dof_middle

        object_root = os.path.join(os.path.expanduser("~"), cfg.object.root)
        object_file = cfg.object.file
        print("Loading asset '%s' from '%s'" % (object_file, object_root))
        self.asset_object = self.gym.load_asset(self.sim, object_root, object_file, gymapi.AssetOptions())
        self.object_half_extend = cfg.object.half_extend

    def _allocate_buffers(self):
        self.reset_buf = torch.ones(self.num_envs, device=self.data_device, dtype=torch.int32)
        self.progress_buf = torch.zeros(self.num_envs, device=self.data_device, dtype=torch.int32)
        self.rew_buf = torch.zeros(self.num_envs, device=self.data_device, dtype=torch.float32)
        self.obs_buf = torch.zeros(self.num_envs, self.dim_obs, device=self.data_device, dtype=torch.float32)
        self.local_centroid_buf = torch.zeros(self.num_envs, 2, device=self.data_device, dtype=torch.int32)
        self.translation_diff_buf = torch.zeros(self.num_envs, device=self.data_device, dtype=torch.float32)
        # self.rotation_diff_buf = torch.zeros(self.num_envs, device=self.data_device, dtype=torch.float32)

    def _set_goal(self, cfg):
        # TODO: set goals for orientations
        self.goal_pos = torch.tensor(cfg.pos, device=self.data_device)  # (3,)

    def _set_reward(self, cfg):
        self.trans_diff_factor = cfg.trans_diff_factor
        self.trans_delta_diff_factor = cfg.trans_delta_diff_factor
        self.reach_threshold = cfg.reach_threshold
        self.reach_bonus = cfg.reach_bonus

    def _reset_idx(self, env_ids):
        # TODO: support random initial pose of the object
        self.root_states[env_ids] = self.initial_root_states[env_ids]
        actor_indices = self.all_actor_indices[env_ids]
        self.gym.set_actor_root_state_tensor_indexed(self.sim, self.root_tensor, gymtorch.unwrap_tensor(actor_indices),
                                                     len(actor_indices))
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

        # TODO: reset trans_diff & rot_diff buffer

    def reset(self) -> VecEnvObs:
        for env_id in range(self.num_envs):
            self._reset_idx(env_id)

        # update obs & centroid buffer
        self.compute_observations()

        return self.obs_buf.numpy()

    def get_dof_shifts_from_actions(self, actions: torch.Tensor):
        """
        Compute the shifts of all the DOFs from the discrete action options on frequency domain.
        Pad the local actions to the global ones.
        We are unable to implement this function in batch operation because of different padding patterns in the batch.
        :param actions: [num_envs, 1]
        :return: dof_shifts: [num_envs, robot_side ** 2]
        """
        dof_shifts = []
        for env_id, action in enumerate(actions):
            dof_shifts.append(self.get_dof_shift_from_action(int(action), env_id))
        return torch.stack(dof_shifts, dim=0).reshape(self.num_envs, self.robot_side ** 2)

    def get_dof_shift_from_action(self, action: int, env_id: int):
        diff_freq = torch.zeros(self.dct_handler.n_freq)
        if action < self.dct_handler.n_freq * 2:
            diff_freq[action // 2] = 1 if action % 2 == 0 else -1
        diff_freq = self.dct_step * diff_freq
        local_dof_shift = self.dct_handler.idct(diff_freq.reshape(1, -1))    # [1, dim_local, dim_local]
        local_dof_shift = local_dof_shift.reshape(self.dct_handler.dim_local, self.dct_handler.dim_local)
        return self.pad_from_local(local_dof_shift, env_id)

    def pad_from_local(self, local_tensor: torch.Tensor, env_id: int):
        """
        :param local_tensor: [dim_local, dim_local]
        :param env_id: int
        :return: global_tensor: [robot_side, robot_side]
        """
        centroid = self.local_centroid_buf[env_id]  # [2]
        m = torch.nn.ReplicationPad2d((centroid[0] - self.half_dim, self.robot_side - centroid[0] - self.half_dim - 1,
                                       centroid[1] - self.half_dim, self.robot_side - centroid[1] - self.half_dim - 1))
        global_tensor = m(local_tensor.unsqueeze(0)).squeeze(0)
        assert global_tensor.shape == (self.robot_side, self.robot_side)
        return global_tensor

    def pre_physics_step(self, actions: torch.Tensor):
        """
        :param actions: [num_envs, ] discrete action options
        """
        reset_env_ids = self.reset_buf.nonzero().squeeze(-1)
        if len(reset_env_ids) > 0:
            self._reset_idx(reset_env_ids)

        dof_shifts = self.get_dof_shifts_from_actions(actions)
        # reset position targets for reset envs
        dof_shifts[reset_env_ids] = 0

        self.dof_position_targets = dof_shifts + self.dof_positions.reshape(self.num_envs, self.robot_side ** 2)
        self.dof_position_targets = torch.clamp(self.dof_position_targets, self.dof_lower_limit, self.dof_upper_limit)
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_position_targets))

    def post_physics_step(self):
        self.progress_buf += 1

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)

        self.compute_observations()
        self.compute_reward()

    def _get_norm_obj_pos(self):
        obj_pos = self.object_positions     # [num_envs, 3]

        # update centroids
        self.local_centroid_buf = torch.round((obj_pos - self.base_unit_pos) / self.robot_row_gap)[:, :2].long()

        # normalized position
        x = 2 * obj_pos[:, 0] / self.robot_size - 1
        y = 2 * obj_pos[:, 1] / self.robot_size - 1
        dof_z = self.dof_upper_limit - self.dof_lower_limit
        z = 2 * (obj_pos[:, 2] - self.object_half_extend - dof_z / 2) / dof_z
        return torch.vstack([x, y, z]).T

    def compute_observations(self):
        # object states
        obj_pos = self._get_norm_obj_pos()  # [num_envs, 3], update centroids
        obj_ori = self.object_orientations  # [num_envs, 4]
        obj_diff_pos = obj_pos - self.goal_pos  # [num_envs, 3]

        # robot states
        robot_local = self.select_from_global(self.dof_positions)  # [num_envs, local_side, local_side]
        robot_dct = self.dct_handler.dct(robot_local)  # [num_envs, dct_handler.n_freq]

        self.obs_buf = torch.cat([obj_pos, obj_ori, obj_diff_pos, robot_dct], dim=-1)

    def select_from_global(self, global_tensors: torch.Tensor):
        """
        :param global_tensors: [num_envs, robot_side, robot_side]
        :return: local_tensors: [num_envs, dim_local, dim_local]
        # TODO: batch implementation
        """
        local_tensors = []
        for env_id, global_tensor in enumerate(global_tensors):
            x = self.local_centroid_buf[env_id][0]
            y = self.local_centroid_buf[env_id][1]
            local_tensors.append(global_tensor[x - self.half_dim:x + self.half_dim + 1, y - self.half_dim:y + self.half_dim + 1])
        local_tensors = torch.stack(local_tensors, dim=0)
        assert local_tensors.shape == (self.num_envs, self.dct_handler.dim_local, self.dct_handler.dim_local)
        return local_tensors

    def compute_reward(self):
        """
        Compute reward and reset
        """
        trans_diff = torch.linalg.norm(self.object_positions - self.goal_pos, dim=-1)   # [num_envs]
        bonus = torch.where(trans_diff < torch.tensor([self.reach_threshold] * self.num_envs),
                            torch.tensor([self.reach_bonus] * self.num_envs), torch.tensor([0] * self.num_envs))
        self.rew_buf = (self.translation_diff_buf - trans_diff) * self.trans_delta_diff_factor + bonus
        self.translation_diff_buf = trans_diff

        timeout = torch.where(self.progress_buf >= self.max_episode_steps,
                              torch.ones(self.num_envs), torch.zeros(self.num_envs))
        x_out = torch.logical_or(self.object_positions[:, 0] < self.half_dim, self.object_positions[:, 0] >= self.robot_size - self.half_dim)
        y_out = torch.logical_or(self.object_positions[:, 1] < self.half_dim, self.object_positions[:, 1] >= self.robot_size - self.half_dim)
        self.reset_buf = torch.logical_or(torch.logical_or(x_out, y_out), timeout)

    def step(self, actions: np.ndarray):
        self.pre_physics_step(torch.tensor(actions, device=self.data_device))
        for i in range(self.control_freq):
            self.render()
            self.gym.simulate(self.sim)
        if self.data_device == 'cpu':
            self.gym.fetch_results(self.sim, True)
        self.post_physics_step()
        return self.obs_buf.numpy(), self.rew_buf.numpy(), self.reset_buf.numpy(), {}

    def close(self):
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)

    def render(self, mode="rgb_array"):
        """
        Draw the frame to the viewer, and check for keyboard events.
        The viewer is created when headless is False.
        It is called in step().
        """
        if self.viewer:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            # check for keyboard events
            # for evt in self.gym.query_viewer_action_events(self.viewer):
            #     if evt.action == "QUIT" and evt.value > 0:
            #         sys.exit()
            #     elif evt.action == "toggle_viewer_sync" and evt.value > 0:
            #         self.enable_viewer_sync = not self.enable_viewer_sync

            # fetch results
            if self.data_device != 'cpu':
                self.gym.fetch_results(self.sim, True)

            # step graphics
            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)

                # Wait for dt to elapse in real time.
                # This synchronizes the physics simulation with the rendering rate.
                self.gym.sync_frame_time(self.sim)

            else:
                self.gym.poll_viewer_events(self.viewer)

            # TODO: virtual display
            # if self.virtual_display and mode == "rgb_array":
            #     img = self.virtual_display.grab()
            #     return np.array(img)

    def seed(self, seed: Optional[int] = None) -> List[Union[None, int]]:
        # TODO: seed increments
        raise NotImplementedError

    def step_wait(self):
        # Not available when using Isaac
        warnings.warn("step_wait() is not available when using Isaac")

    def step_async(self, actions):
        # Not available when using Isaac
        warnings.warn("step_async() is not available when using Isaac")

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        # TODO: not sure whether it needs to be implemented
        warnings.warn("get_attr() is not available when using Isaac")

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        # TODO: not sure whether it needs to be implemented
        warnings.warn("set_attr() is not available when using Isaac")

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> List[Any]:
        # TODO: not sure whether it needs to be implemented
        warnings.warn("env_method() is not available when using Isaac")

    def env_is_wrapped(self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None) -> List[bool]:
        # TODO: not sure whether it needs to be implemented
        warnings.warn("env_is_wrapped() is not available when using Isaac")


@hydra.main(version_base=None, config_path='config', config_name='check_env_isaac')
def main(cfg):
    envs = ArrayRobot(cfg=cfg)

    print("Observation space is", envs.observation_space)
    print("Action space is", envs.action_space)
    obs = envs.reset()
    for _ in range(2000):
        obs, reward, done, info = envs.step(np.random.randint(0, envs.dim_action, envs.num_envs))

    print("done")


if __name__ == '__main__':
    main()