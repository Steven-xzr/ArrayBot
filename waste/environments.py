import pybullet as p
import random
import numpy as np
import os
import os.path as osp
import time
from abc import abstractmethod
import warnings
import gym
import gym.spaces as spaces
from scipy.spatial.transform import Rotation as R

from tablebot import TableBot
from dct_transform import DCT


class BaseEnv(gym.Env):
    """
    Local perception & action space in shape (dim_active, dim_active).
    """
    DOWN = 0
    NOOP = 1
    UP = 2

    def __init__(self, cfg):
        self.cfg = cfg
        self.robot = TableBot(cfg.env)
        self.max_steps = cfg.env.max_episode_steps
        # current_dir = os.path.dirname(os.path.realpath(__file__))
        self.obj_base_shift = np.array([cfg.object.shift * cfg.object.scale] * 3)
        obj_path = osp.join(os.path.expanduser('~'), cfg.object.visual_path)
        vis_shape = p.createVisualShape(p.GEOM_MESH, fileName=obj_path,
                                        meshScale=[cfg.object.scale] * 3,
                                        rgbaColor=[1, 0, 0, 0.9],
                                        visualFramePosition=self.obj_base_shift * -1
                                        )
        col_shape = p.createCollisionShape(p.GEOM_MESH, fileName=obj_path,
                                           meshScale=[cfg.object.scale] * 3,
                                           collisionFramePosition=self.obj_base_shift * -1
                                           )
        # self.object_id = p.loadURDF(cfg.object.path,
        #                             # cfg.object.position,
        #                             # p.getQuaternionFromEuler(cfg.object.orientation),
        #                             globalScaling=cfg.object.scale)
        self.object_id = p.createMultiBody(baseCollisionShapeIndex=col_shape, baseVisualShapeIndex=vis_shape,
                                           # baseInertialFramePosition=self.obj_base_shift
                                           )
        # print(p.getDynamicsInfo(self.object_id, -1))
        p.changeDynamics(self.object_id, -1, mass=cfg.object.mass,
                         lateralFriction=cfg.object.friction.lateral,
                         spinningFriction=cfg.object.friction.spinning,
                         rollingFriction=cfg.object.friction.rolling
                         )
        # center_corr = self.robot.table_size / 2
        # height = (self.robot.limit_lower + self.robot.limit_upper) / 2 + 0.1
        # self.object_position = [center_corr, center_corr, height]

        self.object_position = cfg.object.position
        self.object_orientation = p.getQuaternionFromEuler(cfg.object.orientation)
        # self.reset()
        # self.observation_space = spaces.Dict({
        #     'object_position': spaces.Box(low=-2, high=2, shape=(3,), dtype=np.float32),
        #     'object_orientation': spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32),
        #     'joint_position': spaces.Box(low=-1, high=1, shape=(self.robot.dim_active, self.robot.dim_active),
        #                                  dtype=np.float32),
        #     'diff_position': spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32),
        # })

        # To deal with the case that local perception areas goes out of the border.
        self.last_joint_pos = np.zeros((self.robot.dim_active, self.robot.dim_active), dtype=np.float32)

    def _get_local_centroid_idx(self):
        """
        Infer the idx of the centroid of local perception space from the object.
        """
        def get_idx_from_corr(xx):
            return int((xx - self.robot.act_margin - self.robot.act_size / 2) /
                       (self.robot.act_size + self.robot.act_gap))

        object_position, _ = p.getBasePositionAndOrientation(self.object_id)
        return tuple(map(get_idx_from_corr, object_position[:2]))

    def _get_norm_global_obj_pos(self):
        global_object_pos, _ = p.getBasePositionAndOrientation(self.object_id)
        x = 2 * global_object_pos[0] / self.robot.table_size - 1
        y = 2 * global_object_pos[1] / self.robot.table_size - 1
        z = 2 * (global_object_pos[2] - 0.04 - (self.robot.limit_upper - self.robot.limit_lower) / 2) \
            / (self.robot.limit_upper - self.robot.limit_lower)
        # NOTE: there is potentially a bug when normalizing z.
        return np.array([x, y, z]).astype(np.float32)

    def _get_norm_local_obj_pos(self, centroid):
        def get_corr_from_idx(idx):
            return self.robot.act_margin + self.robot.act_size / 2 + idx * (self.robot.act_size + self.robot.act_gap)

        global_object_pos, _ = p.getBasePositionAndOrientation(self.object_id)
        x = (global_object_pos[0] - get_corr_from_idx(centroid[0])) / (self.robot.act_size + self.robot.act_gap)
        y = (global_object_pos[1] - get_corr_from_idx(centroid[1])) / (self.robot.act_size + self.robot.act_gap)
        z = 2 * (global_object_pos[2] - 0.4 - self.robot.limit_lower) / (self.robot.limit_upper - self.robot.limit_lower)
        return np.array([x, y, z]).astype(np.float32)

    def _get_norm_obj_ori(self):
        _, object_orientation = p.getBasePositionAndOrientation(self.object_id)
        obj_ori = R.from_quat(object_orientation).as_rotvec()
        return (obj_ori / np.pi).astype(np.float32)

    def _get_norm_local_joint_pos(self, centroid):
        x_low = centroid[0] - self.robot.half_dim
        x_high = centroid[0] + self.robot.half_dim + 1
        y_low = centroid[1] - self.robot.half_dim
        y_high = centroid[1] + self.robot.half_dim + 1
        if x_low < 0 or x_high > self.robot.num_side or y_low < 0 or y_high > self.robot.num_side:
            # warnings.warn("Local perception area goes out of the border.")
            return self.last_joint_pos.astype(np.float32)
        else:
            joint_position, _ = self.robot.get_states()
            pos = joint_position[x_low:x_high, y_low:y_high]
            assert pos.shape == (self.robot.dim_active, self.robot.dim_active)
            pos = 2 * (pos - self.robot.limit_lower) / (self.robot.limit_upper - self.robot.limit_lower) - 1
            self.last_joint_pos = pos
            return pos.astype(np.float32)

    def _pad_from_local(self, local_array):
        centroid = self._get_local_centroid_idx()
        global_array = np.pad(local_array, ((centroid[0] - self.robot.half_dim,
                                             self.robot.num_side - centroid[0] - self.robot.half_dim - 1),
                                            (centroid[1] - self.robot.half_dim,
                                             self.robot.num_side - centroid[1] - self.robot.half_dim - 1)),
                              # mode='constant', constant_values=-1
                              mode='edge'
                              )
        assert global_array.shape == (self.robot.num_side, self.robot.num_side)
        return global_array

    @abstractmethod
    def _take_action(self, action: np.ndarray):
        raise NotImplementedError

    @abstractmethod
    def _get_reward(self, action: np.ndarray):
        raise NotImplementedError

    def _is_done(self):
        centroid = self._get_local_centroid_idx()
        return not (self.robot.half_dim <= centroid[0] < self.robot.num_side - self.robot.half_dim and
                    self.robot.half_dim <= centroid[1] < self.robot.num_side - self.robot.half_dim)

    def _reset_robot(self):
        self.robot.reset()
        # center_corr = self.robot.table_size / 2
        # height = (self.robot.limit_lower + self.robot.limit_upper) / 2 + 0.1
        if self.cfg.object.random_position:
            raise NotImplementedError
        else:
            object_corr = self.object_position
        p.resetBasePositionAndOrientation(self.object_id, object_corr, self.object_orientation)
        for _ in range(20):
            p.stepSimulation()
        # return self._get_obs()

    @abstractmethod
    def reset(self, seed=0, options=None):
        raise NotImplementedError

    @abstractmethod
    def _get_obs(self):
        raise NotImplementedError

    @abstractmethod
    def _get_info(self):
        raise NotImplementedError

    def step(self, action: np.ndarray):
        self._take_action(action)
        obs = self._get_obs()
        reward = self._get_reward(action)
        done = self._is_done()
        info = self._get_info()
        return obs, reward, done, False, info

    # def sample_action(self):
    #     return np.random.randint(-1, 2, [self.robot.num_side, self.robot.num_side])

    def render(self, mode='rgb'):
        """
        Do not use the render function. Use the gui mode in pybullet instead.
        """
        warnings.warn('Do not use the render function. Use the gui mode in pybullet instead.')
        # if mode == 'rgb':
        #     return self._get_obs()['rgb']
        # elif mode == 'depth':
        #     return self._get_obs()['depth']
        # else:
        #     raise NotImplementedError


# class BaseEnvSpatial(BaseEnv):
#     def __init__(self, cfg):
#         super().__init__(cfg)
#         self.action_space = spaces.MultiDiscrete(np.ones([self.robot.num_side ** 2], dtype=np.int8) * 3)
#
#     def _take_action(self, action: np.ndarray):
#         """
#         The elements of action should be 2 for UP, 1 for NOOP, 0 for DOWN
#         """
#         assert action.shape == (self.robot.num_act,) or action.shape == (self.robot.num_side, self.robot.num_side)
#         if action.shape == (self.robot.num_act,):
#             action = action.reshape((self.robot.num_side, self.robot.num_side))
#         action = (action - 1).astype(np.int8)
#         self.robot.take_action(action)
#
#     @abstractmethod
#     def _get_reward(self, action: np.ndarray):
#         raise NotImplementedError


class ManiEnvDCT(BaseEnv):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.dct_handler = DCT(cfg.dct.order, cfg.env.robot.tablebot.dim_local)
        self.dct_step = cfg.dct.step
        self.action_space = spaces.Discrete(self.dct_handler.n_freq * 2 + 1)
        self.observation_space = spaces.Dict({
            'object_position': spaces.Box(low=-2, high=2, shape=(3,), dtype=np.float32),
            'object_orientation': spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32),
            # 'joint_position': spaces.Box(low=-1, high=1, shape=(self.robot.dim_active, self.robot.dim_active),
            #                              dtype=np.float32),
            'diff_position': spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32),
            'joint_position': spaces.Box(low=-1, high=1, shape=(self.dct_handler.n_freq,), dtype=np.float32)
        })

        # visualize the goal configuration
        self.goal_pos = np.array([cfg.goal.x, cfg.goal.y, cfg.goal.z])
        goal_ori_euler = np.array([cfg.goal.row, cfg.goal.pitch, cfg.goal.yaw])
        self.goal_ori = R.from_euler('xyz', goal_ori_euler)
        vis_shape = p.createVisualShape(p.GEOM_MESH, fileName=cfg.object.visual_path,
                                        meshScale=[cfg.object.scale] * 3,
                                        rgbaColor=[0, 1, 0, 0.5],
                                        visualFramePosition=self.obj_base_shift * -1)
        p.createMultiBody(0, -1, vis_shape, basePosition=self.goal_pos,
                          baseOrientation=self.goal_ori.as_quat())

        # reward
        self.last_translation_diff = -1
        self.last_rotation_diff = -1
        self.t_diff_factor = cfg.reward.t_diff_factor
        self.r_diff_factor = cfg.reward.r_diff_factor
        self.t_delta_diff_factor = cfg.reward.t_delta_diff_factor
        self.r_delta_diff_factor = cfg.reward.r_delta_diff_factor
        self.reach_threshold = cfg.reward.reach_threshold
        self.reach_bonus = cfg.reward.reach_bonus

    def _take_action(self, action: int):
        diff_freq = np.zeros(self.dct_handler.n_freq)
        if action < self.dct_handler.n_freq * 2:
            diff_freq[action // 2] = 1 if action % 2 == 0 else -1
        diff_freq = self.dct_step * diff_freq
        normalized_diff = self._pad_from_local(self.dct_handler.idct(diff_freq))
        self.robot.set_normalized_diff(normalized_diff)

    @abstractmethod
    def _get_reward(self, action: np.ndarray):
        raise NotImplementedError

    @abstractmethod
    def _get_info(self):
        raise NotImplementedError

    def _get_norm_diff_pos(self):
        obj_pos, _ = p.getBasePositionAndOrientation(self.object_id)
        diff_pos = obj_pos - self.goal_pos
        return diff_pos.astype(np.float32) / self.robot.table_size

    def _get_obs(self):
        # Warning: lacking handlers for object_orientation & joint_velocity (currently not used)
        """
        All positions and pixel values are normalized to [-1, 1]
        """
        centroid = self._get_local_centroid_idx()

        return {
            # 'object_position': self._get_norm_local_obj_pos(centroid),
            'object_position': self._get_norm_global_obj_pos(),
            'object_orientation': self._get_norm_obj_ori(),
            'diff_position': self._get_norm_diff_pos(),
            'joint_position': self.dct_handler.dct(self._get_norm_local_joint_pos(centroid)),
        }

    def reset(self, seed=0, options=None):
        # print('reset called')
        self._reset_robot()
        self.last_translation_diff = self._translation_diff()
        self.last_rotation_diff = self._rotation_diff()
        return self._get_obs(), {}

    def _translation_diff(self):
        return np.linalg.norm(p.getBasePositionAndOrientation(self.object_id)[0] - self.goal_pos)

    def _rotation_diff(self):
        obj_ori = R.from_quat(p.getBasePositionAndOrientation(self.object_id)[1])
        diff_ori = obj_ori.inv() * self.goal_ori
        return np.linalg.norm(diff_ori.as_rotvec())

#
# class LiftEnvDCT(ManiEnvDCT):
#     def __init__(self, cfg):
#         super().__init__(cfg)
#
#     def _get_reward(self, action):
#         object_position, _ = p.getBasePositionAndOrientation(self.object_id)
#         object_height = object_position[2] - 0.01
#         # normalize the reward to [-1, 1]
#         reward = 2 * (object_height - self.robot.limit_lower) / (self.robot.limit_upper - self.robot.limit_lower) - 1
#         return reward
#
#
# class RotateEnvDCT(ManiEnvDCT):
#     def __init__(self, cfg):
#         super().__init__(cfg)
#
#     def _get_reward(self, action):
#         return {"rot_reward": - self._rotation_diff()}
#


class TransEnvDCT(ManiEnvDCT):
    def __init__(self, cfg):
        super().__init__(cfg)

    def _get_reward(self, action):
        info_dict = self._get_info()
        t_diff = info_dict['t_diff']
        t_delta_diff = info_dict['t_delta_diff']
        bonus = info_dict['bonus']
        return t_delta_diff + bonus

    def _get_info(self):
        t_diff = self._translation_diff()
        delta_t_diff = t_diff - self.last_translation_diff
        self.last_translation_diff = t_diff
        bonus = 0
        if t_diff < self.reach_threshold:
            # print("Reach the goal!!!!!")
            bonus = self.reach_bonus
        return {"t_diff": float(self.t_diff_factor / (t_diff + 1e-6)),
                "t_delta_diff": float(- delta_t_diff * self.t_delta_diff_factor),
                "bonus": bonus}

#
# class SE3EnvDCT(ManiEnvDCT):
#     def __init__(self, cfg):
#         super().__init__(cfg)
#
#     def _get_reward(self, action):
#         t_diff = self._translation_diff()
#         r_diff = self._rotation_diff()
#         delta_t_diff = t_diff - self.last_translation_diff
#         delta_r_diff = r_diff - self.last_rotation_diff
#         self.last_translation_diff = t_diff
#         self.last_rotation_diff = r_diff
#         return {"rot_reward": - delta_r_diff * self.rotation_factor,
#                 "trans_reward": - delta_t_diff * self.translation_factor,
#                                 # - t_diff,
#                 "trans_diff": t_diff,
#                 }
#
#
# class LiftEnv(BaseEnvSpatial):
#     def __init__(self, cfg):
#         super(LiftEnv, self).__init__(cfg)
#
#     def _get_reward(self, action):
#         object_position, _ = p.getBasePositionAndOrientation(self.object_id)
#         object_height = object_position[2] - 0.01
#         # normalize the reward to [-1, 1]
#         reward = 2 * (object_height - self.robot.limit_lower) / (self.robot.limit_upper - self.robot.limit_lower) - 1
#         if self.cfg.reward.regularization:
#             regularization = self.cfg.reward.coefficient * self.robot.regularization(self.cfg.reward.mode, action)
#             return reward - regularization
#         return reward
#
#
# class RotateEnv(BaseEnvSpatial):
#     def __init__(self, cfg):
#         super(RotateEnv, self).__init__(cfg)
#         self.goal_ori = np.array([cfg.goal.row, cfg.goal.pitch, cfg.goal.yaw])
#
#     def _get_reward(self, action):
#         _, obj_ori = p.getBasePositionAndOrientation(self.object_id)
#         obj_ori = np.array(p.getEulerFromQuaternion(obj_ori))
#         # ori_diff = np.linalg.norm(obj_ori - self.goal_ori)
#         ori_diff = np.abs((obj_ori - self.goal_ori)[-1])
#         if self.cfg.reward.regularization:
#             regularization = self.cfg.reward.coefficient * self.robot.regularization(self.cfg.reward.mode, action)
#             return - ori_diff - regularization
#         return - ori_diff
