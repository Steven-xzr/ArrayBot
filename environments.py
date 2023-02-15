import pybullet as p
import random
import numpy as np
import os
import os.path as osp
from abc import abstractmethod
import warnings
import gym
import gym.spaces as spaces
from scipy.spatial.transform import Rotation as R

from tablebot import TableBot
from dct_transform import DCT


class BaseEnv(gym.Env):
    """
    To fit the gym and RL framework api, action in the environment is in the shape of (num_act,).
    However, the action in the robot is in the shape of (num_side, num_side).
    """
    DOWN = 0
    NOOP = 1
    UP = 2

    def __init__(self, cfg):
        self.cfg = cfg
        self.robot = TableBot(cfg.tablebot)
        current_dir = os.path.dirname(os.path.realpath(__file__))
        self.object_id = p.loadURDF(cfg.object.path,
                                    # cfg.object.position,
                                    # p.getQuaternionFromEuler(cfg.object.orientation),
                                    globalScaling=cfg.object.scale)
        p.changeDynamics(self.object_id, -1, mass=cfg.object.mass,
                         lateralFriction=cfg.object.friction.lateral,
                         spinningFriction=cfg.object.friction.spinning,
                         rollingFriction=cfg.object.friction.rolling)
        center_corr = self.robot.table_size / 2
        height = (self.robot.limit_lower + self.robot.limit_upper) / 2 + 0.1
        self.object_position = [center_corr, center_corr, height]
        self.object_orientation = p.getQuaternionFromEuler(cfg.object.orientation)
        self.reset()
        self.observation_space = spaces.Dict({
            'object_position': spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32),
            'object_orientation': spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32),
            'joint_position': spaces.Box(low=-1, high=1, shape=(self.robot.num_side, self.robot.num_side),
                                         dtype=np.float32),
        })
        # self.action_space = spaces.Box(low=-1, high=1, shape=(self.robot.num_act,), dtype=np.int8)
        # self.action_space = spaces.MultiDiscrete(np.ones([self.robot.num_side, self.robot.num_side], dtype=np.int8) * 3)

    def _get_obs(self):
        # Warning: lacking handlers for object_orientation & joint_velocity (currently not used)
        """
        All positions and pixel values are normalized to [-1, 1]
        """
        def norm_obj_pos(obj_pos: list):
            x = 2 * obj_pos[0] / self.robot.table_size - 1
            y = 2 * obj_pos[1] / self.robot.table_size - 1
            z = 2 * (obj_pos[2] - self.robot.limit_lower) / (self.robot.limit_upper - self.robot.limit_lower) - 1
            return np.array([x, y, z]).astype(np.float32)

        def norm_obj_ori(obj_ori: np.ndarray):
            return (obj_ori / np.pi).astype(np.float32)

        def norm_image(image: np.ndarray):
            image = 2 * (image - np.amin(image)) / (np.amax(image) - np.amin(image)) - 1
            return image.astype(np.float32)

        def norm_joint_pos(joint_pos: np.ndarray):
            pos = 2 * (joint_pos - self.robot.limit_lower) / (self.robot.limit_upper - self.robot.limit_lower) - 1
            return pos.astype(np.float32)

        object_position, object_orientation = p.getBasePositionAndOrientation(self.object_id)
        object_orientation = R.from_quat(object_orientation).as_rotvec()
        rgb, depth = self.robot.get_images()
        joint_position, joint_velocity = self.robot.get_states()

        return {
            'object_position': norm_obj_pos(object_position),
            'object_orientation': norm_obj_ori(object_orientation),
            # 'rgb': norm_image(rgb), 'depth': norm_image(depth),
            'joint_position': norm_joint_pos(joint_position),
            # 'joint_velocity': joint_velocity
        }

    @abstractmethod
    def _take_action(self, action: np.ndarray):
        raise NotImplementedError

    @abstractmethod
    def _get_reward(self, action: np.ndarray):
        raise NotImplementedError

    def _is_done(self):
        object_position, _ = p.getBasePositionAndOrientation(self.object_id)
        inside = 0 < object_position[0] < self.robot.table_size and 0 < object_position[1] < self.robot.table_size
        return not inside

    def reset(self):
        self.robot.reset()
        # center_corr = self.robot.table_size / 2
        # height = (self.robot.limit_lower + self.robot.limit_upper) / 2 + 0.1
        if self.cfg.object.random_position:
            raise NotImplementedError
        else:
            object_corr = self.object_position
        p.resetBasePositionAndOrientation(self.object_id, object_corr, self.object_orientation)
        for _ in range(100):
            p.stepSimulation()
        return self._get_obs()

    def step(self, action: np.ndarray):
        self._take_action(action)
        obs = self._get_obs()
        reward = self._get_reward(action)
        done = self._is_done()
        return obs, reward, done, {}

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


class BaseEnvSpatial(BaseEnv):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.action_space = spaces.MultiDiscrete(np.ones([self.robot.num_side ** 2], dtype=np.int8) * 3)

    def _take_action(self, action: np.ndarray):
        """
        The elements of action should be 2 for UP, 1 for NOOP, 0 for DOWN
        """
        assert action.shape == (self.robot.num_act,) or action.shape == (self.robot.num_side, self.robot.num_side)
        if action.shape == (self.robot.num_act,):
            action = action.reshape((self.robot.num_side, self.robot.num_side))
        action = (action - 1).astype(np.int8)
        self.robot.take_action(action)

    @abstractmethod
    def _get_reward(self, action: np.ndarray):
        raise NotImplementedError


class ManiEnvDCT(BaseEnv):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.dct_handler = DCT(cfg.dct.order)
        self.dct_step = cfg.dct.step
        self.action_space = spaces.Discrete(self.dct_handler.n_freq * 2)

        # visualize the goal configuration
        self.goal_pos = np.array([cfg.goal.x, cfg.goal.y, cfg.goal.z])
        goal_ori_euler = np.array([cfg.goal.row, cfg.goal.pitch, cfg.goal.yaw])
        self.goal_ori = R.from_euler('xyz', goal_ori_euler)
        vis_shape = p.createVisualShape(p.GEOM_MESH, fileName=cfg.object.visual_path,
                                        meshScale=[cfg.object.scale * 0.05] * 3,
                                        rgbaColor=[0, 1, 0, 0.5])
        vis_goal = p.createMultiBody(0, -1, vis_shape, basePosition=self.goal_pos,
                                     baseOrientation=self.goal_ori.as_quat())

    def _take_action(self, action: int):
        diff_freq = np.zeros(self.dct_handler.n_freq)
        if action < self.dct_handler.n_freq * 2:
            diff_freq[action // 2] = 1 if action % 2 == 0 else -1
        diff_freq = self.dct_step * diff_freq
        self.robot.set_normalized_diff(self.dct_handler.idct(diff_freq))

    @abstractmethod
    def _get_reward(self, action: np.ndarray):
        raise NotImplementedError

    def _translation_diff(self):
        return np.linalg.norm(p.getBasePositionAndOrientation(self.object_id)[0] - self.goal_pos)

    def _rotation_diff(self):
        obj_ori = R.from_quat(p.getBasePositionAndOrientation(self.object_id)[1])
        diff_ori = obj_ori.inv() * self.goal_ori
        return np.linalg.norm(diff_ori.as_rotvec())


class LiftEnvDCT(ManiEnvDCT):
    def __init__(self, cfg):
        super().__init__(cfg)

    def _get_reward(self, action):
        object_position, _ = p.getBasePositionAndOrientation(self.object_id)
        object_height = object_position[2] - 0.01
        # normalize the reward to [-1, 1]
        reward = 2 * (object_height - self.robot.limit_lower) / (self.robot.limit_upper - self.robot.limit_lower) - 1
        return reward


class RotateEnvDCT(ManiEnvDCT):
    def __init__(self, cfg):
        super().__init__(cfg)

    def _get_reward(self, action):
        return {"rot_reward": - self._rotation_diff()}


class TransEnvDCT(ManiEnvDCT):
    def __init__(self, cfg):
        super().__init__(cfg)

    def _get_reward(self, action):
        return {"trans_reward": - self._translation_diff()}


class SE3EnvDCT(ManiEnvDCT):
    def __init__(self, cfg):
        super().__init__(cfg)

    def _get_reward(self, action):
        return {"rot_reward": - self._rotation_diff(),
                "trans_reward": - self._translation_diff() * 50}


class LiftEnv(BaseEnvSpatial):
    def __init__(self, cfg):
        super(LiftEnv, self).__init__(cfg)

    def _get_reward(self, action):
        object_position, _ = p.getBasePositionAndOrientation(self.object_id)
        object_height = object_position[2] - 0.01
        # normalize the reward to [-1, 1]
        reward = 2 * (object_height - self.robot.limit_lower) / (self.robot.limit_upper - self.robot.limit_lower) - 1
        if self.cfg.reward.regularization:
            regularization = self.cfg.reward.coefficient * self.robot.regularization(self.cfg.reward.mode, action)
            return reward - regularization
        return reward


class RotateEnv(BaseEnvSpatial):
    def __init__(self, cfg):
        super(RotateEnv, self).__init__(cfg)
        self.goal_ori = np.array([cfg.goal.row, cfg.goal.pitch, cfg.goal.yaw])

    def _get_reward(self, action):
        _, obj_ori = p.getBasePositionAndOrientation(self.object_id)
        obj_ori = np.array(p.getEulerFromQuaternion(obj_ori))
        # ori_diff = np.linalg.norm(obj_ori - self.goal_ori)
        ori_diff = np.abs((obj_ori - self.goal_ori)[-1])
        if self.cfg.reward.regularization:
            regularization = self.cfg.reward.coefficient * self.robot.regularization(self.cfg.reward.mode, action)
            return - ori_diff - regularization
        return - ori_diff
