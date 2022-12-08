import pybullet as p
import random
import numpy as np
import os
import os.path as osp

from tablebot import TableBot


class BaseEnv:
    def __init__(self, cfg):
        self.cfg = cfg
        self.robot = TableBot(cfg.tablebot)
        current_dir = os.path.dirname(os.path.realpath(__file__))
        self.object_id = p.loadURDF(osp.join(current_dir, cfg.object.path), cfg.object.position)
        self.reset()

    def get_obs(self):
        rgb, depth = self.robot.get_images()
        joint_position, joint_velocity = self.robot.get_states()
        return {'rgb': rgb, 'depth': depth, 'joint_position': joint_position, 'joint_velocity': joint_velocity}

    def get_reward(self):
        raise NotImplementedError

    def is_done(self):
        object_position, _ = p.getBasePositionAndOrientation(self.object_id)
        inside = 0 < object_position[0] < self.robot.table_size and 0 < object_position[1] < self.robot.table_size
        return not inside

    def reset(self):
        self.robot.reset()
        center_corr = self.robot.table_size / 2
        height = (self.robot.limit_lower + self.robot.limit_upper) / 2 + 0.1
        if self.cfg.object.random_position:
            object_corr = [center_corr + (random.random() - 0.5) * self.robot.table_size * 0.8,
                           center_corr + (random.random() - 0.5) * self.robot.table_size * 0.8,
                           height]
        else:
            object_corr = [center_corr, center_corr, height]
        p.resetBasePositionAndOrientation(self.object_id, object_corr, [0, 0, 0, 1])
        for _ in range(100):
            p.stepSimulation()
        return self.get_obs()

    def step(self, action: np.ndarray):
        """
        The elements of action should be 1 for UP, 0 for NOOP, -1 for DOWN
        """
        self.robot.take_action(action)
        obs = self.get_obs()
        reward = self.get_reward()
        done = self.is_done()
        return obs, reward, done, {}


class LiftBlockEnv(BaseEnv):
    def __init__(self, cfg):
        super(LiftBlockEnv, self).__init__(cfg)

    def get_reward(self):
        object_position, _ = p.getBasePositionAndOrientation(self.object_id)
        object_height = object_position[2]
        reward = object_height - (self.robot.limit_lower + self.robot.limit_upper) / 2
        return reward

