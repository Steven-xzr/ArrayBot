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
        # Warning: lacking handlers for object_orientation & joint_velocity (currently not used)
        """
        All positions and pixel values are normalized to [-1, 1]
        """
        def norm_obj_pos(obj_pos: list):
            x = 2 * obj_pos[0] / self.robot.table_size - 1
            y = 2 * obj_pos[1] / self.robot.table_size - 1
            z = 2 * (obj_pos[2] - 0.01 - self.robot.limit_lower) / (self.robot.limit_upper - self.robot.limit_lower) - 1
            return np.array([x, y, z])

        def norm_image(image: np.ndarray):
            return 2 * (image - np.amin(image)) / (np.amax(image) - np.amin(image)) - 1

        def norm_joint_pos(joint_pos: np.ndarray):
            return 2 * (joint_pos - self.robot.limit_lower) / (self.robot.limit_upper - self.robot.limit_lower) - 1

        object_position, object_orientation = p.getBasePositionAndOrientation(self.object_id)
        rgb, depth = self.robot.get_images()
        joint_position, joint_velocity = self.robot.get_states()

        return {'object_position': norm_obj_pos(object_position), 'object_orientation': np.array(object_orientation),
                'rgb': norm_image(rgb), 'depth': norm_image(depth),
                'joint_position': norm_joint_pos(joint_position), 'joint_velocity': joint_velocity}

    def get_reward(self, action: np.ndarray):
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
        reward = self.get_reward(action)
        done = self.is_done()
        return obs, reward, done, {}


class LiftEnv(BaseEnv):
    def __init__(self, cfg):
        super(LiftEnv, self).__init__(cfg)

    def get_reward(self, action):
        object_position, _ = p.getBasePositionAndOrientation(self.object_id)
        object_height = object_position[2] - 0.01
        # normalize the reward to [-1, 1]
        reward = 2 * (object_height - self.robot.limit_lower) / (self.robot.limit_upper - self.robot.limit_lower) - 1
        if self.cfg.reward.regularization:
            regularization = self.cfg.reward.coefficient * self.robot.regularization(self.cfg.reward.mode, action)
            return reward - regularization
        return reward

