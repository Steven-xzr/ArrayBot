import pybullet as p
import pybullet_data
import hydra
import random
import time
import math
import numpy as np
import os
import os.path as osp
import matplotlib.pyplot as plt
from enum import Enum


class Action(Enum):
    DOWN = -1
    STOP = 0
    UP = 1


class TableBot:
    def __init__(self, cfg):
        self._setup_robot(cfg.robot)
        self._setup_camera(cfg.camera)

    def _setup_robot(self, cfg):
        p.connect(p.GUI if cfg.gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        current_dir = os.path.dirname(os.path.realpath(__file__))
        p.loadURDF(cfg.plane.path, cfg.plane.position, useFixedBase=True)
        self.robot_id = p.loadURDF(osp.join(current_dir, cfg.tablebot.path), cfg.tablebot.position, useFixedBase=True)
        # p.resetBasePositionAndOrientation(self.robot_id, [0, 0, 0], [0, 0, 0, 1])

        self.num_act = p.getNumJoints(self.robot_id)
        self.num_side = int(math.sqrt(self.num_act))
        joint_info = p.getJointInfo(self.robot_id, 0)
        self.limit_lower = joint_info[8]
        self.limit_upper = joint_info[9]
        self.act_size = cfg.tablebot.act_size   # size of a side
        self.act_gap = cfg.tablebot.act_gap
        self.act_margin = cfg.tablebot.act_margin
        self.table_size = self.act_size * self.num_side + (self.num_side - 1) * self.act_gap + 2 * self.act_margin

    def _setup_camera(self, cfg):
        self.width = cfg.width  # in pixels
        self.height = cfg.height  # in pixels
        eye_position = [self.table_size / 2, self.table_size / 2, cfg.distance]
        target_position = [self.table_size / 2, self.table_size / 2, 0]
        up_vector = [0, 1, 0]
        self.view_matrix = p.computeViewMatrix(eye_position, target_position, up_vector)
        self.proj_matrix = p.computeProjectionMatrixFOV(cfg.fov, self.width / self.height, cfg.near, cfg.far)

    def idx2corr(self, index):
        remain = index % self.num_side
        times = index // self.num_side
        y = self.num_side - remain - 1
        x = self.num_side - times - 1
        assert -1 < x < self.num_side and -1 < y < self.num_side
        return np.array([x, y])

    def corr2idx(self, corr):
        x = corr[0]
        y = corr[1]
        assert -1 < x < self.num_side and -1 < y < self.num_side
        return self.num_side - y - 1 + (self.num_side - x - 1) * self.num_side

    def list2array(self, list_obj: list):
        assert len(list_obj) == self.num_act
        array_obj = np.empty([self.num_side, self.num_side])
        for idx, obj in enumerate(list_obj):
            x, y = self.idx2corr(idx)
            array_obj[x, y] = obj
        return array_obj

    def array2list(self, array_obj: np.array):
        assert array_obj.shape == (self.num_side, self.num_side)
        list_obj = np.empty(self.num_act)
        for x in range(self.num_side):
            for y in range(self.num_side):
                list_obj[self.corr2idx([x, y])] = array_obj[x, y]
        return list_obj.tolist()

    def get_observations(self):
        width, height, rgb, depth, mask = p.getCameraImage(self.width, self.height, self.view_matrix, self.proj_matrix)
        return rgb, depth

    def get_states(self):
        position = []
        velocity = []
        states = p.getJointStates(self.robot_id, range(self.num_act))
        for s in states:
            position.append(s[0])
            velocity.append(s[1])
        return self.list2array(position), self.list2array(velocity)

    def set_states(self, target_position: np.array, steps=100):
        current_position, _ = self.get_states()
        for t in range(steps):
            position = current_position + (target_position - current_position) * (t + 1) / steps
            list_position = self.array2list(position)
            p.setJointMotorControlArray(self.robot_id, range(self.num_act), p.POSITION_CONTROL, targetPositions=list_position)
            for _ in range(100):
                p.stepSimulation()

    def set_random_states(self):
        target_position = np.random.uniform(self.limit_lower, self.limit_upper, [self.num_side, self.num_side])
        self.set_states(target_position)
        return target_position




@hydra.main(config_path='config', config_name='demo')
def main(cfg):
    robot = TableBot(cfg)

    # list_obj = [0] * 16
    # list_obj[1] = 1
    # array_obj = robot.list2array(list_obj)
    # list_obj_ = robot.array2list(array_obj)

    target = robot.set_random_states()
    now = robot.get_states()[0]
    print(now - target)

    rgb, depth = robot.get_observations()
    plt.imshow(rgb)
    plt.show()
    plt.imshow(depth)
    plt.show()


if __name__ == '__main__':
    main()
