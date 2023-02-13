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
import warnings


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
        p.setGravity(0, 0, -10)

        # Every actuator consists of 3 joints (base -- box -- cylinder -- sphere) or 1 joint (base -- box)
        self.num_joint_per_act = cfg.tablebot.num_joint_per_act
        self.num_act = p.getNumJoints(self.robot_id) // self.num_joint_per_act
        self.num_side = int(math.sqrt(self.num_act))
        self.joint_iter = range(0, self.num_act * self.num_joint_per_act, self.num_joint_per_act)

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

    def array2list(self, array_obj: np.ndarray):
        assert array_obj.shape == (self.num_side, self.num_side)
        list_obj = np.empty(self.num_act)
        for x in range(self.num_side):
            for y in range(self.num_side):
                list_obj[self.corr2idx([x, y])] = array_obj[x, y]
        return list_obj.tolist()

    def get_images(self, with_gt_mask=False):
        width, height, rgb, depth, mask = p.getCameraImage(self.width, self.height, self.view_matrix, self.proj_matrix)
        if with_gt_mask:
            warnings.warn("Acquiring ground-truth information!")
            return rgb, depth, mask
        else:
            return rgb, depth

    def get_states(self):
        position = []
        velocity = []
        states = p.getJointStates(self.robot_id, self.joint_iter)
        for s in states:
            position.append(s[0])
            velocity.append(s[1])
        return self.list2array(position), self.list2array(velocity)

    def set_states(self, target_position: np.ndarray, interp_steps=20, sim_steps=50):
        assert target_position.shape == (self.num_side, self.num_side)
        # consider the joint limits
        target_position = np.clip(target_position, self.limit_lower, self.limit_upper)

        current_position, _ = self.get_states()
        # interpolate intermediate positions so that the robot moves smoothly
        for t in range(interp_steps):
            position = current_position + (target_position - current_position) * (t + 1) / interp_steps
            list_position = self.array2list(position)
            p.setJointMotorControlArray(self.robot_id, self.joint_iter, p.POSITION_CONTROL, targetPositions=list_position)
            # p.stepSimulation() for 100 times leads to an error < 1e-6 meter
            for _ in range(sim_steps):
                p.stepSimulation()

    def set_random_states(self):
        target_position = np.random.uniform(self.limit_lower, self.limit_upper, [self.num_side, self.num_side])
        self.set_states(target_position)
        return target_position

    def reset(self):
        target_position = np.ones([self.num_side, self.num_side]) * (self.limit_lower + self.limit_upper) / 2
        self.set_states(target_position)

    def take_action(self, action: np.ndarray, granularity=10):
        """
        Args:
            action: the elements of action should be 1 for UP, 0 for NOOP, -1 for DOWN
            granularity: how many steps it takes to move from the bottom to the top
        """
        assert action.shape == (self.num_side, self.num_side)
        current_position, _ = self.get_states()
        target_position = current_position + (self.limit_upper - self.limit_lower) / granularity * action
        self.set_states(target_position, interp_steps=10, sim_steps=10)

    def regularization(self, mode='height', action=None):
        if mode == 'height':
            # regularize the sum of the heights of the joints
            current_position, _ = self.get_states()
            return np.sum(current_position) / self.num_act - (self.limit_lower + self.limit_upper) / 2
        elif mode == 'motion':
            # regularize the motion of the action
            assert action is not None
            assert action.shape == (self.num_side, self.num_side) or action.shape == (self.num_act,)
            return np.sum(np.absolute(action)) / self.num_act
        else:
            raise ValueError("Unknown regularization option!")


@hydra.main(version_base=None, config_path='config', config_name='demo')
def main(cfg):
    robot = TableBot(cfg)

    # list_obj = [0] * 16
    # list_obj[1] = 1
    # array_obj = robot.list2array(list_obj)
    # list_obj_ = robot.array2list(array_obj)

    current_dir = os.path.dirname(os.path.realpath(__file__))
    p.loadURDF(osp.join(current_dir, cfg.object.path), cfg.object.position)
    for _ in range(100):
        p.stepSimulation()
    #
    # rgb, depth, mask = robot.get_observations(with_gt_mask=True)
    # plt.imshow(rgb)
    # plt.show()
    # plt.imshow(depth)
    # plt.show()
    # plt.imshow(mask)
    # plt.show()
    #
    # target = robot.set_random_states()
    # now = robot.get_states()[0]
    # print(now - target)
    #
    # rgb, depth, mask = robot.get_observations(with_gt_mask=True)
    # plt.imshow(rgb)
    # plt.show()
    # plt.imshow(depth)
    # plt.show()
    # plt.imshow(mask)
    # plt.show()
    #
    # agent = HandCraftedAgent(robot)
    # agent.make_wave(robot.limit_upper / 3)


if __name__ == '__main__':
    main()
