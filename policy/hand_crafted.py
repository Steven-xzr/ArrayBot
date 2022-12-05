# from tablebot import TableBot
import numpy as np


class HandCraftedAgent:
    def __init__(self, robot):
        self.robot = robot
        self.num_side = robot.num_side
        self.limit_lower = robot.limit_lower
        self.limit_upper = robot.limit_upper

    def make_wave(self, height=None):
        for x in range(self.num_side):
            position = np.ones([self.num_side, self.num_side]) * self.limit_lower
            position[x, :] = self.limit_upper if height is None else height
            self.robot.set_states(position)
