import pybullet as p
import pybullet_data

import random
import time
import math
import numpy as np

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadURDF("plane.urdf", [0, 0, -0.3], useFixedBase=True)
robot_id = p.loadURDF("urdf/tablebot_100.urdf", [0, 0, 0], useFixedBase=True)
# robot_id = p.loadURDF("urdf/tablebot_unit.urdf", [0, 0, 0], useFixedBase=True)
p.resetBasePositionAndOrientation(robot_id, [0, 0, 0], [0, 0, 0, 1])

cube_id = p.loadURDF("cube_small.urdf", [0.08, 0.08, 0.08])

p.setGravity(0, 0, -10)
# p.setRealTimeSimulation(1)

num_act = p.getNumJoints(robot_id)
num_side = int(math.sqrt(num_act))
joint_info = p.getJointInfo(robot_id, 0)
limit_lower = joint_info[8]
limit_upper = joint_info[9]
joint_idx_list = range(num_act)

while True:
    target_position = random.random() * (limit_upper - limit_lower) + limit_lower
    # target_position_array = np.random.rand(num_act) * (limit_upper - limit_lower) + limit_lower * np.ones(num_act)
    p.setJointMotorControl2(robot_id, 2, p.POSITION_CONTROL, target_position)
    # p.setJointMotorControlArray(robot_id, joint_idx_list, p.POSITION_CONTROL, target_position_array)
    for _ in range(100):
        p.stepSimulation()
    print("Target position: {}. Current position: {}.".format(target_position, p.getJointState(robot_id, 0)[0]))
    time.sleep(1)
