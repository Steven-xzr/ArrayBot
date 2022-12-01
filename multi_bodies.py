import pybullet as p
import time
import pybullet_data

act_size = 0.024
act_height = 0.055

act_num_per_side = 12
act_num = act_num_per_side ** 2


p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

colBoxId = p.createCollisionShape(p.GEOM_BOX, halfExtents=[act_size / 2, act_size / 2, act_height / 2])


