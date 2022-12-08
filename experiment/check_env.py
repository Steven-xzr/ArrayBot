import hydra
import numpy as np
import os

from environments import LiftBlockEnv


@hydra.main(version_base=None, config_path='config', config_name='lift_block')
def main(cfg):
    env = LiftBlockEnv(cfg)
    env.reset()

    for _ in range(1000):
        action = np.random.randint(-1, 2, [env.robot.num_side, env.robot.num_side])
        env.step(action)


if __name__ == '__main__':
    main()
