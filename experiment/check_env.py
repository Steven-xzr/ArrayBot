import hydra
import numpy as np
import os

from environments import LiftEnv


@hydra.main(version_base=None, config_path='config', config_name='lift_ball')
def main(cfg):
    env = LiftEnv(cfg)
    env.reset()

    for _ in range(1000):
        action = np.random.randint(-1, 2, [env.robot.num_side, env.robot.num_side])
        _, reward, _, _ = env.step(action)
        print(reward)


if __name__ == '__main__':
    main()
