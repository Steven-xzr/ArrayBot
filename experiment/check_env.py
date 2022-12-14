import hydra
import numpy as np
import os

from environments import LiftEnv
from stable_baselines3.common.env_checker import check_env


@hydra.main(version_base=None, config_path='config', config_name='lift_ball')
def main(cfg):
    env = LiftEnv(cfg)
    env.reset()

    check_env(env, warn=True, skip_render_check=True)

    for _ in range(1000):
        action = np.random.randint(-1, 2, [env.robot.num_side, env.robot.num_side])
        obs, reward, done, info = env.step(action)
        # print(reward)


if __name__ == '__main__':
    main()
