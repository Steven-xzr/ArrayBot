import hydra
import numpy as np
import time
from tqdm import tqdm
from gym.wrappers import TimeLimit, AutoResetWrapper
from stable_baselines3 import PPO
from stable_baselines3.ppo import MultiInputPolicy
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from environments import TransEnvDCT


def make_env(cfg, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param cfg
    :param seed: (int) the initial seed
    :param rank: (int) index of the subprocess
    """

    def _init():
        env = TransEnvDCT(cfg)
        # env = Monitor(TimeLimit(AutoResetWrapper(env), env.max_steps))
        env = Monitor(TimeLimit(env, env.max_steps))
        # Important: use a different seed for each environment
        # env.seed(seed + rank)
        return env

    set_random_seed(seed)
    return _init


@hydra.main(version_base=None, config_path='config', config_name='se3_egad_ppo_dct')
def main(cfg):
    n_procs = 100
    n_steps = 8
    train_env = SubprocVecEnv([make_env(cfg, i) for i in range(n_procs)], start_method='fork')
    eval_env = Monitor(TransEnvDCT(cfg))

    model = PPO(MultiInputPolicy, train_env, verbose=0, n_steps=n_steps, batch_size=n_procs * n_steps,
                tensorboard_log='logs')

    print(train_env.reset())
    for t in tqdm(range(10000)):
        obs, rew, done, info = train_env.step(np.array([train_env.action_space.sample() for _ in range(n_procs)]))
        if np.any(done):
            print(t)


if __name__ == '__main__':
    main()
