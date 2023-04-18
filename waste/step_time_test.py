import hydra
import numpy as np
import time
from gym.wrappers import TimeLimit, AutoResetWrapper
from stable_baselines3 import PPO
from stable_baselines3.ppo import MultiInputPolicy
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from tqdm import tqdm
import pybullet as p
import os


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
    n_procs = 1
    n_steps = 8
    train_env = TransEnvDCT(cfg)
    # train_env = SubprocVecEnv([make_env(cfg, i) for i in range(n_procs)], start_method='fork')
    # eval_env = Monitor(TransEnvDCT(cfg))
    #
    # model = PPO(MultiInputPolicy, train_env, verbose=0, n_steps=n_steps, batch_size=n_procs * n_steps,
    #             tensorboard_log='logs')

    print(train_env.reset())
    #
    # model = PPO(MultiInputPolicy, train_env, verbose=0, n_steps=n_steps, batch_size=n_procs * n_steps)
    # model.learn(total_timesteps=10000, reset_num_timesteps=False, progress_bar=True)

    # logId = p.startStateLogging(p.STATE_LOGGING_PROFILE_TIMINGS, "remote.json")
    for t in tqdm(range(1000)):
        p.stepSimulation()
    # p.stopStateLogging(logId)
        # obs, rew, done, truc, info = train_env.step(train_env.action_space.sample())
        # if done:
        #     print('done at ', t)
        #     train_env.reset()
        # ret = train_env.step(np.array([train_env.action_space.sample() for _ in range(n_procs)]))
        # print(ret)


if __name__ == '__main__':
    main()
