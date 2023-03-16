import hydra
import numpy as np
import time
from gym.wrappers import TimeLimit
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
        env = Monitor(TimeLimit(env, env.max_steps))
        # Important: use a different seed for each environment
        env.seed(seed + rank)
        return env

    set_random_seed(seed)
    return _init


@hydra.main(version_base=None, config_path='config', config_name='se3_egad_ppo_dct')
def main(cfg):
    n_procs = 96
    train_env = SubprocVecEnv([make_env(cfg, i) for i in range(n_procs)], start_method='fork')
    eval_env = Monitor(TransEnvDCT(cfg))

    model = PPO(MultiInputPolicy, train_env, verbose=0, n_steps=16, batch_size=128, tensorboard_log='logs')

    NUM_ITERS = 100
    TRAIN_STEPS = 10000
    # Number of episodes for evaluation
    EVAL_EPS = 2

    for iter in range(NUM_ITERS):
        print("iter:", iter, "starts")
        train_env.reset()
        model.learn(total_timesteps=TRAIN_STEPS, reset_num_timesteps=False, progress_bar=True, tb_log_name='ppo_dct')
        mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=EVAL_EPS)
        print("iter:", iter, " rewards:", np.mean(mean_reward))
        model.save("models/ppo_{}".format(iter))
    train_env.close()


if __name__ == '__main__':
    main()
