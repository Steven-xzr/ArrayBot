import hydra
import os
import wandb
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from termcolor import cprint

import isaacgym
import torch

from isaacgymenvs.tasks import isaacgym_task_map
from isaacgymenvs.utils.reformat import omegaconf_to_dict
from isaacgymenvs.utils.utils import set_np_formatting, set_seed

from algo.ppo.ppo import PPO

from array_robot_vectask import ArrayRobot


@hydra.main(config_name='train_config', config_path='algo/configs')
def main(config: DictConfig):
    if config.checkpoint:
        config.checkpoint = to_absolute_path(config.checkpoint)

    # set numpy formatting for printing only
    set_np_formatting()
    config.seed = set_seed(config.seed)

    cprint('Start Building the Environment', 'green', attrs=['bold'])
    # env = isaacgym_task_map[config.task_name](
    env = ArrayRobot(
        cfg=omegaconf_to_dict(config.task),
        sim_device=config.sim_device,
        rl_device=config.rl_device,
        graphics_device_id=config.graphics_device_id,
        headless=config.headless,
        virtual_screen_capture=False,
        force_render=False,
    )

    minibatch_size = config.train.ppo.horizon_length * config.num_envs
    print(f'Using minibatch size of {minibatch_size} for PPO')
    config.train.ppo.minibatch_size = minibatch_size
    output_dif = os.path.join('algo/outputs', config.output_name)
    os.makedirs(output_dif, exist_ok=True)

    agent = PPO(env, output_dif, full_config=config)
    # connect to wandb
    wandb.init(
        project=config.wandb_project,
        entity=config.wandb_entity,
        name=config.output_name,
        config=omegaconf_to_dict(config),
        mode=config.wandb_mode
    )

    agent.restore_train(config.checkpoint)
    agent.train()

    # close wandb
    wandb.finish()


if __name__ == '__main__':
    main()
