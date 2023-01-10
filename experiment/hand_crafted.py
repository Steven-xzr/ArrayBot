import hydra

from environments import LiftEnv
from policy.hand_crafted import HandCraftedAgent


@hydra.main(version_base=None, config_path='config', config_name='lift_block')
def main(cfg):
    env = LiftEnv(cfg)
    agent = HandCraftedAgent(env.robot)

    while True:
        env.reset()
        agent.make_wave()


if __name__ == '__main__':
    main()
