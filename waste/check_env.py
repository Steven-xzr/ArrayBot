from stable_baselines3.common.env_checker import check_env
from environments import TransEnvDCT
import hydra


@hydra.main(version_base=None, config_path='config', config_name='se3_egad_ppo_dct')
def main(cfg):
    env = TransEnvDCT(cfg)
    check_env(env)


if __name__ == '__main__':
    main()
