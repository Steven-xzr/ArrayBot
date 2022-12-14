import gym
import torch

from environments import LiftEnv
from policy.rl_module import PerceptionXYZ8, QValue8
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy


class FeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        super(FeatureExtractor, self).__init__(observation_space, features_dim)
        self.f_net = PerceptionXYZ8()

    def forward(self, observations: gym.spaces.Dict) -> torch.Tensor:
        obj = torch.Tensor(observations['object_position'])
        joint = torch.Tensor(observations['joint_position'])
        return self.f_net(obj, joint)


# class CustomActorCriticPolicy(ActorCriticPolicy):
#     def __init__(self, *args, **kwargs):
#         super(CustomActorCriticPolicy, self).__init__(
#             *args,
#             features_extractor_class=FeatureExtractor,
#             **kwargs,
#         )


def main():
    model = PPO('MlpPolicy', "CartPole-v1", verbose=1)
    model.learn(500)


if __name__ == '__main__':
    main()