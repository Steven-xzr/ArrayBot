import hydra
import pfrl
import torch

from waste.environments import LiftEnv
from waste.policy.rl_module import PerceptionXYZ8, DeConv8
from pfrl.policies import SoftmaxCategoricalHead


class ActorCritic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.p_net = PerceptionXYZ8()
        self.deconv_net = DeConv8()
        self.v_net = torch.nn.Linear(256, 1)
        self.head = SoftmaxCategoricalHead()

    def forward(self, x):
        object_position = x[0]
        joint_position = x[1]
        state_emb = self.p_net(object_position, joint_position)
        a_prob = self.deconv_net(state_emb)   # b x 3 x 8 x 8
        a_prob = a_prob.reshape(a_prob.shape[0], 3, 8 * 8).permute(0, 2, 1)  # b x 64 x 3
        v = self.v_net(state_emb)
        return tuple([self.head(a_prob), v])


@hydra.main(version_base=None, config_path='config', config_name='lift_block_ppo_ee')
def main(cfg):
    env = LiftEnv(cfg)
    env.reset()

    model = ActorCritic()
    opt = torch.optim.Adam(model.parameters())

    agent = pfrl.agents.PPO(
        model,
        opt,
        gpu=0,
        update_interval=128,
        phi=lambda x: (x['object_position'], x['joint_position']),
    )

    n_episodes = 300
    max_episode_len = 100
    for i in range(1, n_episodes + 1):
        obs = env.reset()
        R = 0  # return (sum of rewards)
        t = 0  # time step
        while True:
            # Uncomment to watch the behavior in a GUI window
            # env.render()
            action = agent.act(obs)
            obs, reward, done, _ = env.step(action)
            R += reward
            t += 1
            reset = t == max_episode_len
            agent.observe(obs, reward, done, reset)
            if done or reset:
                break
        if i % 1 == 0:
            print('episode:', i, 'R:', R / t)
        if i % 50 == 0:
            print('statistics:', agent.get_statistics())
    print('Finished.')


if __name__ == '__main__':
    main()
