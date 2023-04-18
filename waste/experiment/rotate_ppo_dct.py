import hydra
import pfrl
import torch

from waste.environments import RotateEnvDCT
from pfrl.policies import SoftmaxCategoricalHead


class ActorCritic(torch.nn.Module):
    def __init__(self, dim_obj=6, dim_freq=10):
        super().__init__()
        self.p_net = torch.nn.Sequential(
            torch.nn.Linear(dim_obj + dim_freq, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
        )
        self.v_net = torch.nn.Sequential(
            torch.nn.Linear(256, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
        )
        self.a_net = torch.nn.Sequential(
            torch.nn.Linear(256, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 2 * dim_freq),
        )
        self.head = SoftmaxCategoricalHead()

    def forward(self, x):
        obj_pos = x[0]
        obj_ori = x[1]
        freq = x[2]
        state_emb = self.p_net(torch.cat([obj_pos, obj_ori, freq], dim=1))
        a_prob = self.a_net(state_emb)
        v = self.v_net(state_emb)
        return tuple([self.head(a_prob), v])


@hydra.main(version_base=None, config_path='config', config_name='rotate_duck_ppo_dct')
def main(cfg):
    env = RotateEnvDCT(cfg)
    env.reset()

    model = ActorCritic()
    opt = torch.optim.Adam(model.parameters())

    agent = pfrl.agents.PPO(
        model,
        opt,
        gpu=0,
        update_interval=128,
        phi=lambda x: (x['object_position'], x['object_orientation'], env.dct_handler.dct(x['joint_position'])),
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
