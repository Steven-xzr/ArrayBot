import hydra
import numpy as np
import pfrl
import torch

from environments import LiftEnvDCT, RotateEnvDCT, TransEnvDCT, SE3EnvDCT
from pfrl.policies import SoftmaxCategoricalHead


class ActorCritic(torch.nn.Module):
    def __init__(self, dim_obj=6, dim_freq=6):
        super().__init__()
        self.p_net = torch.nn.Sequential(
            torch.nn.Linear(dim_obj + dim_freq, 64),
            torch.nn.LayerNorm(64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.LayerNorm(128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 256),
            torch.nn.LayerNorm(256),
            torch.nn.ReLU(),
        )
        self.v_net = torch.nn.Sequential(
            torch.nn.Linear(256, 64),
            torch.nn.LayerNorm(64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
        )
        self.a_net = torch.nn.Sequential(
            torch.nn.Linear(256, 64),
            torch.nn.LayerNorm(64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 2 * dim_freq + 1),
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


@hydra.main(version_base=None, config_path='config', config_name='se3_egad_ppo_dct')
def main(cfg):
    # env = LiftEnvDCT(cfg)
    env = TransEnvDCT(cfg)
    # env = SE3EnvDCT(cfg)
    env.reset()

    model = ActorCritic()
    opt = torch.optim.Adam(model.parameters())

    agent = pfrl.agents.PPO(
        model,
        opt,
        gpu=0,
        update_interval=128,
        phi=lambda x: (x['object_position'], x['diff_position'], env.dct_handler.dct(x['joint_position'])),
        entropy_coef=cfg.rl.entropy_coef,
        minibatch_size=cfg.rl.minibatch_size,
    )

    n_episodes = 1000
    max_episode_len = 500
    all_steps = 0
    for i in range(1, n_episodes + 1):
        obs = env.reset()
        diff_all = 0
        delta_diff_all = 0
        t = 0  # time step
        while True:
            # Uncomment to watch the behavior in a GUI window
            # env.render()
            action = agent.act(obs)
            obs, reward_dict, done, _ = env.step(action)
            diff_all += reward_dict['t_diff']
            delta_diff_all += reward_dict['t_delta_diff']
            t += 1
            reset = t == max_episode_len
            # if done is True:
            #     print("Out of border!")
            if reset is True:
                print("Time limit reached!")
            penalty = 0 if done is False else cfg.reward.out_of_border_penalty
            agent.observe(obs, reward_dict['bonus'] + reward_dict['t_delta_diff'] + penalty, done, reset)
            if done or reset:
                all_steps += t
                break
        if i % 1 == 0:
            print('episode:', i, 'steps:', t, 'all_steps:', all_steps,
                  'diff:', diff_all / t, 'delta_diff:', delta_diff_all / t)
        if i % 10 == 0:
            print('statistics:', agent.get_statistics())
    print('Finished.')


if __name__ == '__main__':
    main()
