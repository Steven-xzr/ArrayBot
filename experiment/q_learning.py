import hydra
import numpy as np
import pfrl
import torch
import torch.nn as nn

from environments import LiftEnv
from policy.rl_module import PerceptionXYZ8, QValue8


class QFunction(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.p_net = PerceptionXYZ8()
        self.q_net = QValue8()

    def forward(self, object_position, joint_position):
        state_emb = self.p_net(object_position, joint_position)
        q = self.q_net(state_emb)
        return pfrl.action_value.DiscreteActionValue(q)


@hydra.main(version_base=None, config_path='config', config_name='lift_block_dqn')
def main(cfg):
    env = LiftEnv(cfg)
    env.reset()

    q_func = QFunction()
    opt = torch.optim.Adam(q_func.parameters())

    agent = pfrl.agents.DoubleDQN(
        q_func,
        opt,
        gamma=0.99,
        explorer=pfrl.explorers.ConstantEpsilonGreedy(
            epsilon=0.3, random_action_func=env.sample_action
        ),
        replay_buffer=pfrl.replay_buffers.ReplayBuffer(10 ** 6),
        gpu=0,
        update_interval=1,
        target_update_interval=100,
        # phi=lambda x: x.astype(np.float32, copy=False),
    )

    n_episodes = 300
    max_episode_len = 200
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
        if i % 10 == 0:
            print('episode:', i, 'R:', R)
        if i % 50 == 0:
            print('statistics:', agent.get_statistics())
    print('Finished.')


if __name__ == '__main__':
    main()
