import pfrl
import torch
import torch.nn


class PerceptionXYZ(torch.nn.Module):
    def __init__(self, obs_size, n_actions):
        super().__init__()
        self.l1 = torch.nn.Linear(obs_size, 50)
        self.l2 = torch.nn.Linear(50, 50)
        self.l3 = torch.nn.Linear(50, n_actions)

    def forward(self, x, test=False):
        h = torch.nn.functional.relu(self.l1(x))
        h = torch.nn.functional.relu(self.l2(h))
        return pfrl.action_value.DiscreteActionValue(self.l3(h))