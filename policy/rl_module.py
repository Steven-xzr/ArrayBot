import pfrl
import torch
import torch.nn as nn
import torch.nn.functional as F


class PerceptionXYZ8(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(6, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 2)
        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc_all = nn.Linear(64 + 64, 256)

    def forward(self, obj: torch.Tensor, joint: torch.Tensor):
        """
        Turn the observation into a state embedding.

        Args:
            obj: the position of the object, shape: [batch_size, 3]
            joint: the position of the joints, shape: [batch_size, num_side=8, num_side=8]
        Returns:
            state_emb: state embedding, shape: [batch_size, 256]
        """
        obj = F.relu(self.fc1(obj))
        obj = F.relu(self.fc2(obj))

        joint = joint.unsqueeze(1)
        joint, pool_idx1 = F.max_pool2d(F.relu(self.conv1(joint)), 2, return_indices=True)
        joint, pool_idx2 = F.max_pool2d(F.relu(self.conv2(joint)), 2, return_indices=True)
        joint = F.relu(self.conv3(joint))

        state_emb = torch.cat([obj, joint.view(joint.shape[0], -1)], dim=1)
        state_emb = F.relu(self.fc_all(state_emb))
        return state_emb


class QFunction8(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            # input is state_emb, going into a convolution
            nn.ConvTranspose2d(256, 128, 2, 1, 0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # state size. 128 x 2 x 2
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # state size. 64 x 4 x 4
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            # state size. 3 x 8 x 8
        )

    def forward(self, state_emb: torch.Tensor):
        """
        Turn the state embedding into a Q-value matrix. Here, we adopt an image generation approach.
        REF: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
        Args:
            state_emb: the state embedding, shape: [batch_size, 256]
        Return:
            q: the Q value of each action, shape: [batch_size, dim_choice=3 (UP/NOOP/DOWN), num_side=8, num_side=8]
        """
        return self.main(state_emb)
