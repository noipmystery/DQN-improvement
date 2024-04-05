import torch.nn as nn
import torch.nn.functional as F
import torch


class DuelingDQN(nn.Module):
    """
    Arguments:
        in_channels: channels of the input image
        num_actions: total number of actions
    """
    def __init__(self, in_channels, num_actions):
        super(DuelingDQN, self).__init__()
        self.Conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1)
        )
        self.value = nn.Sequential(
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        self.advantage = nn.Sequential(
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        x = self.Conv(x)
        x = x.view(x.size(0), -1)
        value = self.value(x)
        advantage = self.advantage(x)
        return value + advantage - advantage.mean()
