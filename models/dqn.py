"""
dqn.py

Q-network for Double DQN. Simple MLP over the fused observation vector
produced by DetectorFaultControlEnv (which already includes the LSTM's
fault-probability estimate), predicting Q-values for the 5 control actions.
"""
import torch
import torch.nn as nn


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    net = QNetwork(obs_dim=8, n_actions=5)
    x = torch.randn(4, 8)
    q = net(x)
    print("Q shape:", q.shape)
