import torch
import torch.nn as nn

# Dueling Double DQN

class DQN(nn.Module):

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, dueling: bool = True):

        super().__init__()

        self.dueling = dueling

        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        if dueling:

            self.value_stream = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            )

            self.advantage_stream = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, action_dim)
            )

        else:

            self.q_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, x: torch.Tensor):

        features = self.feature(x)

        if self.dueling:

            value = self.value_stream(features)
            advantage = self.advantage_stream(features)
            return value + advantage - advantage.mean(dim = 1, keepdim = True)

        return self.q_head(features)
