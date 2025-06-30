import numpy as np
import scipy.signal
import torch
import torch.nn as nn


def combined_shape(length, shape=None):  
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape) 

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

class MLPActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, dropout_prob=0):
        super().__init__()

        pi_sizes = [obs_dim] + list(hidden_sizes)
        self.fc_layers = []
        for i in range(len(pi_sizes) - 1):
            self.fc_layers.append(nn.Linear(pi_sizes[i], pi_sizes[i + 1]))
            self.fc_layers.append(activation())
            self.fc_layers.append(nn.Dropout(p=dropout_prob))

        self.fc_layers.pop()  # Remove the last dropout layer
        self.pi = nn.Sequential(*self.fc_layers, nn.Linear(pi_sizes[-1], act_dim), nn.Tanh())

    def forward(self, obs):
        return self.pi(obs)


class MLPQFunction(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, dropout_prob=0):
        super().__init__()

        q_sizes = [obs_dim + act_dim] + list(hidden_sizes)
        self.fc_layers = []
        for i in range(len(q_sizes) - 1):
            self.fc_layers.append(nn.Linear(q_sizes[i], q_sizes[i + 1]))
            self.fc_layers.append(activation())
            self.fc_layers.append(nn.Dropout(p=dropout_prob))

        self.fc_layers.pop()  # Remove the last dropout layer
        self.q = nn.Sequential(*self.fc_layers, nn.Linear(q_sizes[-1], 1))

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1)


class MLPActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=(256, 256), activation=nn.ReLU, dropout_prob=0):
        super().__init__()

        # Build policy and value functions with dropout
        self.pi = MLPActor(obs_dim, act_dim, hidden_sizes, activation, dropout_prob)
        self.q = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation, dropout_prob)

    def act(self, obs):
        with torch.no_grad():
            return self.pi(obs).cpu().numpy()