import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
import random
import math


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, s, a, s_, r):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = np.hstack((s, a, s_, r))
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return np.vstack(random.sample(self.memory, batch_size))


class PrioritizedReplayBuffer(object):
    def __init__(self, capacity, alpha, beta):
        self.capacity = capacity
        self.memory = []
        self.priorities = np.zeros((capacity, ), dtype=np.float32)
        self.position = 0
        self.prob_alpha = alpha
        self.beta = beta

    def push(self, s, a, s_, r):
        self.priorities[self.position] = self.priorities.max() if self.memory else 1.0
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = np.hstack((s, a, s_, r))
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.memory) < self.capacity:
            prios = self.priorities[:self.position]
        else:
            prios = self.priorities
        prios = prios ** self.prob_alpha
        prios = prios / prios.sum()
        indices = np.random.choice(len(self.memory), batch_size, p=prios)
        weights = (len(self.memory) * prios[indices]) ** (-self.beta)
        weights = weights / weights.max()
        weights = np.array(weights, dtype=np.float32)
        weights = weights[np.newaxis]
        return np.array([self.memory[idx] for idx in indices]), indices, weights

    def update_priority(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio


class Model(nn.Module):
    def __init__(self, ni, no, dim):
        super().__init__()
        self.layer = nn.Linear(ni, dim)
        self.layer.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(dim, no)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.layer(x)
        x = func.relu(x)
        x = self.out(x)
        return x


class DuelModel(nn.Module):
    def __init__(self, ni, no, dim):
        super().__init__()
        self.advantage = Model(ni, no, dim)
        self.value = Model(ni, 1, dim)

    def forward(self, x):
        adv = self.advantage(x)
        val = self.value(x)
        return val + (adv - adv.mean())


class CategoricalModel(nn.Module):
    def __init__(self, ns, na, dim, n_atoms, vmin, vmax):
        super().__init__()
        self.na, self.n_atoms = na, n_atoms
        self.vmin, self.vmax = vmin, vmax
        self.model = Model(ns, na * n_atoms, dim)

    def forward(self, x):
        x = self.model(x)
        x = func.softmax(x.view(-1, self.n_atoms), dim=1)
        x = x.view(-1, self.na, self.n_atoms)
        return x


class NoisyLinear(nn.Module):
    def __init__(self, input, output, std):
        super().__init__()
        self.input, self.output, self.std = input, output, std
        self.weight_mu = nn.Parameter(torch.FloatTensor(output, input))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(output, input))
        self.register_buffer('weight_epsilon', torch.FloatTensor(output, input))
        self.bias_mu = nn.Parameter(torch.FloatTensor(output))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(output))
        self.register_buffer('bias_epsilon', torch.FloatTensor(output))
        self.reset_parameters()
        self.reset_noise()

    def forward(self, x):
        weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
        bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        return func.linear(x, weight, bias)

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.weight_mu.size(1))
        with torch.no_grad():
            self.weight_mu.uniform_(-mu_range, mu_range)
            self.weight_sigma.data.fill_(self.std / math.sqrt(self.weight_sigma.size(1)))
            self.bias_mu.uniform_(-mu_range, mu_range)
            self.bias_sigma.data.fill_(self.std / math.sqrt(self.bias_sigma.size(0)))

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.input)
        epsilon_out = self._scale_noise(self.output)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def _scale_noise(self, size):
        x = torch.randn(size)
        x = x.sign() * x.abs().sqrt()
        return x

class NoisyModel(nn.Module):
    def __init__(self, ni, no, dim, std):
        super().__init__()
        self.layer = NoisyLinear(ni, dim, std)
        self.out = NoisyLinear(dim, no, std)

    def forward(self, x):
        x = func.relu(self.layer(x))
        x = func.relu(self.out(x))
        return x

    def reset_noise(self):
        self.layer.reset_noise()
        self.out.reset_noise()