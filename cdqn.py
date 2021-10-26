import numpy as np
import torch
import torch.nn as nn
from base import CategoricalModel
from base import ReplayBuffer


# Categorical Deep Q Network
class CDQN(object):
    def __init__(self, ns, na, dim, capacity, lr,
                 batch_size, gamma, epsilon, target_update,
                 num_atoms, vmin, vmax):
        self.ns, self.na = ns, na
        self.num_atoms = num_atoms
        self.vmin, self.vmax = vmin, vmax
        self.eval = CategoricalModel(ns, na, dim, num_atoms, vmin, vmax)
        self.target = CategoricalModel(ns, na, dim, num_atoms, vmin, vmax)
        self.buffer = ReplayBuffer(capacity)
        self.optimizer = torch.optim.Adam(self.eval.parameters(), lr=lr)
        self.loss_func = nn.MSELoss()
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update
        self.steps = 0

    def learn(self):
        if self.steps % self.target_update == 0:
            self.target.load_state_dict(self.eval.state_dict())
        self.steps += 1
        batch_memory = self.buffer.sample(self.batch_size)
        b_s = torch.FloatTensor(batch_memory[:, :self.ns])
        b_a = torch.LongTensor(batch_memory[:, self.ns:self.ns+1])
        b_s_ = torch.FloatTensor(batch_memory[:, self.ns+1:-1])
        b_r = torch.FloatTensor(batch_memory[:, -1:])

        interval = torch.linspace(self.vmin, self.vmax, self.num_atoms)
        delta_z = (self.vmax - self.vmin) / (self.num_atoms - 1)
        next_dist = self.target.forward(b_s_) * interval
        next_action = next_dist.sum(2).max(1)[1]
        next_action = next_action.unsqueeze(1).unsqueeze(1).expand(next_dist.size(0), 1, next_dist.size(2))
        next_dist = next_dist.gather(1, next_action).squeeze(1)

        b_r = b_r.expand_as(next_dist)
        interval = interval.unsqueeze(0).expand_as(next_dist)
        Tz = b_r + self.gamma * interval
        Tz = Tz.clamp(self.vmin, self.vmax)
        b = (Tz - self.vmin) / delta_z
        l = b.floor().long()
        u = b.ceil().long()
        offset = torch.linspace(0, (self.batch_size - 1) * self.num_atoms, self.batch_size)
        offset = offset.long().unsqueeze(1).expand(self.batch_size, self.num_atoms)
        proj_dist = torch.zeros(next_dist.size())
        proj_dist.view(-1).index_add_(0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1))
        proj_dist.view(-1).index_add_(0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1))

        dist = self.eval(b_s)
        b_a = b_a.unsqueeze(1).expand(self.batch_size, 1, self.num_atoms)
        dist = dist.gather(1, b_a).squeeze(1)
        dist.data.clamp_(0.01, 0.99)
        loss = -(proj_dist * dist.log()).sum(1).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def choose_action(self, s):
        s = torch.unsqueeze(torch.FloatTensor(s), 0)
        if np.random.uniform() < self.epsilon:
            next_dist = self.eval.forward(s)
            next_dist = next_dist * torch.linspace(self.vmin, self.vmax, self.num_atoms)
            action = next_dist.sum(2).max(1)[1].numpy()[0]
        else:
            action = np.random.randint(0, self.na)
        return action