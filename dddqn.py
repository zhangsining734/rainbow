import numpy as np
import torch
import torch.nn as nn
from base import DuelModel
from base import ReplayBuffer


# Dueling Double Deep Q Network
class DDDQN(object):
    def __init__(self, ns, na, dim, capacity, lr,
                 batch_size, gamma, epsilon, target_update):
        self.ns, self.na = ns, na
        self.eval = DuelModel(ns, na, dim)
        self.target = DuelModel(ns, na, dim)
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

        q_eval = self.eval(b_s).gather(1, b_a)
        q_next = self.target(b_s_).detach()
        a_max = q_next.max(1)[1].view(self.batch_size, 1)
        q_target = b_r + self.gamma * q_next.gather(1, a_max)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def choose_action(self, s):
        s = torch.unsqueeze(torch.FloatTensor(s), 0)
        if np.random.uniform() < self.epsilon:
            action_value = self.eval.forward(s)
            action = torch.max(action_value, 1)[1].data.numpy()
            action = action[0]
        else:
            action = np.random.randint(0, self.na)
        return action