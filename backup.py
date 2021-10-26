import gym
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
import threading
import sys

from dqn import DQN
from ddqn import DDQN
from dddqn import DDDQN
from pddqn import PDDQN
from ndqn import NDQN
from cdqn import CDQN


def training(func, win, timer, max_steps, hyper):
    # env = gym.make('SpaceInvaders-ram-v0')
    env = gym.make('CartPole-v0')
    na = env.action_space.n
    ns = env.observation_space.shape[0]
    rl = func(ns, na, **hyper)
    # max_episodes = 400
    steps, iter = 0, 0

    rewards = []

    px = win.addPlot(title="Eposide Reward")
    curve = px.plot(pen='y')

    def update():
        curve.setData(rewards)

    timer.timeout.connect(update)

    while steps < max_steps:
        s = env.reset()
        ep_r = 0
        while True:
            env.render()
            a = rl.choose_action(s)
            s_, r, done, _ = env.step(a)
            steps += 1

            x, x_dot, theta, theta_dot = s_
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            r = r1 + r2

            rl.buffer.push(s, a, s_, r)
            ep_r += r
            if len(rl.buffer.memory) == rl.buffer.capacity:
                rl.learn()
                if done:
                    print("Ep ", iter, " Reward : ", round(ep_r, 2))
                    rewards.append(round(ep_r, 2))
                    print("Steps : ", steps, " / ", max_steps)
            if done:
                if len(rl.buffer.memory) < rl.buffer.capacity:
                    print("Memory Filled : ", len(rl.buffer.memory), " / ", rl.buffer.capacity)
                    iter -= 1
                break
            s = s_
        iter += 1


if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    win = pg.GraphicsLayoutWidget(show=True, title="Eposide Reward")
    win.resize(300, 300)
    win.setWindowTitle('Eposide Reward')
    pg.setConfigOptions(antialias=True)

    hyper_common = {
        'dim': 64,
        'capacity': 200,
        'lr': 0.01,
        'batch_size': 32,
        'gamma': 0.9,
        'epsilon': 0.9,
        'target_update': 10,
    }
    hyper_cdqn = {
        'num_atoms': 51,
        'vmin': -10,
        'vmax': 10,
    }
    hyper_pddqn = {
        'alpha': 0.6,
        'beta': 0.4,
    }
    hyper_ndqn = {
        'std': 0.4,
    }

    max_steps = int(5e4)
    timer = QtCore.QTimer()
    timer.start(1000)

    # threading.Thread(target=training, args=(DQN  , win, timer, max_steps, hyper_common)).start()

    # training(DDQN , win, max_steps, hyper_common)
    # training(DDDQN, win, max_steps, hyper_common)
    # training(PDDQN, win, max_steps, {**hyper_common, **hyper_pddqn})
    # training(NDQN , win, max_steps, {**hyper_common, **hyper_ndqn})
    # training(CDQN , win, timer, max_steps, {**hyper_common, **hyper_cdqn})
