import gym
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore

import threading
import time

from dqn import DQN
from ddqn import DDQN
from dddqn import DDDQN
from pddqn import PDDQN
from ndqn import NDQN
from cdqn import CDQN

QUIET = False
RENDER = True


def training(func, max_steps, rewards, indices, hyper):
    env = gym.make('CartPole-v0')
    na = env.action_space.n
    ns = env.observation_space.shape[0]
    rl = func(ns, na, **hyper)
    steps, iter = 0, 0

    while steps < max_steps:
        s = env.reset()
        ep_r = 0
        while True:
            if RENDER:
                env.render()
                if steps == 0:
                    time.sleep(10)
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
                    rewards.append(round(ep_r, 2))
                    indices.append(steps)
                    if len(rewards) > 500:
                        rewards = rewards[1:]
                        indices = indices[1:]
                    if not QUIET:
                        print("Ep ", iter, " Reward : ", round(ep_r, 2))
                        print("Steps : ", steps, " / ", max_steps)
            if done:
                if len(rl.buffer.memory) < rl.buffer.capacity:
                    if not QUIET:
                        print("Memory Filled : ", len(rl.buffer.memory), " / ", rl.buffer.capacity)
                    iter -= 1
                break
            s = s_
        iter += 1


if __name__ == '__main__':
    app = QtGui.QApplication([])
    win = pg.GraphicsLayoutWidget(show=True, title="Gym Rainbow Test")
    win.resize(600, 600)
    win.setWindowTitle("Gym Rainbow Test")
    pg.setConfigOptions(antialias=True)
    plots = win.addPlot(title="Eposide Reward")

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

    rewards = [[], [], [], [], [], []]
    indices = [[], [], [], [], [], []]

    light = 255
    colors = ["#FF0000", "#FFA500", "FFFF00",
              "#00FF00", "#007FFF", "#0000FF"]
    curves = []
    for i in range(6):
        curves.append(plots.plot(pen=pg.mkPen(color=colors[i], width=2)))

    def update():
        for i in range(6):
            curves[i].setData(indices[i], rewards[i])

    max_steps = int(1e4)
    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(1000)

    if not RENDER:
        threading.Thread(target=training, args=(DQN  , max_steps, rewards[0], indices[0], hyper_common)).start()
        # threading.Thread(target=training, args=(DDQN , max_steps, rewards[1], indices[1], hyper_common)).start()
        # threading.Thread(target=training, args=(DDDQN, max_steps, rewards[2], indices[2], hyper_common)).start()
        # threading.Thread(target=training, args=(PDDQN, max_steps, rewards[3], indices[3], {**hyper_common, **hyper_pddqn})).start()
        # threading.Thread(target=training, args=(NDQN , max_steps, rewards[4], indices[4], {**hyper_common, **hyper_ndqn})).start()
        # threading.Thread(target=training, args=(CDQN , max_steps, rewards[5], indices[5], {**hyper_common, **hyper_cdqn})).start()
        pass
    else:
        training(DQN  , max_steps, rewards[0], indices[0], hyper_common)
        # training(DDQN , max_steps, rewards[1], indices[1], hyper_common)
        # training(DDDQN, max_steps, rewards[2], indices[2], hyper_common)
        # training(PDDQN, max_steps, rewards[3], indices[3], {**hyper_common, **hyper_pddqn})
        # training(NDQN , max_steps, rewards[4], indices[4], {**hyper_common, **hyper_ndqn})
        # training(CDQN , max_steps, rewards[5], indices[5], {**hyper_common, **hyper_cdqn})
        pass

    pg.QtGui.QGuiApplication.exec_()


