#!/usr/bin/env python3
import sys
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except Exception:
    pass

import NaoEnv
import numpy as np
import time

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from stable_baselines import DDPG

ENV_ID = 'NaoBulletEnv'


def visualize(name_model):
    model = DDPG.load(name_model)
    env = NaoEnv.NaoEnv(gui=True)
    env = DummyVecEnv([lambda: env])
    # Enjoy trained agent
    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()


def main():
    seed = int(time.time())
    np.random.seed(seed)
    # train the model
    name_model = "models_nao/DDPG_walk_pretrained_with_speed"
    visualize(name_model)

if __name__ == '__main__':
    main()
