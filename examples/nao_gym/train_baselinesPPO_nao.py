#!/usr/bin/env python3
import sys
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except Exception:
    pass

import NaoEnv
import numpy as np
import time
from datetime import datetime

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

ENV_ID = 'NaoBulletEnv'


def train(num_timesteps, seed, model_path=None):

    env = NaoEnv.NaoEnv(gui=True)
    env = DummyVecEnv([lambda: env])

    model = PPO2(
                 MlpPolicy,
                 env,
                 verbose=2,
                 tensorboard_log="logs/PPO2Agent/" + datetime.now().strftime(
                     "%Y%m%d-%H%M%S"))
    model.learn(total_timesteps=num_timesteps)
    model.save(model_path)
    env.close()


def visualize(name_model):
    model = PPO2.load(name_model)
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
    name_model = "models_nao/PPO2_200M"
    train(num_timesteps=int(20e7), seed=seed,
          model_path=name_model)
    # visualize(name_model)

if __name__ == '__main__':
    main()
