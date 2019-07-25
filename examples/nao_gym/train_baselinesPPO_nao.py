#!/usr/bin/env python3
from baselines.common import tf_util as U
from baselines import logger
import NaoEnv
import gym
import numpy as np
import time
from datetime import datetime

ENV_ID = 'NaoBulletEnv'


def train(num_timesteps, seed, model_path=None):
    from baselines.ppo1 import mlp_policy, pposgd_simple
    U.make_session(num_cpu=1).__enter__()

    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(
            name=name,
            ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=2)
    env = getattr(NaoEnv, ENV_ID)(render=True)

    # parameters below were the best found in a simple random search
    # these are good enough to make humanoid walk, but whether those are
    # an absolute best or not is not certain
    env = RewScale(env, 0.1)
    logger.log("NOTE: reward will be scaled by a factor"
               " of 10  in logged stats."
               " Check the monitor for unscaled reward.")
    pi = pposgd_simple.learn(
            env, policy_fn,
            max_timesteps=num_timesteps,
            timesteps_per_actorbatch=4096,
            clip_param=0.1, entcoeff=0.0,
            optim_epochs=10,
            optim_stepsize=1e-4,
            optim_batchsize=64,
            gamma=0.99,
            lam=0.95,
            schedule='constant'
            )
    env.close()
    if model_path:
        U.save_state(model_path)

    return pi


class RewScale(gym.RewardWrapper):
    def __init__(self, env, scale):
        gym.RewardWrapper.__init__(self, env)
        self.scale = scale

    def reward(self, r):
        return r * self.scale


def visualize():
    # construct the model object, load pre-trained model and render
    seed = int(time.time())
    pi = train(num_timesteps=1, seed=seed)
    U.load_state("models/")
    env = getattr(NaoEnv, ENV_ID)(render=True)

    ob = env.reset()
    while True:
        action = pi.act(stochastic=False, ob=ob)[0]
        ob, _, done, _ = env.step(action)
        env.render()
        if done:
            ob = env.reset()


def main():
    logger.configure(dir="logs/PPOAgent/" + datetime.now().strftime(
                     "%Y%m%d-%H%M%S"),
                     format_strs=['stdout', 'log', 'csv', 'tensorboard'])
    logger.set_level(10)
    seed = int(time.time())
    np.random.seed(seed)
    # train the model
    train(num_timesteps=int(1e7), seed=seed,
          model_path="models/")
    # visualize()

if __name__ == '__main__':
    main()
