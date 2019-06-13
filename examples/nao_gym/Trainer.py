#!/usr/bin/env python
# coding: utf-8

import argparse
import numpy as np
import time
import NaoEnv
import kerasrl_utils
from agents import agent_register

np.set_printoptions(precision=3, suppress=True, linewidth=10000)


def add_opts(parser):
    parser.add_argument('--agent',
                        type=str, default="KerasDQNAgent",
                        help="Agent to be trained with.")
    parser.add_argument('--env', type=str,
                        default="2DDetachedCartPolev0Env",
                        help="Environment to be trained in.")
    parser.add_argument('--use-latest',
                        action='store_true',
                        help="Should the trainer"
                        "retrain/show with the most recent save?")
    parser.add_argument('--train-for', type=int,
                        default=100, help="The number of epochs to train for.")
    parser.add_argument('--test-for', type=int,
                        default=0, help="The number of epoch to test for.")
    parser.add_argument('--load-file', type=str,
                        default=None,
                        help="The weight file to load for training.")
    parser.add_argument('--save-file', type=str,
                        default=None,
                        help="The weight file to save after training.")


class Trainer:
    '''
    The trainer class helps to easily set up a gym training session using
    an agent(representing the learning algorithm and the gym
    (being the environment)
    '''
    def __init__(self):
        # initialize random seed
        np.random.seed(int(time.time()))
        self.visualize = True

    def setup_exercise(self, opts):
            # setup agent
            agent = agent_register.make(opts.agent, opts=opts)
            # setup environment
            env = getattr(NaoEnv, opts.env)(render=True)
            env.reset()

            # configurations
            env.seed(int(time.time()))
            if agent.metadata['discrete_actions']:
                agent.configure(env.observation_space.shape,
                                env.action_space.n)
            else:
                agent.configure(env.observation_space.shape,
                                env.action_space.shape[0])

            if opts.use_latest:
                properties = kerasrl_utils.get_latest_save("checkpoints/",
                                                           opts.agent,
                                                           opts.env, 0)
                if properties == []:
                    print("No previous weight saves found for %s-%s" %
                          (opts.agent, opts.env))
                else:
                    opts.load_file = "checkpoints/%s-%s-%s.h5"\
                        % (properties[0], properties[1], properties[2])
                    print("Continue from [%s] " % opts.load_file)

            if opts.load_file is not None:
                print("loading weights from [%s]" % opts.load_file)
                agent.load_weights(opts.load_file)

            # Okay, now it's time to learn something! We visualize the training
            # here for show, but this
            # slows down training quite a lot. You can always safely abort the
            # training prematurely using
            # Ctrl + C.
            agent.train(env, nb_steps=opts.train_for,
                        visualize=self.visualize, verbosity=0)
            # After training is done, we save the final weights.
            if opts.save_file is not None:
                print("saving weights to [%s]" % opts.save_file)
                agent.save_weights(opts.save_file, overwrite=True)

            # Finally, evaluate our algorithm.
            agent.test(env, nb_episodes=opts.test_for,
                       visualize=self.visualize)

if __name__ == "__main__":
    """
    You can also run the trainer as a main class if you want to start your own
    agent/environment combination. If you know your precise arguments,
    just run this as your main.
    """
    trainer = Trainer.Trainer()
    time.sleep(5)
    parser = argparse.ArgumentParser()

    # add all parsing options
    Trainer.add_opts(parser)

    opts, unknown = parser.parse_known_args()

    exec("from agents import %s" % opts.agent)  # import agent type
    exec("from pybullet_envs.gym_locomotion_envs import %s" % opts.env)
    exec("%s.add_opts(parser)" % opts.agent)
    exec("%s.add_opts(parser)" % opts.env)

    # parse arguments
    opts, unknown = parser.parse_known_args()
    print("OPTS", opts)
    print("UNKNOWN", unknown)

    trainer.setup_exercise(opts)
