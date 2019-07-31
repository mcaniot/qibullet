#!/usr/bin/env python
# coding: utf-8

import gym
import time
import numpy as np
from gym import spaces
import pybullet
import pybullet_data
from qibullet import NaoVirtual
from qibullet import SimulationManager


class NaoEnv(gym.Env):
    """
    Gym environment for the NAO robot, walking task
    """

    def __init__(self, gui=False):
        """
        Constructor

        Parameters:
            gui - boolean, the simulation is in DIRECT mode if set to False
            (default value)
        """
        self.controlled_joints = [
            'RAnklePitch',
            'LHipRoll',
            'LKneePitch',
            'RShoulderPitch',
            'RHipRoll',
            'RHipPitch',
            'LHipYawPitch',
            'RShoulderRoll',
            'RHipYawPitch',
            'LElbowYaw',
            'LHipPitch',
            'RAnkleRoll',
            'LAnkleRoll',
            'LShoulderRoll',
            'RKneePitch',
            'LElbowRoll',
            'RElbowYaw',
            'RElbowRoll',
            'LAnklePitch',
            'LShoulderPitch']

        self.starting_position = [
            0.0,
            0.0,
            1.57079632,
            0.0,
            -1.57079632,
            -1.57079632,
            0.0,
            0.0,
            0.0,
            0.0,
            -0.523598775,
            1.04719755,
            -0.523598775,
            0.0,
            0.0,
            0.0,
            -0.523598775,
            1.04719755,
            -0.523598775,
            0.0,
            1.570796326,
            0.0,
            1.570796326,
            1.570796326,
            0.0,
            0.0]

        # Passed to True at the end of an episode
        self.episode_over = False
        self.gui = gui
        self.simulation_manager = SimulationManager()

        self._setupScene()

        # TODO; to be specified
        self.observation_space = spaces.Box(
            low=np.array([-0.2, -0.6, 0.3]),
            high=np.array([0.6, 0.2, 0.8]))

        max_velocities = [self.nao.joint_dict[joint].getMaxVelocity() for
                          joint in self.controlled_joints]
        min_velocities = [-self.nao.joint_dict[joint].getMaxVelocity() for
                          joint in self.controlled_joints]

        self.action_space = spaces.Box(
            low=np.array(min_velocities),
            high=np.array(max_velocities))

    def step(self, action):
        """

        Parameters
        ----------
        action : list of velocities to be applied on the robot's joints

        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
                an environment-specific object representing your observation of
                the environment.
            reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over (bool) :
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info (dict) :
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """
        try:
            action = list(action)
            assert len(action) == len(self.controlled_joints)

        except AssertionError:
            print("Incorrect action")
            return None, None, None, None

        self._setVelocities(self.controlled_joints, action)

        obs, reward = self._getState()
        return obs, reward, self.episode_over, {}

    def reset(self):
        """
        Resets the environment for a new episode
        """
        self.episode_over = False
        self._resetScene()

        obs = None
        # TODO Fill and return the observation
        return obs

    def render(self, mode='human', close=False):
        pass

    def _setVelocities(self, angles, velocities):
        """
        Sets velocities on the robot joints
        """
        for angle, velocity in zip(angles, velocities):
            position = self.nao.getAnglesPosition(angle)

            lower_limit = self.nao.joint_dict[angle].getLowerLimit()
            upper_limit = self.nao.joint_dict[angle].getUpperLimit()

            pybullet.setJointMotorControl2(
                self.nao.robot_model,
                self.nao.joint_dict[angle].getIndex(),
                pybullet.VELOCITY_CONTROL,
                targetVelocity=velocity)

    def _getLinkPosition(self, link_name):
        """
        Returns the position of the link in the world frame
        """
        link_state = pybullet.getLinkState(
            self.nao.robot_model,
            self.nao.link_dict[link_name].getIndex())

        return link_state[0], link_state[1]

    def _getState(self, convergence_criteria=0.12, divergence_criteria=0.6):
        """
        Gets the observation and computes the current reward. Will also
        determine if the episode is over or not, by filling the episode_over
        boolean. When the euclidian distance between the wrist link and the
        cube is inferior to the one defined by the convergence criteria, the
        episode is stopped
        """
        # Get the position of the torso in the world
        torso_pos, _ = self._getLinkPosition("torso")

        # Fill the observation
        obs = None

        # To be passed to True when the episode is over
        # self.episode_over = True

        # Compute the reward
        reward = None

        return obs, reward

    def _setupScene(self):
        """
        Setup a scene environment within the simulation
        """
        self.client = self.simulation_manager.launchSimulation(gui=self.gui)
        self.nao = self.simulation_manager.spawnNao(
            self.client,
            spawn_ground_plane=True)

        self.nao.setAngles(self.controlled_joints, self.starting_position, 1.0)
        time.sleep(1.0)

    def _resetScene(self):
        """
        Resets the scene for a new scenario
        """
        self.nao.goToPosture("Stand", 1.0)

        pybullet.resetBasePositionAndOrientation(
            self.nao.robot_model,
            posObj=[0.0, 0.0, 0.36],
            ornObj=[0.0, 0.0, 0.0, 1.0],
            physicsClientId=self.client)

        self.nao.setAngles(self.controlled_joints, self.starting_position, 1.0)
        time.sleep(1.0)

    def _termination(self):
        """
        Terminates the environment
        """
        self.simulation_manager.stopSimulation(self.client)


def main():
    env = NaoEnv(gui=False)

    # Test observation space and action space sampling
    env.observation_space.sample()
    action = env.action_space.sample()

    # Test resetting environment
    state = env.reset()

    # Test computing a step
    next_state, reward, done, _ = env.step(action.tolist())

    # Terminate env
    env._termination()


if __name__ == "__main__":
    main()
