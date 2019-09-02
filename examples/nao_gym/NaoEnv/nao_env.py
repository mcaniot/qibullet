#!/usr/bin/env python
# coding: utf-8

import gym
import time
import numpy as np
from gym import spaces
import pybullet
from qibullet import SimulationManager

OBS_DIM = 65


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

        self.all_joints = [
            "HeadYaw",
            "HeadPitch",
            "LShoulderPitch",
            "LShoulderRoll",
            "LElbowYaw",
            "LElbowRoll",
            "LWristYaw",
            "LHand",
            "LHipYawPitch",
            "LHipRoll",
            "LHipPitch",
            "LKneePitch",
            "LAnklePitch",
            "LAnkleRoll",
            "RHipYawPitch",
            "RHipRoll",
            "RHipPitch",
            "RKneePitch",
            "RAnklePitch",
            "RAnkleRoll",
            "RShoulderPitch",
            "RShoulderRoll",
            "RElbowYaw",
            "RElbowRoll",
            "RWristYaw",
            "RHand"]

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
        self.counter = 0
        self.number_of_step_in_episode = 0
        self.last_step_reward = 0
        self._setupScene()

        obs_space = np.inf * np.ones([OBS_DIM])
        self.observation_space = spaces.Box(
            low=-obs_space,
            high=obs_space)

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
        self.number_of_step_in_episode += 1

        obs, reward = self._getState()
        return obs, reward, self.episode_over, {}

    def reset(self):
        """
        Resets the environment for a new episode
        """
        self.episode_over = False
        self.previous_x = 0
        self.counter = 0
        self.number_of_step_in_episode = 0
        self.last_step_reward = 0
        self._resetScene()

        obs, _ = self._getState()
        return obs

    def _hardResetJointState(self):
        for joint, position in zip(self.all_joints, self.starting_position):
            pybullet.setJointMotorControl2(
                self.nao.robot_model,
                self.nao.joint_dict[joint].getIndex(),
                pybullet.VELOCITY_CONTROL,
                targetVelocity=0,
                physicsClientId=self.client)
            pybullet.resetJointState(
                    self.nao.robot_model,
                    self.nao.joint_dict[joint].getIndex(),
                    position,
                    physicsClientId=self.client)
        self._resetJointState()

    def _resetJointState(self):
        self.nao.setAngles(self.all_joints,
                           self.starting_position, 1.0)

    def render(self, mode='human', close=False):
        pass

    def _setVelocities(self, joints, velocities):
        """
        Sets velocities on the robot joints
        """
        for joint, velocity in zip(joints, velocities):
            pybullet.setJointMotorControl2(
                self.nao.robot_model,
                self.nao.joint_dict[joint].getIndex(),
                pybullet.VELOCITY_CONTROL,
                targetVelocity=velocity,
                physicsClientId=self.client)

    def _getJointState(self, joint_name):
        """
        Returns the state of the joint in the world frame
        """
        position, velocity, _, _ =\
            pybullet.getJointState(
                self.nao.robot_model,
                self.nao.joint_dict[joint_name].getIndex(),
                physicsClientId=self.client)

        return position, velocity

    def _getLinkState(self, link_name):
        """
        Returns the state of the link in the world frame
        """
        (x, y, z), (qx, qy, qz, qw), _, _, _, _, (vx, vy, vz),\
            (vroll, vpitch, vyaw) = pybullet.getLinkState(
            self.nao.robot_model,
            self.nao.link_dict[link_name].getIndex(),
            computeLinkVelocity=1,
            physicsClientId=self.client)

        return (x, y, z), (qx, qy, qz, qw), (vx, vy, vz), (vroll, vpitch, vyaw)

    def _getContactFeet(self):
        foot_list = ["r_ankle", "l_ankle"]
        contact_list = []
        for foot_joint in foot_list:
            points = pybullet.getContactPoints(
                bodyA=self.nao.robot_model,
                linkIndexA=self.nao.link_dict[foot_joint].getIndex(),
                physicsClientId=self.client)
            if len(points) > 0:
                contact_list.append(1)
            else:
                contact_list.append(0)
        return contact_list

    def _getState(self, convergence_criteria=0.12, divergence_criteria=0.6):
        """
        Gets the observation and computes the current reward. Will also
        determine if the episode is over or not, by filling the episode_over
        boolean. When the euclidian distance between the wrist link and the
        cube is inferior to the one defined by the convergence criteria, the
        episode is stopped
        """
        # Get the information on the joints
        joint_position_list = []
        joint_velocity_list = []

        for joint in self.controlled_joints:
            pos, vel = self._getJointState(joint)
            joint_position_list.append(pos)
            joint_velocity_list.append(vel)

        joint_position_list = np.array(
            joint_position_list,
            dtype=np.float32
        )

        joint_velocity_list = np.array(
            joint_velocity_list,
            dtype=np.float32
        )

        feet_contact = np.array(
            self._getContactFeet(),
            dtype=np.float32
        )

        (x, y, z), (qx, qy, qz, qw), (vx, vy, vz), (vroll, vpitch, vyaw) =\
            self._getLinkState("torso")
        roll, pitch, yaw = pybullet.getEulerFromQuaternion([qx, qy, qz, qw])
        torso_state = np.array(
            [x, y, z, yaw, vx, vy, vz, vroll, vpitch, vyaw],
            dtype=np.float32
        )

        (x, y, z), (qx, qy, qz, qw), (vx, vy, vz), (vroll, vpitch, vyaw) =\
            self._getLinkState("r_ankle")
        roll, pitch, yaw = pybullet.getEulerFromQuaternion([qx, qy, qz, qw])
        r_ankle_state = np.array(
            [x, y, z, vx, vy, vz],
            dtype=np.float32
        )

        (x, y, z), (qx, qy, qz, qw), (vx, vy, vz), (vroll, vpitch, vyaw) =\
            self._getLinkState("l_ankle")
        roll, pitch, yaw = pybullet.getEulerFromQuaternion([qx, qy, qz, qw])
        l_ankle_state = np.array(
            [x, y, z, vx, vy, vz],
            dtype=np.float32
        )

        self.counter = self.number_of_step_in_episode / 3
        counter = np.array(
            [self.counter],
            dtype=np.float32
        )

        # Fill the observation
        obs = np.concatenate(
            [counter] +
            [joint_position_list] +
            [joint_velocity_list] +
            [torso_state] +
            [r_ankle_state] +
            [l_ankle_state] +
            [feet_contact])

        reward = 0
        # To be passed to True when the episode is over
        if torso_state[2] < 0.27 or torso_state[0] > 14:
            if torso_state[2] < 0.27:
                reward += -100
            if torso_state[0] > 14:
                reward += 100
            reward += int(torso_state[0] / 0.08) * 10
            self.episode_over = True
        # Compute the reward
        # delta x : speed
        reward += (torso_state[0] - self.previous_x) * 100
        if self.counter - self.last_step_reward > 200:
            self.last_step_reward = self.counter
            if l_ankle_state[3] >= 0:
                reward += 0.00001 * l_ankle_state[3]
            if r_ankle_state[3] >= 0:
                reward += 0.00001 * r_ankle_state[3]

        self.previous_x = torso_state[0]
        return obs, reward

    def _setupScene(self):
        """
        Setup a scene environment within the simulation
        """
        self.client = self.simulation_manager.launchSimulation(gui=self.gui)
        self.nao = self.simulation_manager.spawnNao(
            self.client,
            spawn_ground_plane=True)
        self._resetJointState()
        time.sleep(1.0)

    def _resetScene(self):
        """
        Resets the scene for a new scenario
        """

        pybullet.resetBasePositionAndOrientation(
            self.nao.robot_model,
            posObj=[0.0, 0.0, 0.36],
            ornObj=[0.0, 0.0, 0.0, 1.0],
            physicsClientId=self.client)
        balance_constraint = pybullet.createConstraint(
            parentBodyUniqueId=self.nao.robot_model,
            parentLinkIndex=-1,
            childBodyUniqueId=-1,
            childLinkIndex=-1,
            jointType=pybullet.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0],
            parentFrameOrientation=[0, 0, 0, 1],
            childFramePosition=[0.0, 0.0, 0.36],
            childFrameOrientation=[0.0, 0.0, 0.0, 1.0],
            physicsClientId=self.client)
        self._hardResetJointState()
        pybullet.removeConstraint(
            balance_constraint,
            physicsClientId=self.client)
        time.sleep(0.5)

    def _termination(self):
        """
        Terminates the environment
        """
        self.simulation_manager.stopSimulation(self.client)


# def main():
#     env = NaoEnv(gui=True)
#
#     # Test observation space and action space sampling
#     env.observation_space.sample()
#     action = env.action_space.sample()
#
#     # Test resetting environment
#     state = env.reset()
#
#     # Test computing a step
#     next_state, reward, done, _ = env.step(action.tolist())
#
#     # Terminate env
#     env._termination()
#
#
# if __name__ == "__main__":
#     main()
