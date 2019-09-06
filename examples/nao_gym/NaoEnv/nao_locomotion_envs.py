import numpy as np
from pybullet_envs.scene_stadium import SinglePlayerStadiumScene
from pybullet_envs.gym_locomotion_envs import WalkerBaseBulletEnv
from pybullet_envs.env_bases import MJCFBaseBulletEnv
import pybullet
from NaoEnv.nao_locomotors import Nao


class WalkerBaseQibulletEnv(MJCFBaseBulletEnv):
    def __init__(self, robot, render=False):
        # print("WalkerBase::__init__ start")
        MJCFBaseBulletEnv.__init__(self, robot, render)

        self.camera_x = 0
        self.walk_target_x = 1e3  # kilometer away
        self.walk_target_y = 0
        self.stateId = -1

    def create_single_player_scene(self, bullet_client):
        self.stadium_scene = SinglePlayerStadiumScene(
            bullet_client,
            gravity=9.8,
            # timestep=0.0165 / 4,
            timestep=0.01,
            frame_skip=1)
        return self.stadium_scene

    def reset(self):
        if (self.stateId >= 0):
            # print("restoreState self.stateId:",self.stateId)
            self._p.restoreState(self.stateId)

        r = MJCFBaseBulletEnv.reset(self)
        self._p.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 0)

        self.parts, self.jdict, self.ordered_joints, self.robot_body =\
            self.robot.addToScene(
                self._p, self.stadium_scene.ground_plane_mjcf)
        self.ground_ids = set([(self.parts[f].bodies[self.parts[f].bodyIndex],
                                self.parts[f].bodyPartIndex)
                               for f in self.foot_ground_object_names])
        self._p.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 1)
        if (self.stateId < 0):
            self.stateId = self._p.saveState()
            # print("saving state self.stateId:",self.stateId)

        return r

    def _isDone(self):
        return self._alive < 0

    def move_robot(self, init_x, init_y, init_z):
        "Used by multiplayer stadium to move sideways,"
        "to another running lane."
        self.cpp_robot.query_position()
        pose = self.cpp_robot.root_part.pose()
        pose.move_xyz(
            init_x, init_y, init_z
        )
        self.cpp_robot.set_pose(pose)

    electricity_cost = -2.0
    stall_torque_cost = -0.1
    foot_collision_cost = -1.0
    foot_ground_object_names = set(["floor"])
    joints_at_limit_cost = -0.1

    def step(self, a):
        if not self.scene.multiplayer:
            self.robot.apply_action(a)
            self.scene.global_step()

        state = self.robot.calc_state()
        self._alive = float(
            self.robot.alive_bonus(
                state[0] + self.robot.initial_z,
                self.robot.body_rpy[1]))
        done = self._isDone()
        if not np.isfinite(state).all():
            print("~INF~", state)
            done = True

        potential_old = self.potential
        self.potential = self.robot.calc_potential()
        progress = float(self.potential - potential_old)

        feet_collision_cost = 0.0
        for i, f in enumerate(self.robot.feet):
            contact_ids = set((x[2], x[4]) for x in f.contact_list())
            if (self.ground_ids & contact_ids):
                self.robot.feet_contact[i] = 1.0
            else:
                self.robot.feet_contact[i] = 0.0

        electricity_cost = self.electricity_cost * float(
            np.abs(a * self.robot.joint_speeds).mean())
        electricity_cost += self.stall_torque_cost * float(np.square(a).mean())

        joints_at_limit_cost = float(
                self.joints_at_limit_cost * self.robot.joints_at_limit)
        debugmode = 0
        if (debugmode):
            print("alive=")
            print(self._alive)
            print("progress")
            print(progress)
            print("electricity_cost")
            print(electricity_cost)
            print("joints_at_limit_cost")
            print(joints_at_limit_cost)
            print("feet_collision_cost")
            print(feet_collision_cost)

        self.rewards = [
            self._alive, progress, electricity_cost, joints_at_limit_cost,
            feet_collision_cost
        ]
        if (debugmode):
            print("rewards=")
            print(self.rewards)
            print("sum rewards")
            print(sum(self.rewards))
        self.HUD(state, a, done)
        self.reward += sum(self.rewards)

        return state, sum(self.rewards), bool(done), {}

    def camera_adjust(self):
        x, y, z = self.body_xyz
        self.camera_x = 0.98 * self.camera_x + (1 - 0.98) * x
        self.camera.move_and_look_at(self.camera_x, y - 2.0, 1.4, x, y, 1.0)


class NaoBulletEnv(WalkerBaseQibulletEnv):

    def __init__(self, render=False):
        self.robot = Nao()
        WalkerBaseBulletEnv.__init__(self, self.robot, render)
