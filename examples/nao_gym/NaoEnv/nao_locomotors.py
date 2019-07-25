from NaoEnv.nao_bases import QibulletBasedRobot
import numpy as np
from qibullet import NaoVirtual


class WalkerBaseQibullet(QibulletBasedRobot):

    def __init__(self, fn, robot_name, action_dim, obs_dim, power):
        QibulletBasedRobot.__init__(self, fn, robot_name, action_dim, obs_dim)
        self.power = power
        self.camera_x = 0
        self.start_pos_x, self.start_pos_y, self.start_pos_z = 0, 0, 0
        self.walk_target_x = 1e3    # kilometer away
        self.walk_target_y = 0
        self.body_xyz = [0, 0, 0]
        self.list_id = []
        self.vcc_init = None
        self.previous_theta_target = None

    def robot_specific_reset(self, bullet_client):
        self._p = bullet_client
        for j in self.ordered_joints:
            j.reset_current_position(
                self.np_random.uniform(low=-0.1, high=0.1) + j.get_position(),
                0)

        self.feet = [self.parts[f] for f in self.foot_list]
        self.feet_contact = np.array([0.0 for f in self.foot_list],
                                     dtype=np.float32)
        self.scene.actor_introduce(self)
        self.initial_z = None

    def apply_action(self, a):
        assert (np.isfinite(a).all())
        if self.previous_theta_target is None:
            self.previous_theta_target = [0] * len(a)
        for n, j in enumerate(self.ordered_joints):
            try:
                k = 7
                x_target = np.clip(a[n], -1, +1)
                theta_target = (x_target + 1) *\
                    (j.lowerLimit - j.upperLimit)/2 + j.upperLimit
                speed_target = (theta_target -
                                self.previous_theta_target[n]) / k
                j.set_velocity(
                    float(np.clip(speed_target, -k, +k)))
                self.previous_theta_target[n] = x_target
            except Exception as e:
                print(e)
                self.list_id.append(n)

    def calc_state(self):
        j = np.array(
            [j.current_relative_position() for j in self.ordered_joints],
            dtype=np.float32).flatten()
        # even elements [0::2] position, scaled to -1..+1 between limits
        # odd elements    [1::2] angular speed, scaled to show -1..+1
        self.joint_speeds = j[1::2]
        self.joints_at_limit = np.count_nonzero(np.abs(j[0::2]) > 0.99)

        body_pose = self.robot_body.pose()
        parts_xyz = np.array(
            [p.pose().xyz() for p in self.parts.values()]).flatten()
        self.body_xyz = (parts_xyz[0::3].mean(),
                         parts_xyz[1::3].mean(), body_pose.xyz()[2]
                         )  # torso z is more informative than mean z
        self.body_rpy = body_pose.rpy()
        r, p, yaw = self.body_rpy
        z = self.body_xyz[2]
        if self.initial_z is None:
            self.initial_z = z
        self.walk_target_theta = np.arctan2(
            self.walk_target_y - self.body_xyz[1],
            self.walk_target_x - self.body_xyz[0])
        self.walk_target_dist = np.linalg.norm(
                [self.walk_target_y - self.body_xyz[1],
                 self.walk_target_x - self.body_xyz[0]])
        (vx, vy, vz), (vroll, vpitch, vyaw) = self.getBodySpeed()
        fsr_force_z = np.array(
            self.getFsrValue(),
            dtype=np.float32
        )
        acc_x, acc_y, acc_z = [0, 0, 0]
        if self.vcc_init is None:
            self.vcc_init = [vx, vy, vz]
        else:
            acc_x, acc_y, acc_z = [
                (vx - self.vcc_init[0])/self.scene.dt,
                (vy - self.vcc_init[1])/self.scene.dt,
                (vz - self.vcc_init[2])/self.scene.dt,
            ]
        variables_in_array = np.array(
            [
                self.body_xyz[2],
                self.body_rpy[2],
                acc_x,
                acc_y,
                acc_z,
                vroll,
                vpitch,
                vyaw
            ],
            dtype=np.float32
        )
        ob_space = np.clip(
                    np.concatenate(
                        [variables_in_array] +
                        [fsr_force_z] +
                        [j] +
                        [self.feet_contact]),
                    -5, +5)
        return ob_space

    def calc_potential(self):
        # progress in potential field is speed*dt, typical speed is about
        # 2-3 meter per second, this potential will change 2-3 per frame
        # (not per second),
        # all rewards have rew/frame units and close to 1.0
        debugmode = 0
        if (debugmode):
            print("calc_potential: self.walk_target_dist")
            print(self.walk_target_dist)
            print("self.scene.dt")
            print(self.scene.dt)
            print("self.scene.frame_skip")
            print(self.scene.frame_skip)
            print("self.scene.timestep")
            print(self.scene.timestep)
        return -self.walk_target_dist / self.scene.dt


class Nao(WalkerBaseQibullet):
    random_yaw = False
    foot_list = ["r_sole", "l_sole"]

    def __init__(self):
        WalkerBaseQibullet.__init__(self,
                                    NaoVirtual(),
                                    "torso",
                                    action_dim=20,
                                    obs_dim=58,
                                    power=0.020)

    def alive_bonus(self, z, pitch):
        knees = np.array(
            [j.current_relative_position() for j in [self.jdict["LKneePitch"],
                                                     self.jdict["RKneePitch"]
                                                     ]],
            dtype=np.float32).flatten()
        knees_at_limit = np.count_nonzero(np.abs(knees[0::2]) > 0.99)
        return +4-knees_at_limit if z > 0.25 else -1

    def robot_specific_reset(self, bullet_client):
        WalkerBaseQibullet.robot_specific_reset(self, bullet_client)
        self.head = self.parts["Head"]
