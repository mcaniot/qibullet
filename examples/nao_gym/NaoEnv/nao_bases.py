from pybullet_envs.robot_bases import XmlBasedRobot


class QibulletBasedRobot(XmlBasedRobot):
    """
    Base class for URDF .xml based robots.
    """

    def __init__(self,
                 robot_virtual,
                 robot_name,
                 action_dim,
                 obs_dim,
                 basePosition=[0, 0, 0],
                 baseOrientation=[0, 0, 0, 1],
                 fixed_base=False,
                 self_collision=True):
        XmlBasedRobot.__init__(self, robot_name, action_dim,
                               obs_dim, self_collision)

        self.robot_virtual = robot_virtual
        self.basePosition = basePosition
        self.baseOrientation = baseOrientation
        self.fixed_base = fixed_base
        self.is_loaded = False
        self.part_list = [
            'l_wrist', 'RBicep', 'r_wrist', 'r_sole', 'LHip',
            'RElbow', 'RPelvis', 'LPelvis', 'LBicep', 'LAnklePitch',
            'l_sole', 'RThigh', 'LThigh', 'RAnklePitch', 'l_ankle',
            'RHip', 'RForeArm', 'torso', 'LTibia', 'RTibia',
            'LShoulder', 'LElbow', 'Head', 'r_ankle', 'RShoulder',
            'LForeArm']
        self.jdict_list = ['RAnklePitch', 'LHipRoll', 'LKneePitch',
                           'RShoulderPitch', 'RHipRoll', 'RHipPitch',
                           'LHipYawPitch', 'RShoulderRoll', 'RHipYawPitch',
                           'LElbowYaw', 'LHipPitch', 'RAnkleRoll',
                           'LAnkleRoll', 'LShoulderRoll', 'RKneePitch',
                           'LElbowRoll', 'RElbowYaw', 'RElbowRoll',
                           'LAnklePitch', 'LShoulderPitch']

    def reset(self, bullet_client):
        self._p = bullet_client
        self.ordered_joints = []
        if not self.is_loaded:
                self.robot_virtual.loadRobot(self.basePosition,
                                             self.baseOrientation)
                self.is_loaded = True
        else:
                self.robot_virtual.goToPosture("Stand", 1)
        self.parts, self.jdict, self.ordered_joints, self.robot_body =\
            self.addToScene(
                self._p,
                self.robot_virtual.robot_model)
        for item in list(self.parts.keys()):
                if item not in self.part_list:
                        del self.parts[item]
        for item in list(self.jdict.keys()):
                if item not in self.jdict_list:
                        del self.jdict[item]
        self.ordered_joints = self.jdict.values()
        self.robot_specific_reset(self._p)

        s = self.calc_state()
        self.potential = self.calc_potential()

        return s

    def calc_potential(self):
        return 0
