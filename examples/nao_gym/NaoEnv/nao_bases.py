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
        self.fsr_index_list = []
        self.jdict_list = ['RAnklePitch', 'LHipRoll', 'LKneePitch',
                           'RShoulderPitch', 'RHipRoll', 'RHipPitch',
                           'LHipYawPitch', 'RShoulderRoll', 'RHipYawPitch',
                           'LElbowYaw', 'LHipPitch', 'RAnkleRoll',
                           'LAnkleRoll', 'LShoulderRoll', 'RKneePitch',
                           'LElbowRoll', 'RElbowYaw', 'RElbowRoll',
                           'LAnklePitch', 'LShoulderPitch']

    def resetPoseStance(self):
        joint_names = [
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
        position_values = [
            0,
            0,
            1.57079632,
            0,
            -1.57079632,
            -1.57079632,
            0,
            0,
            0,
            0,
            -0.523598775,
            1.04719755,
            -0.523598775,
            0,
            0,
            0,
            -0.523598775,
            1.04719755,
            -0.523598775,
            0,
            1.570796326,
            0,
            1.570796326,
            1.570796326,
            0,
            0]
        for i in range(0, len(joint_names)):
            joint = joint_names[i]
            position = position_values[i]
            self._p.resetJointState(
                    self.robot_virtual.robot_model,
                    self.robot_virtual.joint_dict[joint].getIndex(),
                    position,
                    physicsClientId=self.robot_virtual.getPhysicsClientId())

    def enableFsrSensor(self):
        num_joint = self._p.getNumJoints(
            self.robot_virtual.robot_model,
            physicsClientId=self.robot_virtual.getPhysicsClientId())
        for index in range(0, num_joint):
            joint_info = self._p.getJointInfo(
                self.robot_virtual.robot_model,
                index,
                physicsClientId=self.robot_virtual.getPhysicsClientId()
            )
            if "FSR" in joint_info[1].decode('utf-8'):
                self.fsr_index_list.append(index)
        for fsr in self.fsr_index_list:
            self._p.enableJointForceTorqueSensor(
                self.robot_virtual.robot_model,
                fsr,
                True,
                physicsClientId=self.robot_virtual.getPhysicsClientId()
            )

    def getFsrValue(self):
        fsr_value_list = []
        for fsr in self.fsr_index_list:
            _, _, reaction_forces, motor_torque =\
                self._p.getJointState(
                    self.robot_virtual.robot_model,
                    fsr,
                    physicsClientId=self.robot_virtual.getPhysicsClientId()
                )
            fsr_value_list.append(reaction_forces[2])
        return fsr_value_list

    def getBodySpeed(self):
        (x, y, z), (a, b, c, d), _, _, _, _, (vx, vy, vz),\
            (vroll, vpitch, vyaw) =\
            self._p.getLinkState(
                self.robot_body.bodies[self.robot_body.bodyIndex],
                self.robot_body.bodyPartIndex, computeLinkVelocity=1,
                physicsClientId=self.robot_virtual.getPhysicsClientId())
        return (vx, vy, vz), (vroll, vpitch, vyaw)

    def reset(self, bullet_client):
        self._p = bullet_client
        self.ordered_joints = []
        if not self.is_loaded:
            self.robot_virtual.loadRobot(self.basePosition,
                                         self.baseOrientation)
            self.is_loaded = True
        self.resetPoseStance()
        self.enableFsrSensor()
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
