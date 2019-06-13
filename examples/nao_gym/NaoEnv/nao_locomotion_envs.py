from pybullet_envs.gym_locomotion_envs import WalkerBaseBulletEnv
from NaoEnv.nao_locomotors import Nao


class NaoBulletEnv(WalkerBaseBulletEnv):

    def __init__(self, render=False):
        self.robot = Nao()
        WalkerBaseBulletEnv.__init__(self, self.robot, render)
