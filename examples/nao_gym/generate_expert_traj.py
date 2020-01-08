import sys
CV2_ROS = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if CV2_ROS in sys.path:
    sys.path.remove(CV2_ROS)
    sys.path.append(CV2_ROS)
from stable_baselines.gail import generate_expert_traj
from NaoEnv import NaoEnvPretrained

def main():
    env = NaoEnvPretrained(gui=True)

    generate_expert_traj(
        env.walking_expert_speed,
        'models_nao/walk_pretrained_with_speed',
        env,
        n_episodes=1)

if __name__ == "__main__":
    main()
