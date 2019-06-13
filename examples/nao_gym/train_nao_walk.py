from pybullet_envs.gym_locomotion_envs import NaoBulletEnv, HumanoidBulletEnv
from baselines import deepq
import time
# import time
PATH_MODEL_FOLDER = "models/"


def callback(lcl, glb):
    total = sum(lcl['episode_rewards'][-101:-1]) / 100
    totalt = lcl['t']
    is_solved = totalt > 2000 and total >= -50
    return is_solved


def main():
    env = NaoBulletEnv(render=True)
    # env = HumanoidBulletEnv(render=True)
    env.reset()
    for _ in range(500):
        env.render()
        env.step(env.action_space.sample())
        # time.sleep(0.5)
    env.close()
    # model = deepq.models.mlp([64])
    # act = deepq.learn(
    #     env,
    #     q_func=model,
    #     lr=1e-3,
    #     max_timesteps=10000,
    #     buffer_size=50000,
    #     exploration_fraction=0.1,
    #     exploration_final_eps=0.02,
    #     print_freq=10)
    # print("Saving model to nao_walk_model.pkl")
    # act.save(PATH_MODEL_FOLDER + "nao_walk_model.pkl")


if __name__ == '__main__':
    main()
