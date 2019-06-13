import NaoEnv


def main():
    env = NaoEnv.NaoBulletEnv(render=True)
    env.reset()
    for _ in range(500):
        env.render()
        env.step(env.action_space.sample())
    env.close()


if __name__ == '__main__':
    main()
