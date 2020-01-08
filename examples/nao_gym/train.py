from baselines_tools import *

def main():
    seed = int(time.time())
    np.random.seed(seed)
    # train the model
    train(num_timesteps=int(20e7), seed=seed,
          model_path=PATH_MODEL)

if __name__ == '__main__':
    main()