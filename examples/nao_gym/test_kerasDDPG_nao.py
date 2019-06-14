import argparse
import Trainer

trainer = Trainer.Trainer()
argparser = argparse.ArgumentParser()
Trainer.add_opts(argparser)

# precoded options
opts = argparser.parse_args()
opts.agent = "KerasDDPGAgent-v0"
opts.env = "NaoBulletEnv"
opts.train_for = 0
opts.test_for = 100
opts.load_file = "checkpoints/%s-%s.h5" % (opts.agent, opts.env)

print("\n OPTS", opts)
trainer.setup_exercise(opts)
