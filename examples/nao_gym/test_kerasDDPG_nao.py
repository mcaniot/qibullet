import datetime
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
datenow = '{:%Y%m%d%H%M%S}'.format(datetime.datetime.now())
opts.load_file = "checkpoints/%s-%s-%s.h5" % (opts.agent, opts.env, "test-3-v2")

print("\n OPTS", opts)
trainer.setup_exercise(opts)
