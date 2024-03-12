from discretised.trainer import DiscretisedBFNTrainer
# 150e3a3656bc3e6c76366ee98da5b0fd9f7c16ea
checkpoint_file = '/home/rfsm2/rds/hpc-work/MLMI4/bfn_model_checkpoint.pth'
trainer = DiscretisedBFNTrainer(checkpoint_file=checkpoint_file)
trainer.train()