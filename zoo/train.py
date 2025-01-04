import sys
import argparse

import rl_zoo3
import rl_zoo3.train
from rl_zoo3.train import train
from rl_zoo3.enjoy import enjoy

from sbx import DDPG, DQN, PPO, SAC, TD3, TQC, CrossQ

rl_zoo3.ALGOS["ddpg"] = DDPG
rl_zoo3.ALGOS["dqn"] = DQN
rl_zoo3.ALGOS["sac"] = SAC
rl_zoo3.ALGOS["ppo"] = PPO
rl_zoo3.ALGOS["td3"] = TD3
rl_zoo3.ALGOS["tqc"] = TQC
rl_zoo3.ALGOS["crossq"] = CrossQ
rl_zoo3.train.ALGOS = rl_zoo3.ALGOS
rl_zoo3.exp_manager.ALGOS = rl_zoo3.ALGOS

import sys
from rl_zoo3.train import train

if __name__ == "__main__":
  hyperparams = [
      "batch_size:256",
      "buffer_size:1000000",
      "gamma:0.99",
      "gradient_steps:1",
      "learning_rate:'lin_7.3e-4'",
      "learning_starts:10000",
      "policy:'MlpPolicy'",
      "policy_kwargs:dict(net_arch=[400,300])",
      "tau:0.01",
      "train_freq:1"
  ]

  sys.argv = [
      "python",
      "--algo", "tqc",
      "--env", "LunarLanderContinuous-v3",
      "--device", "cuda",
      "--optimize-hyperparameters",
      "--hyperparams"
  ] + hyperparams

  train()
