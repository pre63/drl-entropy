import sys
import argparse

import rl_zoo3
import rl_zoo3.train
from rl_zoo3.train import train
from rl_zoo3.enjoy import enjoy

from sbx import DDPG, DQN, PPO, SAC, TD3, TQC, CrossQ

from Models.EnTRPO.EnTRPO import EnTRPO, sample_entrpo_params
from Models.TRPO.TRPO import TRPO

rl_zoo3.ALGOS["ddpg"] = DDPG
rl_zoo3.ALGOS["dqn"] = DQN
rl_zoo3.ALGOS["sac"] = SAC
rl_zoo3.ALGOS["ppo"] = PPO
rl_zoo3.ALGOS["td3"] = TD3
rl_zoo3.ALGOS["tqc"] = TQC
rl_zoo3.ALGOS["crossq"] = CrossQ

rl_zoo3.ALGOS["entrpo"] = EnTRPO
rl_zoo3.hyperparams_opt.HYPERPARAMS_SAMPLER["entrpo"] = sample_entrpo_params

rl_zoo3.ALGOS["trpo"] = TRPO


rl_zoo3.train.ALGOS = rl_zoo3.ALGOS
rl_zoo3.exp_manager.ALGOS = rl_zoo3.ALGOS
rl_zoo3.hyperparams_opt.HYPERPARAMS_SAMPLER = rl_zoo3.hyperparams_opt.HYPERPARAMS_SAMPLER


import sys
from rl_zoo3.train import train

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", type=str, default="ppo")

  params = parser.parse_args()
  model = params.model

  sys.argv = [
      "python",
      "--algo", model,
      "--env", "LunarLanderContinuous-v3",
      "--device", "cuda",
      "--optimize-hyperparameters"
  ]

  if model == "entrpo":
    sys.argv += ["--conf-file", "Models/EnTRPO/EnTRPO.yaml"]

  train()
