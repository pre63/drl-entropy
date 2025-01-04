import sys
import argparse

import rl_zoo3
import rl_zoo3.train
from rl_zoo3.train import train
from rl_zoo3.enjoy import enjoy

from sbx import DDPG, DQN, PPO, SAC, TD3, TQC, CrossQ

from Models.EnTRPO.EnTRPO import EnTRPO, sample_entrpo_params
from Models.TRPOQ.TRPOQ import TRPOQ, sample_trpoq_params
from Models.TRPOQ.TRPOQ2 import TRPOQ2

rl_zoo3.ALGOS["ddpg"] = DDPG
rl_zoo3.ALGOS["dqn"] = DQN
rl_zoo3.ALGOS["sac"] = SAC
rl_zoo3.ALGOS["ppo"] = PPO
rl_zoo3.ALGOS["td3"] = TD3
rl_zoo3.ALGOS["tqc"] = TQC
rl_zoo3.ALGOS["crossq"] = CrossQ

rl_zoo3.ALGOS["entrpo"] = EnTRPO
rl_zoo3.hyperparams_opt.HYPERPARAMS_SAMPLER["entrpo"] = sample_entrpo_params

rl_zoo3.ALGOS["trpoq"] = TRPOQ
rl_zoo3.ALGOS["trpoq2"] = TRPOQ2
rl_zoo3.hyperparams_opt.HYPERPARAMS_SAMPLER["trpoq"] = sample_trpoq_params
rl_zoo3.hyperparams_opt.HYPERPARAMS_SAMPLER["trpoq2"] = sample_trpoq_params

rl_zoo3.train.ALGOS = rl_zoo3.ALGOS
rl_zoo3.exp_manager.ALGOS = rl_zoo3.ALGOS
rl_zoo3.hyperparams_opt.HYPERPARAMS_SAMPLER = rl_zoo3.hyperparams_opt.HYPERPARAMS_SAMPLER

from zoo.configure import configure

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", type=str, default="ppo")
  parser.add_argument("--envs", type=int, default=4)

  params = parser.parse_args()
  model = params.model

  if model == "entrpo":
    conf_file = "Models/EnTRPO/EnTRPO.yml"
  elif model == "trpoq" or model == "trpoq2":
    conf_file = "Models/TRPOQ/TRPOQ.yml"

  configure(
      algo=model,
      env="LunarLanderContinuous-v3",
      device="cuda",
      optimize_hyperparameters=True,
      conf_file=conf_file,
      n_eval_envs=params.envs,
  )
