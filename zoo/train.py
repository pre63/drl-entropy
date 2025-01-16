import argparse
import os
import time
from math import inf
from statistics import mean, stdev

import matplotlib.pyplot as plt
import pandas as pd
import rl_zoo3
import rl_zoo3.train
import yaml
from sbx import SAC, TQC

import Environments
from Models.EnTRPO.EnTRPO import EnTRPO, EnTRPOHigh, EnTRPOLow, sample_entrpo_params
from Models.EnTRPOR.EnTRPOR import EnTRPOR, sample_entrpor_params
from Models.PPO import PPO
from Models.TRPO import TRPO
from Models.TRPOQ.TRPOQ import TRPOQ, sample_trpoq_params
from Models.TRPOQ.TRPOQ2 import TRPOQ2, sample_trpoq2_params
from Models.TRPOQ.TRPOQH import TRPOQH, TRPOQHO, sample_trpoqh_params, sample_trpoqho_params
from Models.TRPOR.TRPOR import TRPOR, sample_trpor_params
from zoo.configure import configure

# Register models
models = {
    "entrpo": {"model": EnTRPO, "sample": sample_entrpo_params},
    "entrpolow": {"model": EnTRPOLow, "sample": sample_entrpo_params},
    "entrpohigh": {"model": EnTRPOHigh, "sample": sample_entrpo_params},
    "trpoq": {"model": TRPOQ, "sample": sample_trpoq_params},
    "trpoq2": {"model": TRPOQ2, "sample": sample_trpoq2_params},
    "trpor": {"model": TRPOR, "sample": sample_trpor_params},
    "entrpor": {"model": EnTRPOR, "sample": sample_entrpor_params},
    "trpoqh": {"model": TRPOQH, "sample": sample_trpoqh_params},
    "trpoqho": {"model": TRPOQHO, "sample": sample_trpoqho_params},
    "sac": {"model": SAC},
    "tqc": {"model": TQC},
    "trpo": {"model": TRPO},
    "ppo": {"model": PPO},
}

for model_name, value in models.items():
  model_class = value["model"]
  rl_zoo3.ALGOS[model_name] = model_class
  sample = value["sample"] if "sample" in value else None
  if sample is not None:
    rl_zoo3.hyperparams_opt.HYPERPARAMS_SAMPLER[model_name] = sample

rl_zoo3.train.ALGOS = rl_zoo3.ALGOS
rl_zoo3.exp_manager.ALGOS = rl_zoo3.ALGOS

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", type=str, default="ppo")
  parser.add_argument("--env", type=str, default="LunarLanderContinuous-v3")
  parser.add_argument("--envs", type=int, default=10)
  parser.add_argument("--device", type=str, default="cpu")
  parser.add_argument("--optimize", type=bool, default=False)
  parser.add_argument("--conf_file", type=str, default=None)
  parser.add_argument("--trials", type=int, default=40)
  parser.add_argument("--n_jobs", type=int, default=10)
  parser.add_argument("--n_timesteps", type=int, default=1000000)

  params = parser.parse_args()
  model = params.model
  conf_file = params.conf_file

  if conf_file is None and os.path.exists(f"Hyperparameters/{model.lower()}.yml"):
    conf_file = f"Hyperparameters/{model.lower()}.yml"

  configure(
      algo=params.model,
      env=params.env,
      n_eval_envs=params.envs,
      device=params.device,
      optimize_hyperparameters=params.optimize,
      conf_file=conf_file,
      n_trials=params.trials,
      n_timesteps=params.n_timesteps,
      n_jobs=params.n_jobs,
  )
