import argparse
import os

import rl_zoo3
import rl_zoo3.train

import Environments
from Models.EnTRPO.EnTRPO import EnTRPO, EnTRPOHigh, EnTRPOLow, sample_entrpo_params
from Models.Experimental.TRPOQ.TRPOQ import TRPOQ, sample_trpoq_params
from Models.Experimental.TRPOQ.TRPOQ2 import TRPOQ2, sample_trpoq2_params
from Models.Experimental.TRPOQ.TRPOQH import TRPOQH, TRPOQHO, sample_trpoqh_params, sample_trpoqho_params
from Models.GenTRPO.GenTRPO import GenTRPO, sample_gentrpo_params
from Models.SB3 import PPO, TRPO, sample_ppo_params, sample_trpo_params
from Models.TRPOER.TRPOER import TRPOER, sample_trpoer_params
from Models.TRPOR.TRPOR import TRPOR, sample_trpor_params
from Zoo.Configure import configure

# Register models
models = {
  "entrpo": {"model": EnTRPO, "sample": sample_entrpo_params},
  "entrpolow": {"model": EnTRPOLow, "sample": sample_entrpo_params},
  "entrpohigh": {"model": EnTRPOHigh, "sample": sample_entrpo_params},
  "trpoq": {"model": TRPOQ, "sample": sample_trpoq_params},
  "trpoq2": {"model": TRPOQ2, "sample": sample_trpoq2_params},
  "trpor": {"model": TRPOR, "sample": sample_trpor_params},
  "trpoqh": {"model": TRPOQH, "sample": sample_trpoqh_params},
  "trpoqho": {"model": TRPOQHO, "sample": sample_trpoqho_params},
  "trpo": {"model": TRPO, "sample": sample_trpo_params},
  "ppo": {"model": PPO, "sample": sample_ppo_params},
  "gentrpo": {"model": GenTRPO, "sample": sample_gentrpo_params},
  "trpoer": {"model": TRPOER, "sample": sample_trpoer_params},
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
  parser.add_argument("--model", type=str, default="trpo")
  parser.add_argument("--env", type=str, default="LunarLanderContinuous-v3")
  parser.add_argument("--device", type=str, default="cpu")
  parser.add_argument("--optimize", type=bool, default=False)
  parser.add_argument("--conf_file", type=str, default=None)
  parser.add_argument("--trials", type=int, default=160)
  parser.add_argument("--n_jobs", type=int, default=10)
  parser.add_argument("--n_timesteps", type=int, default=0)

  params = parser.parse_args()
  model = params.model
  conf_file = params.conf_file

  if conf_file is None and os.path.exists(f".hyperparameters/{model.lower()}.yml"):
    conf_file = f".hyperparameters/{model.lower()}.yml"

  configure(
    algo=params.model,
    env=params.env,
    device=params.device,
    optimize_hyperparameters=params.optimize,
    conf_file=conf_file,
    n_trials=params.trials,
    n_timesteps=params.n_timesteps,
    n_jobs=params.n_jobs,
  )
