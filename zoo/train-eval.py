import argparse
import os
import time
from math import inf
from statistics import mean, stdev

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rl_zoo3
import rl_zoo3.train
import yaml

import Environments
from Models.EnTRPO.EnTRPO import EnTRPO, EnTRPOHigh, EnTRPOLow, sample_entrpo_params
from Models.EnTRPOR.EnTRPOR import EnTRPOR, sample_entrpor_params
from Models.SB3 import PPO, TRPO
from Models.SBX import SAC, TQC
from Models.TRPOQ.TRPOQ import TRPOQ, sample_trpoq_params
from Models.TRPOQ.TRPOQ2 import TRPOQ2, sample_trpoq2_params
from Models.TRPOQ.TRPOQH import TRPOQH, TRPOQHO, sample_trpoqh_params, sample_trpoqho_params
from Models.TRPOR.TRPOR import TRPOR, sample_trpor_params
from zoo.configure import configure
from zoo.train_eval_plot import plot_all_from_csv

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


def load_reward_threshold(conf_file, env):
  with open(conf_file, "r") as file:
    config = yaml.safe_load(file)
  if env in config and "reward_threshold" in config[env]:
    return config[env]["reward_threshold"]
  raise ValueError(f"Reward threshold not found for environment: {env}")


def evaluate_training(algo, env, n_eval_envs, device, optimize_hyperparameters, conf_file, n_trials, n_timesteps, n_jobs):
  results_dir = "results/"
  csv_file = f"{results_dir}results.csv"
  os.makedirs(results_dir, exist_ok=True)
  reward_threshold = load_reward_threshold(conf_file, env)
  print(f"Reward threshold for {env}: {reward_threshold}")
  for run in range(10):
    print(f"Training {algo}, Run {run + 1}")
    start_time = time.time()
    log_folder = ".eval"
    exp_manager = configure(
        algo=algo,
        env=env,
        n_eval_envs=n_eval_envs,
        device=device,
        optimize_hyperparameters=optimize_hyperparameters,
        conf_file=conf_file,
        n_trials=n_trials,
        n_timesteps=n_timesteps,
        n_jobs=n_jobs,
        log_folder=log_folder,
        verbose=0,
        train_eval=True,
    )
    wall_time = time.time() - start_time
    save_path = exp_manager.save_path
    if not os.path.exists(save_path):
      print(
          f"Warning: Log path not found for {algo} run {run}. Skipping.")
      continue
    rewards = exp_manager.training_rewards
    timesteps = list(range(1, len(rewards) + 1))
    if len(rewards) == 0:
      print(
          f"Warning: No timesteps or rewards logged for {algo} on {env}, run {run + 1}. Skipping.")
      continue
    run_data_file = os.path.join(save_path, f"run_{run + 1}_data.csv")
    run_data = pd.DataFrame({"Timesteps": timesteps, "Reward": rewards})
    run_data.to_csv(run_data_file, index=False)
    print(f"Run data saved to {run_data_file}")
    mean_reward = mean(rewards)
    std_reward = stdev(rewards)
    sample_complexity = inf
    if reward_threshold is not None:
      cumulative_rewards = 0
      for i, reward in enumerate(rewards, start=1):
        cumulative_rewards += reward
        cumulative_mean = cumulative_rewards / i
        if cumulative_mean >= reward_threshold:
          sample_complexity = i
          break
    run_metrics = {
        "Model": algo,
        "Env": env,
        "Run": run + 1,
        "Wall Time (s)": wall_time,
        "Sample Complexity": sample_complexity,
        "Mean Reward": mean_reward,
        "Std Dev": std_reward,
        "File": run_data_file,
        "Save Path": save_path,
    }
    run_df = pd.DataFrame([run_metrics])
    if os.path.exists(csv_file):
      run_df.to_csv(csv_file, mode="a", header=False, index=False)
    else:
      run_df.to_csv(csv_file, index=False)
    print(f"Summary metrics for run {run + 1} saved to {csv_file}")
  plot_all_from_csv(csv_file, results_dir)


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

  # Set default configuration file path if not provided
  if params.conf_file is None:
    default_conf_path = f"Hyperparameters/{params.model.lower()}.yml"
    if os.path.exists(default_conf_path):
      params.conf_file = default_conf_path
    else:
      raise FileNotFoundError(
          f"Configuration file not found for model {params.model}. Provide a valid --conf_file.")

  evaluate_training(
      algo=params.model,
      env=params.env,
      n_eval_envs=params.envs,
      device=params.device,
      optimize_hyperparameters=params.optimize,
      conf_file=params.conf_file,
      n_trials=params.trials,
      n_timesteps=params.n_timesteps,
      n_jobs=params.n_jobs,
  )
