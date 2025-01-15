import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rl_zoo3
import rl_zoo3.train
from sbx import SAC, TQC
from stable_baselines3.common.results_plotter import load_results, ts2xy

# Register Environments
import Environments
from Models.EnTRPO.EnTRPO import EnTRPO, EnTRPOHigh, EnTRPOLow, sample_entrpo_params
from Models.EnTRPOR.EnTRPOR import EnTRPOR, sample_entrpor_params
from Models.TRPOQ.TRPOQ import TRPOQ, sample_trpoq_params
from Models.TRPOQ.TRPOQ2 import TRPOQ2, sample_trpoq2_params
from Models.TRPOQ.TRPOQH import TRPOQH, TRPOQHO, sample_trpoqh_params, sample_trpoqho_params
from Models.TRPOR.TRPOR import TRPOR, sample_trpor_params
from Models.TRPO import TRPO
from Models.PPO import PPO
from zoo.configure import configure

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

import yaml


def load_reward_threshold(conf_file, env):
  """
  Load the reward threshold for the given environment from the YAML configuration file.

  Args:
      conf_file (str): Path to the YAML configuration file.
      env (str): Environment name.

  Returns:
      float: Reward threshold for the specified environment.
  """
  with open(conf_file, "r") as file:
    config = yaml.safe_load(file)
  if env in config and "reward_threshold" in config[env]:
    return config[env]["reward_threshold"]
  raise ValueError(f"Reward threshold not found for environment: {env}")


def plot_all_from_csv(csv_path, results_dir):
  """
  Plot all environments and models from a given CSV file.

  Args:
      csv_path (str): Path to the CSV containing the results.
      results_dir (str): Directory to save the resulting plot.
  """
  os.makedirs(results_dir, exist_ok=True)

  # Read metrics from CSV
  if not os.path.exists(csv_path):
    print(f"CSV file not found: {csv_path}")
    return

  results_df = pd.read_csv(csv_path)
  grouped = results_df.groupby("Env")

  # Prepare tiled subplots
  n_envs = len(grouped)
  cols = 2
  rows = (n_envs + 1) // cols
  fig, axes = plt.subplots(rows, cols, figsize=(10, 5 * rows))
  axes = axes.flatten()

  for i, (env, group) in enumerate(grouped):
    for model in group["Model"].unique():
      model_data = group[group["Model"] == model].iloc[0]
      timesteps = np.array(eval(model_data["Timesteps"]))
      rewards = np.array(eval(model_data["Rewards"]))
      axes[i].plot(timesteps, rewards, label=f"{model}")
      axes[i].fill_between(
          timesteps,
          rewards - model_data["Std Dev"],
          rewards + model_data["Std Dev"],
          alpha=0.3
      )

    axes[i].set_title(f"{env}")
    axes[i].set_xlabel("Timesteps")
    axes[i].set_ylabel("Reward")
    axes[i].legend()

  # Remove unused subplots
  for i in range(len(grouped), len(axes)):
    fig.delaxes(axes[i])

  plt.tight_layout()
  plot_path = os.path.join(results_dir, "all_envs_rewards.png")
  plt.savefig(plot_path)
  plt.close()
  print(f"Tiled plot for all environments saved at: {plot_path}")


def evaluate_training(algo, env, n_eval_envs, device, optimize_hyperparameters, conf_file, n_trials, n_timesteps, n_jobs):
  csv_file = f"results/model_comparison_{env}.csv"
  results_dir = f"results/{algo}/all_envs"

  # Load reward threshold from the YAML configuration
  reward_threshold = load_reward_threshold(conf_file, env)
  print(f"Reward threshold for {env}: {reward_threshold}")

  # Check if the CSV exists to load existing results
  if os.path.exists(csv_file):
    existing_data = pd.read_csv(csv_file)
  else:
    existing_data = pd.DataFrame()

  for run in range(10):
    # Skip runs that are already completed
    if not existing_data.empty and ((existing_data["Model"] == algo) & (existing_data["Env"] == env) & (existing_data["Run"] == run + 1)).any():
      print(f"Skipping already completed run {run + 1} for {algo} on {env}")
      continue

    print(f"Training {algo}, Run {run + 1}")
    start_time = time.time()

    log_folder = ".eval"

    # Train with RL Zoo's configure
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
        eval_freq=n_timesteps // 1000,
        verbose=0
    )

    wall_time = time.time() - start_time
    save_path = exp_manager.save_path
    if not os.path.exists(save_path):
      print(f"Warning: Log path not found for {algo} run {run}. Skipping.")
      continue

    # Load evaluation results
    results = load_results(save_path)
    x, y = ts2xy(results, 'timesteps')
    print(f"Loaded {len(x)} timesteps for {algo} on {env}")

    # Collect metrics
    idx = np.argmax(y >= reward_threshold) if np.any(y >= reward_threshold) else np.inf
    sample_complexity = x[idx] if idx != np.inf else np.inf

    run_metrics = {
        "Model": algo,
        "Env": env,
        "Run": run + 1,
        "Wall Time (s)": wall_time,
        "Sample Complexity": sample_complexity,
        "Mean Reward": np.mean(y),
        "Std Dev": np.std(y),
        "Timesteps": [x.tolist()],  # Save timesteps as a list
        "Rewards": [y.tolist()]    # Save rewards as a list
    }

    # Append the new run's results to the CSV
    run_df = pd.DataFrame([run_metrics])
    if os.path.exists(csv_file):
      run_df.to_csv(csv_file, mode='a', header=False, index=False)
    else:
      run_df.to_csv(csv_file, index=False)

    print(f"Metrics for run {run + 1} updated in: {csv_file}")

    # Update the plot after each run
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
  parser.add_argument("--train-eval", type=bool, default=False)

  params = parser.parse_args()
  model = params.model
  conf_file = params.conf_file

  if conf_file is None and os.path.exists(f"Hyperparameters/{model.lower()}.yml"):
    conf_file = f"Hyperparameters/{model.lower()}.yml"

  if params.train_eval:
    evaluate_training(
        algo=params.model,
        env=params.env,
        n_eval_envs=params.envs,
        device=params.device,
        optimize_hyperparameters=params.optimize,
        conf_file=conf_file,
        n_trials=params.trials,
        n_timesteps=params.n_timesteps,
        n_jobs=params.n_jobs)
  else:
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
