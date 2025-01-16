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
from stable_baselines3.common.results_plotter import load_results, ts2xy

# Register Environments
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


def plot_all_from_csv(csv_path, results_dir):

  os.makedirs(results_dir, exist_ok=True)

  # Check if the aggregated CSV exists
  if not os.path.exists(csv_path):
    print(f"CSV file not found: {csv_path}")
    return

  # Read the aggregated results CSV
  results_df = pd.read_csv(csv_path)

  # Group by environments
  grouped = results_df.groupby("Env")

  # Create a single plot with one graph per environment
  n_envs = len(grouped)
  fig, axes = plt.subplots(n_envs, 1, figsize=(12, 5 * n_envs), dpi=150)  # High DPI for better resolution
  if n_envs == 1:
    axes = [axes]  # Ensure axes is iterable for a single subplot

  for ax, (env, group) in zip(axes, grouped):
    for model in group["Model"].unique():
      # Filter by model
      model_data = group[group["Model"] == model]

      # Aggregate all runs for the same model
      aggregated_df = pd.DataFrame()  # DataFrame to hold all runs' data
      for _, row in model_data.iterrows():
        run_file = row["File"]  # Path to the run-specific data CSV
        if os.path.exists(run_file):
          run_data = pd.read_csv(run_file)
          # Set Timesteps as index for proper alignment
          run_data.set_index("Timesteps", inplace=True)
          # Add the run's Reward data to the aggregation
          aggregated_df = pd.concat([aggregated_df, run_data["Reward"]], axis=1)
        else:
          print(f"Run file not found: {run_file}")

      if not aggregated_df.empty:
        aggregated_df.columns = range(aggregated_df.shape[1])  # Rename columns for consistency
        mean_rewards = aggregated_df.mean(axis=1)  # Mean across runs
        std_rewards = aggregated_df.std(axis=1)  # Standard deviation across runs

        # Plot standard deviation as a light shaded region
        ax.fill_between(
            mean_rewards.index,
            mean_rewards - std_rewards,
            mean_rewards + std_rewards,
            alpha=0.2,  # Light transparency for the shaded area
            color="gray"  # Standard color for all shaded regions
        )

        # Plot mean as a clear solid line (after the shaded region)
        ax.plot(mean_rewards.index, mean_rewards, label=f"{model} (Mean)", linewidth=1.5)

    ax.set_title(env)
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Reward")
    ax.legend(loc='upper left')

  plt.tight_layout()
  plot_path = os.path.join(results_dir, "combined_env_rewards.png")
  plt.savefig(plot_path)
  plt.close()
  print(f"Combined plot for all environments saved at: {plot_path}")


def evaluate_training(algo, env, n_eval_envs, device, optimize_hyperparameters, conf_file, n_trials, n_timesteps, n_jobs):
  results_dir = "results/"
  csv_file = f"{results_dir}results.csv"

  os.makedirs(results_dir, exist_ok=True)

  # Load reward threshold from the YAML configuration
  reward_threshold = load_reward_threshold(conf_file, env)
  print(f"Reward threshold for {env}: {reward_threshold}")

  for run in range(10):
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
        verbose=0,
        train_eval=True,
    )

    wall_time = time.time() - start_time
    save_path = exp_manager.save_path

    if not os.path.exists(save_path):
      print(f"Warning: Log path not found for {algo} run {run}. Skipping.")
      continue

    # Extract rewards and generate corresponding timesteps
    rewards = exp_manager.training_rewards
    timesteps = list(range(1, len(rewards) + 1))  # Ensure timesteps is a valid sequence

    # Ensure data integrity
    if len(rewards) == 0:
      print(f"Warning: No timesteps or rewards logged for {algo} on {env}, run {run + 1}. Skipping.")
      continue

    # Save raw run data to a separate file
    run_data_file = os.path.join(save_path, f"run_{run + 1}_data.csv")
    run_data = pd.DataFrame({"Timesteps": timesteps, "Reward": rewards})  # Correct column names
    run_data.to_csv(run_data_file, index=False)
    print(f"Run data saved to {run_data_file}")

    # Calculate statistics
    mean_reward = mean(rewards)
    std_reward = stdev(rewards)

    # Calculate sample complexity
    sample_complexity = inf
    if reward_threshold is not None:
      cumulative_rewards = 0
      for i, reward in enumerate(rewards, start=1):
        cumulative_rewards += reward
        cumulative_mean = cumulative_rewards / i
        if cumulative_mean >= reward_threshold:
          sample_complexity = i
          break  # Stop once the threshold is reached

    # Save summary data to the shared CSV
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
      run_df.to_csv(csv_file, mode='a', header=False, index=False)
    else:
      run_df.to_csv(csv_file, index=False)
    print(f"Summary metrics for run {run + 1} saved to {csv_file}")

    # Update the plot with the current data
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
