import os
import time
from collections import defaultdict
from math import inf
from statistics import mean, stdev

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


# Good
def downsample_data_with_smoothing(timesteps, rewards, max_points=1000, window_size=50):

  if len(timesteps) <= max_points:
    # Apply rolling average smoothing only
    smoothed_rewards = pd.Series(rewards).rolling(window=window_size, min_periods=1).mean()
    return timesteps, smoothed_rewards.values

  # Convert to numpy arrays for reshaping
  timesteps = np.array(timesteps)
  rewards = np.array(rewards)

  # Apply rolling average smoothing
  smoothed_rewards = pd.Series(rewards).rolling(window=window_size, min_periods=1).mean()

  # Downsample the smoothed data
  bin_size = int(np.ceil(len(timesteps) / max_points))
  binned_timesteps = np.mean(timesteps[: len(timesteps) // bin_size * bin_size].reshape(-1, bin_size), axis=1)
  binned_rewards = np.mean(smoothed_rewards[: len(smoothed_rewards) // bin_size * bin_size].values.reshape(-1, bin_size), axis=1)

  # Handle any remaining data points
  remainder_timesteps = timesteps[len(timesteps) // bin_size * bin_size :]
  remainder_rewards = smoothed_rewards[len(smoothed_rewards) // bin_size * bin_size :].values
  if len(remainder_timesteps) > 0:
    binned_timesteps = np.append(binned_timesteps, np.mean(remainder_timesteps))
    binned_rewards = np.append(binned_rewards, np.mean(remainder_rewards))

  return binned_timesteps, binned_rewards


def plot_all_from_csv(csv_path, results_dir):
  """
    Generate plots for all environments in a grid with 2 rows.
    Args:
        csv_path (str): Path to the CSV containing the aggregated results.
        results_dir (str): Directory to save the resulting plot.
    """
  os.makedirs(results_dir, exist_ok=True)
  if not os.path.exists(csv_path):
    print(f"CSV file not found: {csv_path}")
    return

  results_df = pd.read_csv(csv_path)
  grouped = results_df.groupby("Env")
  n_envs = len(grouped)
  line_styles = ["-", "--", "-.", ":"]
  markers = ["o", "s", "^", "D"]
  max_points = 1000
  marker_interval = max(1, max_points // 50)  # Ensure no more than 100 markers per line

  # Calculate grid dimensions
  n_cols = (n_envs + 1) // 2  # Ensure all plots fit into a 2-row grid
  fig, axes = plt.subplots(2, n_cols, figsize=(12 * n_cols, 10), dpi=150)

  # Flatten axes for easier iteration, in case of a single row/column
  axes = axes.flatten()

  for ax, (env, group) in zip(axes, grouped):
    for idx, model in enumerate(group["Model"].unique()):
      model_data = group[group["Model"] == model]
      aggregated_df = pd.DataFrame()

      for _, row in model_data.iterrows():
        run_file = row["File"]
        if os.path.exists(run_file):
          run_data = pd.read_csv(run_file)
          run_data.set_index("Timesteps", inplace=True)
          aggregated_df = pd.concat([aggregated_df, run_data["Reward"]], axis=1)

      if not aggregated_df.empty:
        mean_rewards = aggregated_df.mean(axis=1)
        std_rewards = aggregated_df.std(axis=1)

        # Downsample data
        downsampled_timesteps, downsampled_mean_rewards = downsample_data_with_smoothing(mean_rewards.index.values, mean_rewards.values)
        _, downsampled_std_rewards = downsample_data_with_smoothing(mean_rewards.index.values, std_rewards.values)

        # Select line style and marker
        line_style = line_styles[idx % len(line_styles)]
        marker = markers[idx % len(markers)]

        # Plot mean and standard deviation
        ax.fill_between(
          downsampled_timesteps,
          downsampled_mean_rewards - downsampled_std_rewards,
          downsampled_mean_rewards + downsampled_std_rewards,
          alpha=0.2,
        )
        ax.plot(
          downsampled_timesteps,
          downsampled_mean_rewards,
          label=f"{model}",
          linewidth=1.5,
          linestyle=line_style,
          marker=marker,
          markevery=marker_interval,
        )

    ax.set_title(env)
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Reward")
    ax.legend(loc="upper left")

  # Hide unused subplots
  for ax in axes[len(grouped) :]:
    ax.axis("off")

  plt.tight_layout()
  plot_path = os.path.join(results_dir, "results.png")
  plt.savefig(plot_path, dpi=150)
  plt.close()
  print(f"Combined plot for all environments saved at: {plot_path}")


# Good
def get_data_csv_files(path):
  return [os.path.join(root, file) for root, _, files in os.walk(path) for file in files if file.endswith("data.csv")]


# Good
def group_paths_by_env(paths):
  grouped = defaultdict(list)
  for path in paths:
    # Extract environment name from the path
    # Assuming environment name is the part before the last underscore in the second-to-last folder
    parts = path.split("/")
    if len(parts) > 2:
      env_name = parts[-2].split("_")[0]
      grouped[env_name].append(path)
  return dict(grouped)


# Good
def plot_from_path(env_path, results_dir, filter_envs=None, filter_models=None):
  csv_files = get_data_csv_files(env_path)
  grouped = group_paths_by_env(csv_files)

  # filter_envs is a list of environment names to plot
  if filter_envs:
    grouped = {env: paths for env, paths in grouped.items() if env in filter_envs}

  # filter_models is a list of model names to plot
  if filter_models:
    grouped = {env: [path for path in paths if any(model in path for model in filter_models)] for env, paths in grouped.items()}

  plot_rewards_from_grouped_paths(grouped, results_dir)


def resample_to_fixed_points(run_data, num_points, common_timesteps):
  normalized_timesteps = (run_data.index - run_data.index.min()) / (run_data.index.max() - run_data.index.min())
  interpolated_rewards = np.interp(common_timesteps, normalized_timesteps, run_data["Reward"])
  return pd.DataFrame({"Timesteps": common_timesteps, "Reward": interpolated_rewards})


def plot_rewards_from_grouped_paths(grouped, results_dir, num_points=10000):
  os.makedirs(results_dir, exist_ok=True)

  n_envs = len(grouped)
  n_cols = (n_envs + 1) // 2
  fig, axes = plt.subplots(2, n_cols, figsize=(12 * n_cols, 10), dpi=150)
  axes = axes.flatten()

  # Define patterns and markers for accessibility
  line_styles = ["-", "--", "-.", ":"]
  markers = ["o", "s", "^", "D"]
  max_points = 1000
  marker_interval = max(1, max_points // 50)  # Ensure no more than 100 markers per line

  for ax, (env, paths) in zip(axes, grouped.items()):
    model_aggregated = {}
    common_timesteps = np.linspace(0, 1, num=num_points)

    for file_path in paths:
      if os.path.exists(file_path):
        model = os.path.basename(os.path.dirname(os.path.dirname(file_path)))  # Extract model name
        run_data = pd.read_csv(file_path)
        run_data.set_index("Timesteps", inplace=True)

        resampled_data = resample_to_fixed_points(run_data, num_points, common_timesteps)

        # Downsample the resampled data
        downsampled_timesteps, downsampled_rewards = downsample_data_with_smoothing(
          resampled_data.index, resampled_data["Reward"], max_points=1000, window_size=50
        )

        if model not in model_aggregated:
          model_aggregated[model] = []
        model_aggregated[model].append(pd.Series(downsampled_rewards, index=downsampled_timesteps))

    for idx, (model, rewards_list) in enumerate(model_aggregated.items()):
      if rewards_list:
        stacked_rewards = pd.concat(rewards_list, axis=1)
        mean_rewards = stacked_rewards.mean(axis=1)
        std_rewards = stacked_rewards.std(axis=1)

        # Use styles and markers to differentiate
        line_style = line_styles[idx % len(line_styles)]
        marker = markers[idx % len(markers)]

        ax.fill_between(
          mean_rewards.index,
          mean_rewards - std_rewards,
          mean_rewards + std_rewards,
          alpha=0.2,
        )
        ax.plot(
          mean_rewards.index,
          mean_rewards,
          label=model,
          linewidth=1.5,
          linestyle=line_style,
          marker=marker,
          markevery=marker_interval,  # Plot markers only at intervals
        )

    ax.set_title(env)
    ax.set_xlabel("Normalized Timesteps (0 to 1)")
    ax.set_ylabel("Reward")
    ax.legend(loc="upper left")

  for ax in axes[len(grouped) :]:
    ax.axis("off")

  plt.tight_layout()
  plot_path = os.path.join(results_dir, "resampled.png")
  plt.savefig(plot_path, dpi=150)
  plt.close()
  print(f"Combined plot for all environments saved at: {plot_path}")


if __name__ == "__main__":
  filter_envs = ["Ant-v5", "Humanoid-v5", "InvertedDoublePendulum-v5", "Pendulum-v1"]
  filter_models = ["trpor", "entrpo", "trpo"]
  results_dir = ".plots"
  plot_from_path(".eval", results_dir, filter_envs=filter_envs, filter_models=filter_models)

  plot_all_from_csv("results/results.csv", results_dir)
