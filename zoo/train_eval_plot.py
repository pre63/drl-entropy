import os
import time
from math import inf
from statistics import mean, stdev

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


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

  # Calculate grid dimensions
  n_cols = (n_envs + 1) // 2  # Ensure all plots fit into a 2-row grid
  fig, axes = plt.subplots(2, n_cols, figsize=(12 * n_cols, 10), dpi=150)

  # Flatten axes for easier iteration, in case of a single row/column
  axes = axes.flatten()

  for ax, (env, group) in zip(axes, grouped):
    for model in group["Model"].unique():
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

        # Plot mean and standard deviation
        ax.fill_between(
          downsampled_timesteps,
          downsampled_mean_rewards - downsampled_std_rewards,
          downsampled_mean_rewards + downsampled_std_rewards,
          alpha=0.2,
        )
        ax.plot(downsampled_timesteps, downsampled_mean_rewards, label=f"{model}", linewidth=1.5)

    ax.set_title(env)
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Reward")
    ax.legend(loc="upper left")

  # Hide unused subplots
  for ax in axes[len(grouped) :]:
    ax.axis("off")

  plt.tight_layout()
  plot_path = os.path.join(results_dir, "combined_env_rewards.png")
  plt.savefig(plot_path, dpi=150)
  plt.close()
  print(f"Combined plot for all environments saved at: {plot_path}")


if __name__ == "__main__":
  plot_all_from_csv("results/results.csv", "results")
