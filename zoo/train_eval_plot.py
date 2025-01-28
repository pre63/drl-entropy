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
def downsample_data_with_smoothing(timesteps, rewards, max_points=1000, window_size=500):

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


def plot_all_from_csv(csv_path, results_dir, filter_envs=None, filter_models=None):
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
  marker_interval = max(1, max_points // 25)

  # Predefine style and marker mappings for consistent ordering
  all_models = sorted(results_df["Model"].unique())
  style_mapping = {model: line_styles[i % len(line_styles)] for i, model in enumerate(all_models)}
  marker_mapping = {model: markers[i % len(markers)] for i, model in enumerate(all_models)}

  n_cols = (n_envs + 1) // 2
  fig, axes = plt.subplots(2, n_cols, figsize=(12 * n_cols, 10), dpi=150)
  axes = axes.flatten()

  for ax, (env, group) in zip(axes, grouped):
    for model in sorted(group["Model"].unique()):  # Sort models for consistent order
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

        downsampled_timesteps, downsampled_mean_rewards = downsample_data_with_smoothing(mean_rewards.index.values, mean_rewards.values)
        _, downsampled_std_rewards = downsample_data_with_smoothing(mean_rewards.index.values, std_rewards.values)

        line_style = style_mapping[model]
        marker = marker_mapping[model]

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
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Rewards")
    ax.legend(loc="upper left")

  for ax in axes[len(grouped) :]:
    ax.axis("off")

  plt.tight_layout()
  plot_path = os.path.join(results_dir, "results.png")
  plt.savefig(plot_path, dpi=150)
  plt.close()
  print(plot_path)


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


def resample_to_fixed_points(run_data, num_points, common_timesteps):
  normalized_timesteps = (run_data.index - run_data.index.min()) / (run_data.index.max() - run_data.index.min())
  interpolated_rewards = np.interp(common_timesteps, normalized_timesteps, run_data["Reward"])
  return pd.DataFrame({"Timesteps": common_timesteps, "Reward": interpolated_rewards})


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

  plot_rewards_sd_and_reject_outliers(grouped, results_dir)
  plot_sample_efficiency(grouped, results_dir)
  plot_learning_stability(grouped, results_dir)
  plot_learning_stability_cv(grouped, results_dir)
  plot_sample_efficiency_combined(grouped, results_dir)


def plot_learning_stability_cv(grouped, results_dir, num_points=10000):
  """
    Plots learning stability using the Coefficient of Variation (CV) to show relative variability.

    Args:
        grouped (dict): Dictionary with environments as keys and lists of file paths as values.
        results_dir (str): Directory to save the resulting plots.
    """
  os.makedirs(results_dir, exist_ok=True)

  n_envs = len(grouped)
  n_cols = (n_envs + 1) // 2
  fig, axes = plt.subplots(2, n_cols, figsize=(12 * n_cols, 10), dpi=150)
  axes = axes.flatten()

  # Define consistent styles and markers
  all_models = sorted(set(model for paths in grouped.values() for path in paths for model in [os.path.basename(os.path.dirname(os.path.dirname(path)))]))
  line_styles = ["-", "--", "-.", ":"]
  markers = ["o", "s", "^", "D"]
  style_mapping = {model: line_styles[i % len(line_styles)] for i, model in enumerate(all_models)}
  marker_mapping = {model: markers[i % len(markers)] for i, model in enumerate(all_models)}

  for ax, (env, paths) in zip(axes, sorted(grouped.items())):
    model_aggregated = {}
    common_timesteps = np.linspace(0, 1, num=num_points)  # Normalize timesteps

    for file_path in sorted(paths):
      if os.path.exists(file_path):
        model = os.path.basename(os.path.dirname(os.path.dirname(file_path)))  # Extract model name
        run_data = pd.read_csv(file_path)
        run_data.set_index("Timesteps", inplace=True)

        # Resample rewards for consistent comparison
        resampled_data = resample_to_fixed_points(run_data, num_points, common_timesteps)

        downsampled_timesteps, downsampled_rewards = downsample_data_with_smoothing(
          resampled_data.index, resampled_data["Reward"], max_points=100, window_size=1000
        )

        if model not in model_aggregated:
          model_aggregated[model] = []
        model_aggregated[model].append(pd.Series(downsampled_rewards, index=downsampled_timesteps))

    for model in sorted(model_aggregated.keys()):
      rewards_list = model_aggregated[model]
      if rewards_list:
        stacked_rewards = pd.concat(rewards_list, axis=1)
        mean_rewards = stacked_rewards.mean(axis=1)
        std_rewards = stacked_rewards.std(axis=1)

        # Calculate Coefficient of Variation (CV)
        cv = std_rewards / mean_rewards.replace(0, np.nan)  # Avoid division by zero

        # Plot CV
        ax.plot(mean_rewards.index, cv, label=f"{model} CV", linewidth=1.5, linestyle=style_mapping[model], marker=marker_mapping[model], markevery=5)

    ax.set_title(f"Environment: {env}")
    ax.set_xlabel("Normalized Episodes")
    ax.set_ylabel("Coefficient of Variation (CV)")
    ax.legend(loc="upper left")

  for ax in axes[len(grouped) :]:
    ax.axis("off")

  plt.tight_layout()
  plot_path = os.path.join(results_dir, "learning_stability_cv.png")
  plt.savefig(plot_path, dpi=150)
  plt.close()
  print(plot_path)


def plot_learning_stability(grouped, results_dir, num_points=10000):
  """
    Plots learning stability by showing mean rewards with standard deviation as a shaded region.

    Args:
        grouped (dict): Dictionary with environments as keys and lists of file paths as values.
        results_dir (str): Directory to save the resulting plots.
    """
  os.makedirs(results_dir, exist_ok=True)

  n_envs = len(grouped)
  n_cols = (n_envs + 1) // 2
  fig, axes = plt.subplots(2, n_cols, figsize=(12 * n_cols, 10), dpi=150)
  axes = axes.flatten()

  # Define consistent styles and markers
  all_models = sorted(set(model for paths in grouped.values() for path in paths for model in [os.path.basename(os.path.dirname(os.path.dirname(path)))]))
  line_styles = ["-", "--", "-.", ":"]
  markers = ["o", "s", "^", "D"]
  style_mapping = {model: line_styles[i % len(line_styles)] for i, model in enumerate(all_models)}
  marker_mapping = {model: markers[i % len(markers)] for i, model in enumerate(all_models)}

  for ax, (env, paths) in zip(axes, sorted(grouped.items())):
    model_aggregated = {}
    common_timesteps = np.linspace(0, 1, num=num_points)  # Normalize timesteps

    for file_path in sorted(paths):
      if os.path.exists(file_path):
        model = os.path.basename(os.path.dirname(os.path.dirname(file_path)))  # Extract model name
        run_data = pd.read_csv(file_path)
        run_data.set_index("Timesteps", inplace=True)

        # Resample rewards for consistent comparison
        resampled_data = resample_to_fixed_points(run_data, num_points, common_timesteps)

        downsampled_timesteps, downsampled_rewards = downsample_data_with_smoothing(
          resampled_data.index, resampled_data["Reward"], max_points=1000, window_size=500
        )

        if model not in model_aggregated:
          model_aggregated[model] = []
        model_aggregated[model].append(pd.Series(downsampled_rewards, index=downsampled_timesteps))

    for model in sorted(model_aggregated.keys()):
      rewards_list = model_aggregated[model]
      if rewards_list:
        stacked_rewards = pd.concat(rewards_list, axis=1)
        mean_rewards = stacked_rewards.mean(axis=1)
        std_rewards = stacked_rewards.std(axis=1)

        # Plot mean with standard deviation
        ax.fill_between(
          mean_rewards.index,
          mean_rewards - std_rewards,
          mean_rewards + std_rewards,
          alpha=0.2,
          label=f"{model} ± std",
        )
        ax.plot(
          mean_rewards.index,
          mean_rewards,
          label=model,
          linewidth=1.5,
          linestyle=style_mapping[model],
          marker=marker_mapping[model],
          markevery=50,
        )

    ax.set_title(f"Environment: {env}")
    ax.set_xlabel("Normalized Episodes")
    ax.set_ylabel("Reward")
    ax.legend(loc="upper left")

  for ax in axes[len(grouped) :]:
    ax.axis("off")

  plt.tight_layout()
  plot_path = os.path.join(results_dir, "learning_stability.png")
  plt.savefig(plot_path, dpi=150)
  plt.close()
  print(plot_path)


def plot_rewards_sd_and_reject_outliers(grouped, results_dir, num_points=10000, outlier_threshold=2.5):
  os.makedirs(results_dir, exist_ok=True)

  n_envs = len(grouped)
  n_cols = (n_envs + 1) // 2
  fig, axes = plt.subplots(2, n_cols, figsize=(12 * n_cols, 10), dpi=150)
  axes = axes.flatten()

  # Define styles and markers for consistent mapping
  all_models = sorted(set(model for paths in grouped.values() for path in paths for model in [os.path.basename(os.path.dirname(os.path.dirname(path)))]))
  line_styles = ["-", "--", "-.", ":"]
  markers = ["o", "s", "^", "D"]
  style_mapping = {model: line_styles[i % len(line_styles)] for i, model in enumerate(all_models)}
  marker_mapping = {model: markers[i % len(markers)] for i, model in enumerate(all_models)}
  max_points = 1000
  marker_interval = max(1, max_points // 25)

  for ax, (env, paths) in zip(axes, sorted(grouped.items())):
    model_aggregated = {}
    common_timesteps = np.linspace(0, 1, num=num_points)

    for file_path in sorted(paths):
      if os.path.exists(file_path):
        model = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
        run_data = pd.read_csv(file_path)
        run_data.set_index("Timesteps", inplace=True)

        resampled_data = resample_to_fixed_points(run_data, num_points, common_timesteps)

        downsampled_timesteps, downsampled_rewards = downsample_data_with_smoothing(
          resampled_data.index, resampled_data["Reward"], max_points=1000, window_size=500
        )

        if model not in model_aggregated:
          model_aggregated[model] = []
        model_aggregated[model].append(pd.Series(downsampled_rewards, index=downsampled_timesteps))

    for model in sorted(model_aggregated.keys()):
      rewards_list = model_aggregated[model]
      if rewards_list:
        stacked_rewards = pd.concat(rewards_list, axis=1)

        mean_rewards = stacked_rewards.mean(axis=1).to_numpy()
        std_rewards = stacked_rewards.std(axis=1).to_numpy()

        z_scores = (stacked_rewards.to_numpy() - mean_rewards[:, None]) / std_rewards[:, None]
        filtered_rewards = stacked_rewards.loc[:, (np.abs(z_scores) < outlier_threshold).all(axis=0)]

        mean_rewards = filtered_rewards.mean(axis=1)
        std_rewards = filtered_rewards.std(axis=1)

        line_style = style_mapping[model]
        marker = marker_mapping[model]

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
          markevery=marker_interval,
        )

    ax.set_title(env)
    ax.set_xlabel("Normalized Episodes")
    ax.set_ylabel("Rewards")
    ax.legend(loc="upper left")

  for ax in axes[len(grouped) :]:
    ax.axis("off")

  plt.tight_layout()
  plot_path = os.path.join(results_dir, "resampled_outlier.png")
  plt.savefig(plot_path, dpi=150)
  plt.close()
  print(plot_path)


def plot_sample_efficiency(grouped, results_dir, filter_envs=None, filter_models=None):
  """
    Plots sample efficiency for grouped paths based on episode counts within a fixed timestep budget.

    Args:
        grouped (dict): Dictionary with environments as keys and lists of file paths as values.
        results_dir (str): Directory to save the resulting plots.
    """
  os.makedirs(results_dir, exist_ok=True)

  n_envs = len(grouped)
  n_cols = (n_envs + 1) // 2
  fig, axes = plt.subplots(2, n_cols, figsize=(12 * n_cols, 10), dpi=150)
  axes = axes.flatten()

  # Define consistent styles and markers
  all_models = sorted(set(model for paths in grouped.values() for path in paths for model in [os.path.basename(os.path.dirname(os.path.dirname(path)))]))
  line_styles = ["-", "--", "-.", ":"]
  markers = ["o", "s", "^", "D"]
  style_mapping = {model: line_styles[i % len(line_styles)] for i, model in enumerate(all_models)}
  marker_mapping = {model: markers[i % len(markers)] for i, model in enumerate(all_models)}

  # Timestep budget
  timestep_budget = 1_000_000

  for ax, (env, paths) in zip(axes, sorted(grouped.items())):
    model_episode_counts = {}

    for file_path in sorted(paths):
      if os.path.exists(file_path):
        model = os.path.basename(os.path.dirname(os.path.dirname(file_path)))  # Extract model name
        run_data = pd.read_csv(file_path)

        # Filter data to only timesteps within the budget
        within_budget = run_data[run_data["Timesteps"] <= timestep_budget]

        # Count episodes within the budget
        num_episodes = len(within_budget)

        if model not in model_episode_counts:
          model_episode_counts[model] = []
        model_episode_counts[model].append(num_episodes)

    # Plot episode counts for each model
    for model in sorted(model_episode_counts.keys()):
      episode_counts = model_episode_counts[model]
      mean_episodes = np.mean(episode_counts)
      std_episodes = np.std(episode_counts)

      line_style = style_mapping[model]
      marker = marker_mapping[model]

      ax.bar(model, mean_episodes, yerr=std_episodes, capsize=5, label=model, alpha=0.75)

    ax.set_title(f"Environment: {env}")
    ax.set_ylabel("Episodes Completed")
    ax.set_xlabel("Models")
    ax.legend(loc="upper left")

  for ax in axes[len(grouped) :]:
    ax.axis("off")

  plt.tight_layout()
  plot_path = os.path.join(results_dir, "sample_efficiency.png")
  plt.savefig(plot_path, dpi=150)
  plt.close()
  print(plot_path)


def plot_sample_efficiency_combined(grouped, results_dir, filter_envs=None, filter_models=None):
  """
    Plots combined sample efficiency for all environments based on episode counts within a fixed timestep budget.

    Args:
        grouped (dict): Dictionary with environments as keys and lists of file paths as values.
        results_dir (str): Directory to save the resulting plot.
        filter_envs (list): List of environments to include (default: all).
        filter_models (list): List of models to include (default: all).
    """
  os.makedirs(results_dir, exist_ok=True)

  # Define consistent styles and markers
  all_models = sorted(set(model for paths in grouped.values() for path in paths for model in [os.path.basename(os.path.dirname(os.path.dirname(path)))]))
  line_styles = ["-", "--", "-.", ":"]
  markers = ["o", "s", "^", "D"]
  style_mapping = {model: line_styles[i % len(line_styles)] for i, model in enumerate(all_models)}
  marker_mapping = {model: markers[i % len(markers)] for i, model in enumerate(all_models)}

  # Timestep budget
  timestep_budget = 1_000_000
  combined_model_data = {model: [] for model in all_models}

  # Aggregate episode counts across environments
  for env, paths in grouped.items():
    if filter_envs and env not in filter_envs:
      continue

    model_episode_counts = {}

    for file_path in sorted(paths):
      if os.path.exists(file_path):
        model = os.path.basename(os.path.dirname(os.path.dirname(file_path)))  # Extract model name
        if filter_models and model not in filter_models:
          continue

        run_data = pd.read_csv(file_path)

        # Filter data to only timesteps within the budget
        within_budget = run_data[run_data["Timesteps"] <= timestep_budget]

        # Count episodes within the budget
        num_episodes = len(within_budget)

        if model not in model_episode_counts:
          model_episode_counts[model] = []
        model_episode_counts[model].append(num_episodes)

    for model, counts in model_episode_counts.items():
      combined_model_data[model].extend(counts)

  # Calculate mean and std for each model
  aggregated_means = {model: np.mean(counts) if counts else 0 for model, counts in combined_model_data.items()}
  aggregated_stds = {model: np.std(counts) if counts else 0 for model, counts in combined_model_data.items()}

  # Plot combined bar chart
  fig, ax = plt.subplots(figsize=(12, 8), dpi=150)
  models = sorted(aggregated_means.keys())
  means = [aggregated_means[model] for model in models]
  stds = [aggregated_stds[model] for model in models]

  ax.bar(models, means, yerr=stds, capsize=5, alpha=0.75)
  ax.set_title("Combined Sample Efficiency Across Environments")
  ax.set_ylabel("Average Episodes Completed")
  ax.set_xlabel("Models")
  ax.set_xticks(models)
  ax.set_xticklabels(models, rotation=45, ha="right")

  # Save the plot
  plt.tight_layout()
  plot_path = os.path.join(results_dir, "sample_efficiency_combined.png")
  plt.savefig(plot_path, dpi=150)
  plt.close()
  print(plot_path)


if __name__ == "__main__":
  filter_envs = None
  filter_models = None
  results_dir = ".plots"
  plot_from_path(".eval", results_dir, filter_envs=filter_envs, filter_models=filter_models)

  plot_all_from_csv("results/results.csv", results_dir, filter_envs=filter_envs, filter_models=filter_models)
