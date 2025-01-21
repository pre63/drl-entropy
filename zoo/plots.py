import os
import pickle
import shutil

import matplotlib.pyplot as plt
import numpy as np
import optuna
from optuna.visualization import (
  plot_contour,
  plot_edf,
  plot_intermediate_values,
  plot_optimization_history,
  plot_parallel_coordinate,
  plot_param_importances,
  plot_rank,
  plot_slice,
  plot_timeline,
)


def load_and_copy_studies(folder_path, result_path):
  last_study = None
  for file in os.listdir(folder_path):
    if file.endswith(".pkl"):
      file_path = os.path.join(folder_path, file)
      dest_path = os.path.join(result_path, file)
      shutil.copy(file_path, dest_path)
      with open(file_path, "rb") as f:
        try:
          study = pickle.load(f)
          if isinstance(study, optuna.study.Study):
            last_study = study
        except Exception as e:
          print(f"Error loading {file_path}: {e}")
  return last_study


def save_optuna_plots(study, model_name, result_path):
  plots = {
    "optimization_history": plot_optimization_history(study),
    "parallel_coordinate": plot_parallel_coordinate(study),
    "param_importances": plot_param_importances(study),
    "edf": plot_edf(study),
    "intermediate_values": plot_intermediate_values(study),
    "contour": plot_contour(study),
    "slice": plot_slice(study),
    "rank": plot_rank(study),
    "timeline": plot_timeline(study),
  }

  for plot_name, plot_object in plots.items():
    plot_file = os.path.join(result_path, f"{model_name}_{plot_name}.html")
    plot_object.write_html(plot_file)


def calculate_sample_efficiency(best_trial):
  # Assuming sample efficiency is calculated as reward divided by total trials or a similar metric
  reward = best_trial.value
  num_trials = best_trial.number + 1  # Trials are zero-indexed
  sample_efficiency = reward / num_trials if num_trials > 0 else 0
  return sample_efficiency


def plot_custom_metrics_best_trial(study, result_path, model_name):
  best_trial = study.best_trial
  rewards_per_episode = best_trial.value
  sample_efficiency = calculate_sample_efficiency(best_trial)

  plt.figure()
  plt.bar(["Best Trial"], [rewards_per_episode])
  plt.title("Best Trial - Total Reward")
  plt.ylabel("Total Reward")
  plt.savefig(os.path.join(result_path, f"{model_name}_best_trial_reward.png"))
  plt.close()

  plt.figure()
  plt.bar(["Best Trial"], [sample_efficiency])
  plt.title("Best Trial - Sample Efficiency")
  plt.ylabel("Sample Efficiency")
  plt.savefig(os.path.join(result_path, f"{model_name}_best_trial_sample_efficiency.png"))
  plt.close()


def process_logs(logs_folder=".logs"):
  if not os.path.exists(logs_folder):
    print("Logs folder not found.")
    return

  for model_name in os.listdir(logs_folder):
    model_folder_path = os.path.join(logs_folder, model_name)
    if not os.path.isdir(model_folder_path):
      continue

    result_path = f"./results/{model_name}"
    os.makedirs(result_path, exist_ok=True)

    study = load_and_copy_studies(model_folder_path, result_path)
    if not study:
      print(f"No valid Optuna studies found for model: {model_name}")
      continue

    save_optuna_plots(study, f"{model_name}", result_path)
    plot_custom_metrics_best_trial(study, result_path, f"{model_name}")


process_logs()
