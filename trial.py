import os
import sys

import csv
from datetime import datetime

import optuna
from optuna.pruners import MedianPruner

from sb3_contrib import TRPO
from stable_baselines3 import PPO

from Models.TRPO import optimal as optimal_trpo
from Models.PPO import optimal as optimal_ppo
from Models.EnTRPO import EnTRPO, optimal as optimal_entrpo
from Models.EnTRPOR import EnTRPOR, optimal as optimal_entrpor


from Evaluation import optimize_model


def save(model, model_name):
  date_time = datetime.now().strftime("%Y%m%d-%H%M%S")
  model.save(os.path.join(f"results/{model_name}{date_time}", f"{model_name}_lunarlander_final"))


def get_num_envs(default=4):
  return int(sys.argv[3]) if len(sys.argv) > 3 else default


def get_model(default="PPO"):
  model_name = sys.argv[1].upper() if len(sys.argv) > 1 else default

  match model_name:
    case "TRPO":
      return TRPO, model_name, optimal_trpo
    case "ENTRPO":
      return EnTRPO, model_name, optimal_entrpo
    case "ENTRPOR":
      return EnTRPOR, model_name, optimal_entrpor
    case "PPO":
      return PPO, model_name, optimal_ppo
    case _:
      raise ValueError(f"Invalid model name: {model_name}")


def get_num_trials(default):
  return int(sys.argv[2]) if len(sys.argv) > 2 else default


def add_to_experiments(metrics):
    filename = "Experiments.csv"
    file_exists = os.path.isfile(filename)

    # Write trial outcome to the CSV
    with open(filename, mode="a", newline="") as csv_file:
        writer = csv.writer(csv_file, quotechar='"', quoting=csv.QUOTE_ALL)

        if not file_exists:
            headers = list(metrics.keys())
            writer.writerow(headers)

        row = list(metrics.values())
        writer.writerow(row)


def add_to_trials(trials):
  results_file = "Trials.csv"
  if os.path.isfile(results_file):
    trials.to_csv(results_file, mode="a", header=False, index=False)
  else:
    trials.to_csv(results_file, mode="w", index=False)


if __name__ == "__main__":
  num_envs = get_num_envs(4)
  n_trials = get_num_trials(10)

  model_class, model_name, optimal_function = get_model()
  print(f"\nRunning {n_trials} trials for {num_envs} environments with {model_name}\n")

  # Set up Optuna study
  study = optuna.create_study(direction="maximize", pruner=MedianPruner())

  def objective(trial):
    return optimize_model(trial, model_class, optimal_function, model_name, num_envs=num_envs)

  study.optimize(objective, n_trials=n_trials)

  # Log the best trial
  best_trial = study.best_trial
  best_metrics = best_trial.user_attrs["metrics"]

  print("Best trial:")
  print(f"Parameters: {best_trial.params}")
  print(f"Metrics: {best_metrics}")

  # Save Optuna results to a CSV file
  add_to_trials(study.trials_dataframe())

  add_to_experiments({"model_name": model_name, **best_trial.params, **best_metrics})
