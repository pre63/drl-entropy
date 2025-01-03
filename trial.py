import os
import sys

import argparse

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

from Reporting import add_to_experiments, add_to_trials


def save(model, model_name):
  date_time = datetime.now().strftime("%Y%m%d-%H%M%S")
  model.save(os.path.join(f"results/{model_name}{date_time}", f"{model_name}_lunarlander_final"))


def get_num_envs(default=4):
  return int(sys.argv[3]) if len(sys.argv) > 3 else default


def get_model(model_name="PPO"):

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


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Optuna hyperparameter optimization with SB3 models.")
  parser.add_argument("--model", type=str, required=True, choices=["ppo", "trpo", "entrpo", "entrpor"])
  parser.add_argument("--trials", type=int, default=20)
  parser.add_argument("--timesteps", type=int, default=10000)
  parser.add_argument("--envs", type=int, default=4)
  args = parser.parse_args()

  num_envs = args.envs
  n_trials = args.trials
  total_timesteps = args.timesteps
  model_name = args.model.upper()

  model_class, model_name, optimal_function = get_model(model_name)
  print(f"\nRunning {n_trials} trials for {num_envs} environments with {model_name}\n")

  storage = optuna.storages.JournalStorage(
      optuna.storages.journal.JournalFileBackend(".optuna/storage"),
  )

  # Set up Optuna study
  study_name = f"{model_name}_study"
  study = optuna.create_study(study_name=study_name, direction="maximize", pruner=MedianPruner(), storage=storage, load_if_exists=True)

  def objective(trial):
    return optimize_model(trial, model_class, optimal_function, model_name, num_envs=num_envs, total_timesteps=total_timesteps)

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
