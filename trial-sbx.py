import os
import argparse

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

import gymnasium as gym

from sbx import DDPG, CrossQ, PPO, SAC, TD3, TQC
from stable_baselines3.common.callbacks import EvalCallback

from Environments.LunarLander import make

from Callbacks.EvaluationMetrics import EvaluationMetrics

from Models.DDPG import optimal as optimize_ddpg_hyperparameters
from Models.CrossQ import optimal as optimize_crossq_hyperparameters
from Models.PPO import optimal as optimize_ppo_hyperparameters
from Models.SAC import optimal as optimize_sac_hyperparameters
from Models.TD3 import optimal as optimize_td3_hyperparameters
from Models.TQC import optimal as optimize_tqc_hyperparameters

from Reporting import add_to_experiments, add_to_trials


def correct_batch_size(n_steps, n_envs):
  valid_batch_sizes = [bs for bs in [64, 128, 256, 512] if (n_steps * n_envs) % bs == 0]
  if not valid_batch_sizes:
    print(f"Changing batch size from {n_steps * n_envs} to {n_envs}")
    return n_steps * n_envs  # Fallback to full batch size if no valid batch size found
  return max(valid_batch_sizes)  # Return the largest valid batch size for efficiency


def init_tensorboard(model_name):
  log_dir = "./.logs/"
  tensorboard_log_dir = os.path.join(log_dir, "tensorboard")
  os.makedirs(tensorboard_log_dir, exist_ok=True)
  return tensorboard_log_dir


def learn(trial, hyperparam_func, model_class, total_timesteps, env):
  hyperparams = hyperparam_func(trial, total_timesteps)
  if "batch_size" in hyperparams and "n_steps" in hyperparams:
    hyperparams["batch_size"] = correct_batch_size(hyperparams["n_steps"], 1)

  hyperparams_string = ", ".join([f"{key}={value}" for key, value in hyperparams.items()])
  print(f"Training with hyperparameters: {hyperparams_string}")

  total_timesteps = hyperparams.pop("total_timesteps")
  tensorboard_log_dir = init_tensorboard(model_name)

  n_eval_episodes = max(1000, total_timesteps // 1000)
  eval_callback = EvaluationMetrics(env, eval_freq=5000, n_eval_episodes=n_eval_episodes, verbose=0)

  model = model_class(**hyperparams, env=env, verbose=0, tensorboard_log=tensorboard_log_dir)

  model.learn(total_timesteps=total_timesteps, callback=[eval_callback])

  return model, eval_callback


hyperparameter_funcs = {"DDPG": optimize_ddpg_hyperparameters, "CROSSQ": optimize_crossq_hyperparameters, "PPO": optimize_ppo_hyperparameters, "SAC": optimize_sac_hyperparameters, "TD3": optimize_td3_hyperparameters, "TQC": optimize_tqc_hyperparameters}

model_classes = {"DDPG": DDPG, "CROSSQ": CrossQ, "PPO": PPO, "SAC": SAC, "TD3": TD3, "TQC": TQC}

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Optuna hyperparameter optimization with SBX")
  parser.add_argument("--model", type=str, required=True, choices=["ppo", "dqn", "sac", "td3", "tqc", "ddpg", "crossq"])
  parser.add_argument("--trials", type=int, default=20)
  parser.add_argument("--timesteps", type=int, default=10000)
  parser.add_argument("--envs", type=int, default=4)
  args = parser.parse_args()

  timesteps = args.timesteps
  model_name = args.model.upper()
  trials = args.trials
  envs = args.envs

  env = make()

  storage = JournalStorage(JournalFileBackend(".optuna-sbx/storage"))

  study = optuna.create_study(study_name=f"{model_name}_study", direction="maximize", storage=storage, load_if_exists=True)

  def optimize(trial):
    model, eval_callback = learn(trial, hyperparameter_funcs[model_name], model_classes[model_name], timesteps, env)
    objective = eval_callback.eval_total_rewards

    def to_python_native(value):
      """Convert NumPy types to native Python types."""
      if hasattr(value, "item"):  # Handles scalar NumPy values
        return value.item()
      elif isinstance(value, (float, int)):  # Already native types
        return value
      elif isinstance(value, (list, tuple)):  # Convert lists with NumPy types inside
        return [to_python_native(v) for v in value]
      return float(value) if "float" in str(type(value)) else int(value)

    # Enforcing native types in the metrics dictionary
    metrics = {
        "model_name": model_name,
        "eval/success_count": to_python_native(eval_callback.eval_success_count),
        "eval/success_rate": to_python_native(eval_callback.eval_success_rate),
        "eval/mean_reward": to_python_native(eval_callback.eval_mean_reward),
        "eval/mean_ep_length": to_python_native(eval_callback.eval_mean_ep_length),
        "eval/mean_success_rate": to_python_native(eval_callback.eval_mean_success_rate),
        "eval/total_rewards": to_python_native(eval_callback.eval_total_rewards),

        "total_timesteps": to_python_native(timesteps),
        "objective": to_python_native(objective),
    }

    trial.set_user_attr("metrics", metrics)

    return objective

  study.optimize(optimize, n_trials=trials)

  best_trial = study.best_trial
  print(f"Best trial number: {best_trial.number}, Best reward: {best_trial.value}, Best hyperparameters: {best_trial.params}")

  # Save Optuna results to a CSV file
  add_to_trials(study.trials_dataframe())

  best_metrics = best_trial.user_attrs["metrics"] if "metrics" in best_trial.user_attrs else {}
  add_to_experiments({"model_name": model_name, **best_trial.params, **best_metrics})
