import os
import sys
from datetime import datetime
import optuna
from optuna.pruners import MedianPruner
import matplotlib.pyplot as plt
import gymnasium as gym
from Models.EnTRPO3 import EnTRPO
from Environments.LunarLander import make
from sb3_contrib import TRPO
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback


class SuccessEvalCallback(EvalCallback):
  def __init__(self, eval_env, **kwargs):
    super().__init__(eval_env, **kwargs)
    self.success_count = 0
    self.total_episodes = 0
    self.success_rate = 0.0

  def _evaluate_policy(self):
    # Perform evaluation using the evaluation environment
    episode_rewards, episode_lengths = [], []
    success_count = 0

    for _ in range(self.n_eval_episodes):
      obs = self.eval_env.reset()
      done = False
      episode_reward = 0.0

      while not done:
        action, _ = self.model.predict(obs, deterministic=self.deterministic)
        obs, reward, done, info = self.eval_env.step(action)
        episode_reward += reward

        # Check for 'success' in info
        if "success" in info[0] and info[0]["success"]:
          success_count += 1

      episode_rewards.append(episode_reward)
      episode_lengths.append(info[0].get("episode", {}).get("l", 0))

    # Update success metrics
    self.success_count += success_count
    self.total_episodes += self.n_eval_episodes
    if self.total_episodes > 0:
      self.success_rate = self.success_count / self.total_episodes

    # Log success metrics
    if self.logger:
      self.logger.record("eval/success_total", self.success_count)
      self.logger.record("eval/success_rate", self.success_rate)

    return episode_rewards, episode_lengths


class SuccessMetricsCallback(BaseCallback):
  """
  Custom callback to track success and success rate during training.
  """

  def __init__(self, verbose=0):
    super().__init__(verbose)
    self.success_count = 0
    self.total_episodes = 0
    self.success_rate = 0

  def _on_step(self):
    # Check for success in the current episode
    infos = self.locals.get("infos", [])
    for info in infos:
      if info.get("success", False):
        self.success_count += 1
    return True

  def _on_rollout_end(self):
    # Update the total episodes count and success rate
    self.total_episodes += len(self.locals.get("infos", []))
    if self.total_episodes > 0:
      self.success_rate = self.success_count / self.total_episodes

    # Log success metrics to TensorBoard
    if self.logger:
      self.logger.record("success/total_success", self.success_count)
      self.logger.record("success/success_rate", self.success_rate)


def make_env(rank, seed=0):
  def _init():
    env = make()
    env.seed(seed + rank)
    env = Monitor(env, f"./.logs/env_{rank}")
    return env
  return _init


def make_eval_env(seed=0):
  """
  Create a SubprocVecEnv for evaluation to match the training environment type.
  """
  def _init():
    env = make()
    env.seed(seed)
    return Monitor(env, f"./.logs/eval_env")
  return SubprocVecEnv([_init])


def init_tensorboard(model_name):
  log_dir = "./.logs/"
  tensorboard_log_dir = os.path.join(log_dir, "tensorboard")
  os.makedirs(tensorboard_log_dir, exist_ok=True)
  return tensorboard_log_dir


def init_eval(env, model_name):
  log_dir = f"./.logs/{model_name}"
  os.makedirs(log_dir, exist_ok=True)

  # Set up callbacks for evaluation and saving models
  checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=log_dir, name_prefix=model_name)
  eval_callback = EvalCallback(
      env,
      best_model_save_path=log_dir,
      log_path=log_dir,
      eval_freq=5000,
      n_eval_episodes=5,
      deterministic=True,
  )
  return checkpoint_callback, eval_callback


def save(model, model_name):
  date_time = datetime.now().strftime("%Y%m%d-%H%M%S")
  model.save(os.path.join(f"results/{model_name}{date_time}", f"{model_name}_lunarlander_final"))


class TrialPruningCallback(BaseCallback):
  """
  A custom callback that integrates Optuna trial pruning.
  """

  def __init__(self, trial, eval_callback, verbose=0):
    super().__init__(verbose)
    self.trial = trial
    self.eval_callback = eval_callback  # Reference to the EvalCallback to access evaluation results

  def _on_step(self):
    # Access the latest evaluation results
    if len(self.eval_callback.evaluations_results) > 0:
      # Get the latest mean reward from the EvalCallback
      mean_reward = self.eval_callback.evaluations_results[-1][0]  # Mean reward of the last evaluation
      self.trial.report(mean_reward, self.n_calls)

      # Check if the trial should be pruned
      if self.trial.should_prune():
        raise optuna.TrialPruned()
    return True


def optimal_trpo(trial):
  """
  {'gamma': 0.991067322494488, 'gae_lambda': 0.9239071133310828, 'target_kl': 0.007793948758802845, 'cg_damping': 0.08320338221830197, 'cg_max_steps': 18, 'line_search_max_iter': 7, 'n_steps': 2048, 'batch_size': 256, 'total_timesteps': 500000}
  """
  # Define the TRPO-specific optimization space
  params = {
      "gamma": trial.suggest_float("gamma", 0.98, 0.999, log=True),
      "gae_lambda": trial.suggest_float("gae_lambda", 0.9, 0.99),
      "target_kl": trial.suggest_float("target_kl", 0.005, 0.02),
      "cg_damping": trial.suggest_float("cg_damping", 0.01, 0.1),
      "cg_max_steps": trial.suggest_int("cg_max_steps", 10, 20),
      "line_search_max_iter": trial.suggest_int("line_search_max_iter", 5, 15),
      "n_steps": trial.suggest_categorical("n_steps", [1024, 2048, 4096]),
      "batch_size": trial.suggest_int("batch_size", 64, 256, step=64),
      "total_timesteps": trial.suggest_int("total_timesteps", 100_000, 500_000, step=100_000),
  }
  return params


def optimal_entrpo(trial):
  """
    {'gamma': 0.984822325244406, 'gae_lambda': 0.9558315622252784, 'target_kl': 0.011111159461187507, 'cg_damping': 0.07451122406533858, 'cg_max_steps': 20, 'line_search_max_iter': 8, 'ent_coef': 0.007312508870093316, 'n_steps': 1024, 'batch_size': 64, 'total_timesteps': 200000}
  """
  # Define the EnTRPO-specific optimization space
  params = {
      "gamma": trial.suggest_float("gamma", 0.98, 0.999, log=True),
      "gae_lambda": trial.suggest_float("gae_lambda", 0.9, 0.99),
      "target_kl": trial.suggest_float("target_kl", 0.005, 0.02),
      "cg_damping": trial.suggest_float("cg_damping", 0.01, 0.1),
      "cg_max_steps": trial.suggest_int("cg_max_steps", 10, 20),
      "line_search_max_iter": trial.suggest_int("line_search_max_iter", 5, 15),
      "ent_coef": trial.suggest_float("ent_coef", 0.0, 0.05),
      "n_steps": trial.suggest_int("n_steps", 1024, 4096, step=1024),
      "batch_size": trial.suggest_int("batch_size", 64, 256, step=64),
      "total_timesteps": trial.suggest_int("total_timesteps", 100_000, 500_000, step=100_000),
  }
  return params


def optimal_ppo(trial):
  """

  """
  # Define the PPO-specific optimization space
  params = {
      "gamma": trial.suggest_float("gamma", 0.98, 0.999, log=True),
      "gae_lambda": trial.suggest_float("gae_lambda", 0.9, 0.99),
      "n_steps": trial.suggest_int("n_steps", 1024, 4096, step=1024),
      "batch_size": trial.suggest_int("batch_size", 64, 256, step=64),
      "ent_coef": trial.suggest_float("ent_coef", 0.0, 0.05),
      "total_timesteps": trial.suggest_int("total_timesteps", 100_000, 500_000, step=100_000),
  }
  return params


def get_model():
  model_name = sys.argv[1].upper() if len(sys.argv) > 1 else "TRPO"

  match model_name:
    case "TRPO":
      return TRPO, model_name, optimal_trpo
    case "ENTRPO":
      return EnTRPO, model_name, optimal_entrpo
    case "PPO":
      return PPO, model_name, optimal_ppo
    case _:
      raise ValueError(f"Invalid model name: {model_name}")


def optimize_model(trial, model_class, optimal_function, model_name):
    # Get suggested parameters for the trial
  trial_params = optimal_function(trial)

  # Extract total timesteps
  total_timesteps = trial_params.pop("total_timesteps")

  # Set up environment
  num_envs = 12
  env = SubprocVecEnv([make_env(i) for i in range(num_envs)])
  eval_env = make_eval_env()
  tensorboard_log_dir = init_tensorboard(model_name)

  # Initialize model
  model = model_class(
      "MlpPolicy",
      env,
      tensorboard_log=tensorboard_log_dir,
      verbose=1,
      device="cpu",
      **trial_params
  )

  # Set up callbacks
  checkpoint_callback, _ = init_eval(env, model_name)
  success_eval_callback = SuccessEvalCallback(eval_env, eval_freq=5000, n_eval_episodes=10, deterministic=True)
  success_metrics_callback = SuccessMetricsCallback()

  # Train the model
  model.learn(
      total_timesteps=total_timesteps,
      callback=[checkpoint_callback, success_eval_callback, success_metrics_callback, TrialPruningCallback(trial, success_eval_callback)]
  )

  # Use success rate from evaluation as the optimization objective
  success_rate = success_eval_callback.success_rate
  return success_rate


if __name__ == "__main__":
  model_class, model_name, optimal_function = get_model()

  # Set up Optuna study
  study = optuna.create_study(direction="maximize", pruner=MedianPruner())
  study.optimize(lambda trial: optimize_model(trial, model_class, optimal_function, model_name), n_trials=20)

  # Print best trial
  print("Best trial:")
  print(study.best_trial.params)

  # Save Optuna results
  os.makedirs("./results", exist_ok=True)
  study.trials_dataframe().to_csv(f"./results/optuna_results_{model_name}.csv")
