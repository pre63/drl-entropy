import os
import numpy as np

import optuna

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, CheckpointCallback


from Environments.LunarLander import make


class TimestepsProgressCallback(BaseCallback):
  """
  A callback to dynamically adjust total timesteps if exceeded
  and log the overall progress of training.
  """

  def __init__(self, total_timesteps, verbose = 0):
    super().__init__(verbose)
    self.total_timesteps = total_timesteps
    self.interval = 1000 if total_timesteps > 1000000 else 100 if total_timesteps > 100000 else 10
    self.progress_step = total_timesteps // self.interval
    self.next_update = self.progress_step

  def _on_step(self) -> bool:
    current_timesteps = self.model.num_timesteps

    # Adjust total timesteps if exceeded
    if current_timesteps > self.total_timesteps:
      self.total_timesteps = current_timesteps
      self.progress_step = self.total_timesteps// self.interval
      self.next_update = current_timesteps + self.progress_step

    # Log progress if the next update step is reached
    if current_timesteps >= self.next_update:
      progress_percentage = (current_timesteps / self.total_timesteps) * 100
      print(f"[Progress: {progress_percentage:.2f}% ({current_timesteps}/{self.total_timesteps} timesteps)]")
      self.next_update += self.progress_step

    return True

  def _on_training_end(self) -> None:
    print(f"Training completed. Total timesteps: {self.model.num_timesteps}.")


class EvaluationMetricsCallback(EvalCallback):
  def __init__(self, eval_env, eval_freq=5000, n_eval_episodes=10, verbose=0, **kwargs):
    super().__init__(eval_env, eval_freq=eval_freq, n_eval_episodes=n_eval_episodes, verbose=verbose, **kwargs)
    self.eval_success_count = 0
    self.eval_total_episodes = 0
    self.eval_success_rate = 0.0
    self.eval_mean_reward = -1
    self.eval_mean_ep_length = -1
    self.eval_mean_success_rate = -1

  def _on_step(self) -> bool:
    if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
      episode_rewards, episode_lengths = self.evaluate()
      self.log_metrics(episode_rewards, episode_lengths)
    return True

  def evaluate(self):
    print("Evaluating model for {} episodes".format(self.n_eval_episodes))
    episode_rewards, episode_lengths = [], []
    success_count = 0

    for _ in range(self.n_eval_episodes):
      obs = self.eval_env.reset()
      done = False
      episode_reward = 0.0

      while not done:
        action, _ = self.model.predict(obs, deterministic=self.deterministic)
        obs, reward, done, info = self.eval_env.step(action)
        info = info[0]
        episode_reward += reward

        if "success" in info and info["success"]:
          success_count += 1

      episode_rewards.append(episode_reward)
      episode_lengths.append(info.get("episode", {}).get("l", 0))

    self.eval_success_count += success_count
    self.eval_total_episodes += self.n_eval_episodes
    self.eval_success_rate = self.eval_success_count / self.eval_total_episodes if self.eval_total_episodes > 0 else 0
    self.eval_mean_reward = np.mean(episode_rewards)
    self.eval_mean_ep_length = np.mean(episode_lengths)
    self.eval_mean_success_rate = success_count / self.n_eval_episodes

    return episode_rewards, episode_lengths

  def log_metrics(self, episode_rewards, episode_lengths):
    if self.logger:
      self.logger.record("eval/success_count", self.eval_success_count)
      self.logger.record("eval/success_rate", self.eval_success_rate)
      self.logger.record("eval/mean_reward", self.eval_mean_reward)
      self.logger.record("eval/mean_ep_length", self.eval_mean_ep_length)
      self.logger.record("eval/mean_success_rate", self.eval_mean_success_rate)


class TrainingMetricsCallback(BaseCallback):
  """
  A callback to track training metrics during steps and rollouts, including dynamic normalization values.
  """

  def __init__(self, max_reward, verbose=0):
    super().__init__(verbose)

    # Track reward history and normalization parameters
    self.reward_history = []
    self.max_reward = max_reward

    # Dynamic normalization values
    self.max_rollout_reward_variance = 1e-6  # Start with a small value to avoid division by zero
    self.max_convergence_metric = 1e-6

    # Step-level metrics
    self.step_success_count = 0
    self.step_total_episodes = 0
    self.step_success_rate = 0

    # Rollout-level metrics
    self.rollout_reward_history = []
    self.rollout_episode_lengths = []
    self.rollout_reward_variance = None
    self.rollout_converged = False

    self.steps = 0

  def _on_step(self):
    """
    Update step-level metrics during training.
    """
    self.steps += 1
    rewards = self.locals.get("rewards", [])
    self.reward_history.extend(rewards)

    infos = self.locals.get("infos", [])
    for info in infos:
      if info.get("success", False):
        self.step_success_count += 1

    return True

  def _on_rollout_end(self):
    """
    Update rollout-level metrics at the end of each training rollout.
    """
    infos = self.locals.get("infos", [])
    rewards = self.locals.get("rewards", [])

    self.step_total_episodes += len(infos)
    self.step_success_rate = self.step_success_count / self.step_total_episodes if self.step_total_episodes > 0 else 0

    self.rollout_reward_history.extend(rewards)
    self.rollout_episode_lengths.extend([info.get("episode", {}).get("l", 0) for info in infos if "episode" in info])

    if len(self.rollout_reward_history) > 1:
      self.rollout_reward_variance = np.var(self.rollout_reward_history)

    # Update dynamic max values
    if self.rollout_reward_variance and self.rollout_reward_variance > self.max_rollout_reward_variance:
      self.max_rollout_reward_variance = self.rollout_reward_variance

    if self.logger:
      self.logger.record("rollout/reward_variance", self.rollout_reward_variance)
      self.logger.record("rollout/success_rate", self.step_success_rate)
      self.logger.record("rollout/total_episodes", self.step_total_episodes)
      self.logger.record("rollout/success_count", self.step_success_count)

  def get_convergence_metric(self):
    """
    Compute a scalar metric representing training convergence, normalized to [0, 1].
    """
    if not self.reward_history:
      return 0

    # AUC (Area Under the Curve)
    auc = np.sum(self.reward_history)

    # Reward stability (1 - normalized variance)
    if len(self.reward_history) > 1:
      variance = np.var(self.reward_history)
      stability = 1 - variance / (self.max_reward - np.min(self.reward_history))
    else:
      stability = 1

    # Final reward (mean of the last 10% of rewards)
    final_rewards = self.reward_history[-max(1, len(self.reward_history) // 10):]
    final_avg_reward = np.mean(final_rewards)

    # Weighted Convergence Metric
    w1, w2, w3 = 0.5, 0.3, 0.2
    convergence_metric = w1 * auc + w2 * stability + w3 * final_avg_reward

    # Normalize convergence metric
    max_possible_convergence = self.steps * self.max_reward  # Maximum possible sum of rewards
    max_possible_convergence = max(max_possible_convergence, 1e-6)  # Avoid division by zero
    convergence_metric_norm = np.clip(convergence_metric / max_possible_convergence, 0, 1)

    return convergence_metric_norm

  def get_normalized_metrics(self):
    """
    Return normalized values for rollout_reward_variance and convergence_metric.
    """
    # Normalize rollout_reward_variance
    if self.rollout_reward_variance is not None:
      rollout_reward_variance_norm = self.rollout_reward_variance / self.max_rollout_reward_variance
    else:
      rollout_reward_variance_norm = 1.0

    # Normalize convergence_metric
    convergence_metric_norm = self.get_convergence_metric()

    return rollout_reward_variance_norm, convergence_metric_norm


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


def init_tensorboard(model_name):
  log_dir = "./.logs/"
  tensorboard_log_dir = os.path.join(log_dir, "tensorboard")
  os.makedirs(tensorboard_log_dir, exist_ok=True)
  return tensorboard_log_dir


def optimize_model(trial, model_class, optimal_function, model_name, num_envs=4):
  trial_params = optimal_function(trial)
  total_timesteps = trial_params.pop("total_timesteps")
  n_steps = trial_params.get("n_steps", 2048)
  buffer_size = n_steps * num_envs

  # Adjust batch size to be a divisor of the buffer size
  batch_size = trial_params.get("batch_size", 64)
  if buffer_size % batch_size != 0:
      # Find the largest divisor of buffer_size close to the original batch size
    divisors = [i for i in range(1, buffer_size + 1) if buffer_size % i == 0]
    batch_size = min(divisors, key=lambda x: abs(x - batch_size))
    trial_params["batch_size"] = batch_size
    print(f"Adjusted batch_size to {batch_size} to match buffer_size {buffer_size}")

  n_eval_episodes = max(100, total_timesteps // 200 // 5)
  env = SubprocVecEnv([make_env(i) for i in range(num_envs)])
  eval_env = make_eval_env()
  tensorboard_log_dir = init_tensorboard(model_name)

  model = model_class(
      "MlpPolicy",
      env,
      tensorboard_log=tensorboard_log_dir,
      verbose=1,
      device="cpu",
      **trial_params,
  )

  eval_callback = EvaluationMetricsCallback(
      eval_env=eval_env,
      eval_freq=5000,
      n_eval_episodes=n_eval_episodes,
      deterministic=True,
  )

  training_callback = TrainingMetricsCallback(
      max_reward=env.max_reward if hasattr(env, "max_reward") else 0,
  )

  progressCallback = TimestepsProgressCallback(total_timesteps)

  callbacks = [eval_callback, training_callback, progressCallback, TrialPruningCallback(trial, eval_callback)]

  model.learn(
      total_timesteps=total_timesteps,
      callback=callbacks,
  )

  trial_params_string = "_".join([f"{key}-{value}" for key, value in trial_params.items()])
  model.save(os.path.join(f".models/{model_name}", f"{model_name}_{trial_params_string}"))

  rollout_reward_variance_norm, convergence_metric_norm = training_callback.get_normalized_metrics()
  success_rate = eval_callback.eval_success_rate

  objective = success_rate
  print(f"Trial {trial.number} - Success Rate: {success_rate}, Reward Variance: {rollout_reward_variance_norm}, Convergence metric: {convergence_metric_norm}, Objective: {objective}")

  metrics = {
      "eval/success_count": eval_callback.eval_success_count,
      "eval/success_rate": eval_callback.eval_success_rate,
      "eval/mean_reward": eval_callback.eval_mean_reward,
      "eval/mean_ep_length": eval_callback.eval_mean_ep_length,
      "eval/mean_success_rate": eval_callback.eval_mean_success_rate,

      "step/success_count": training_callback.step_success_count,
      "step/total_episodes": training_callback.step_total_episodes,
      "step/success_rate": training_callback.step_success_rate,

      "rollout/reward_variance": training_callback.rollout_reward_variance,
      "rollout/success_rate": training_callback.step_success_rate,
      "rollout/success_count": training_callback.step_success_count,
      "rollout/total_episodes": training_callback.step_total_episodes,

      "reward_variance_norm": rollout_reward_variance_norm,
      "convergence_metric_norm": convergence_metric_norm,
      "total_timesteps": total_timesteps,
      "objective": objective,
  }

  trial.set_user_attr("metrics", metrics)

  return objective
