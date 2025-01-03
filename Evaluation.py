import os

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, CheckpointCallback

from Environments.LunarLander import make

from Callbacks.EvaluationMetrics import EvaluationMetrics
from Callbacks.TimestepsProgress import TimestepsProgress
from Callbacks.TrainingMetrics import TrainingMetrics
from Callbacks.TrialPruning import TrialPruning


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


def optimize_model(trial, model_class, optimal_function, model_name, total_timesteps=10000, num_envs=4):
  hyperparams = optimal_function(trial, total_timesteps)
  total_timesteps = hyperparams.pop("total_timesteps")
  n_steps = hyperparams.get("n_steps", 2048)
  buffer_size = n_steps * num_envs

  # Adjust batch size to be a divisor of the buffer size
  batch_size = hyperparams.get("batch_size", 64)
  if buffer_size % batch_size != 0:
      # Find the largest divisor of buffer_size close to the original batch size
    divisors = [i for i in range(1, buffer_size + 1) if buffer_size % i == 0]
    batch_size = min(divisors, key=lambda x: abs(x - batch_size))
    hyperparams["batch_size"] = batch_size
    print(f"Adjusted batch_size to {batch_size} to match buffer_size {buffer_size}")

  n_eval_episodes = max(100, total_timesteps // 200 // 5)
  env = SubprocVecEnv([make_env(i) for i in range(num_envs)])
  eval_env = make_eval_env()
  tensorboard_log_dir = init_tensorboard(model_name)

  model = model_class(**hyperparams, env=env, verbose=0, tensorboard_log=tensorboard_log_dir)

  eval_callback = EvaluationMetrics(
      eval_env=eval_env,
      eval_freq=5000,
      n_eval_episodes=n_eval_episodes,
      deterministic=True,
  )

  training_callback = TrainingMetrics(
      max_reward=env.max_reward if hasattr(env, "max_reward") else 0,
  )

  progressCallback = TimestepsProgress(total_timesteps)

  callbacks = [eval_callback, training_callback, progressCallback, TrialPruning(trial, eval_callback)]

  model.learn(
      total_timesteps=total_timesteps,
      callback=callbacks,
  )

  trial_params_string = "_".join([f"{key}-{round(value, 6) if isinstance(value, float) else value}" for key, value in hyperparams.items()])
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
      "eval/total_rewards": eval_callback.eval_total_rewards,

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
