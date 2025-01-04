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
    return env
  return _init


def make_eval_env(seed=0):
  """
  Create a SubprocVecEnv for evaluation to match the training environment type.
  """
  def _init():
    env = make()
    env.seed(seed)
    return env
  return SubprocVecEnv([_init])


def init_tensorboard(model_name):
  log_dir = "./.logs/"
  tensorboard_log_dir = os.path.join(log_dir, "tensorboard")
  os.makedirs(tensorboard_log_dir, exist_ok=True)
  return tensorboard_log_dir


def correct_batch_size(n_steps, n_envs):
  valid_batch_sizes = [bs for bs in [64, 128, 256, 512] if (n_steps * n_envs) % bs == 0]
  if not valid_batch_sizes:
    print(f"Changing batch size from {n_steps * n_envs} to {n_envs}")
    return n_steps * n_envs  # Fallback to full batch size if no valid batch size found
  return max(valid_batch_sizes)  # Return the largest valid batch size for efficiency


def optimize_model(trial, model_class, optimal_function, model_name, total_timesteps=10000, num_envs=4):
  hyperparams = optimal_function(trial, total_timesteps)
  total_timesteps = hyperparams.pop("total_timesteps")

  if "batch_size" in hyperparams and "n_steps" in hyperparams:
    hyperparams["batch_size"] = correct_batch_size(hyperparams["n_steps"], 1)

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
      "eval/success_count": to_python_native(eval_callback.eval_success_count),
      "eval/success_rate": to_python_native(eval_callback.eval_success_rate),
      "eval/mean_reward": to_python_native(eval_callback.eval_mean_reward),
      "eval/mean_ep_length": to_python_native(eval_callback.eval_mean_ep_length),
      "eval/mean_success_rate": to_python_native(eval_callback.eval_mean_success_rate),
      "eval/total_rewards": to_python_native(eval_callback.eval_total_rewards),

      "step/success_count": to_python_native(training_callback.step_success_count),
      "step/total_episodes": to_python_native(training_callback.step_total_episodes),
      "step/success_rate": to_python_native(training_callback.step_success_rate),

      "rollout/reward_variance": to_python_native(training_callback.rollout_reward_variance),
      "rollout/success_rate": to_python_native(training_callback.step_success_rate),
      "rollout/success_count": to_python_native(training_callback.step_success_count),
      "rollout/total_episodes": to_python_native(training_callback.step_total_episodes),

      "reward_variance_norm": to_python_native(rollout_reward_variance_norm),
      "convergence_metric_norm": to_python_native(convergence_metric_norm),
      "total_timesteps": to_python_native(total_timesteps),
      "objective": to_python_native(objective),
  }

  trial.set_user_attr("metrics", metrics)

  return objective
