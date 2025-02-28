import os
from datetime import datetime
from itertools import chain, combinations

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.ndimage import uniform_filter1d  # For smoothing
from stable_baselines3.common.callbacks import BaseCallback

from Models.GenTRPO.GenTRPO import GenTRPO
from Models.SB3 import PPO, TRPO
from Models.TRPOER.TRPOER import TRPOER
from Models.TRPOR.TRPOR import TRPOR


class TrainingDataCallback(BaseCallback):
  def __init__(self, verbose=0):
    super(TrainingDataCallback, self).__init__(verbose)
    self.rewards = []
    self.entropies = []

  def _on_step(self) -> bool:
    return True

  def _on_rollout_end(self) -> None:
    if hasattr(self.model, "rollout_buffer"):
      rewards = self.model.rollout_buffer.rewards
      if rewards.size > 0:
        mean_reward = np.mean(rewards)
        self.rewards.append(mean_reward)
        print(f"Rollout end: Mean reward = {mean_reward}")

      for rollout_data in self.model.rollout_buffer.get(batch_size=None):
        observations = rollout_data.observations
        distribution = self.model.policy.get_distribution(observations)
        entropy_mean = distribution.entropy().mean().item()
        self.entropies.append(entropy_mean)


class EntropyInjectionWrapper(gym.Wrapper):
  def __init__(self, env, noise_configs=None):
    super().__init__(env)
    if not (isinstance(self.action_space, gym.spaces.Box) and isinstance(self.observation_space, gym.spaces.Box)):
      raise ValueError("This wrapper is designed for continuous action and observation spaces only.")
    self.noise_configs = noise_configs if noise_configs is not None else []
    self._validate_configs()
    self.base_std = 1.0
    self.base_range = 1.0
    self.base_scale = 1.0
    self.base_p = 0.5

  def _validate_configs(self):
    if not self.noise_configs:
      return
    for config in self.noise_configs:
      required_keys = {"component", "type", "entropy_level"}
      if not all(key in config for key in required_keys):
        raise ValueError("Each noise_config must include 'component', 'type', and 'entropy_level'.")
      component, noise_type, entropy_level = config["component"], config["type"], config["entropy_level"]
      if component not in ["obs", "reward", "action"]:
        raise ValueError("Component must be 'obs', 'reward', or 'action'.")
      if noise_type not in ["gaussian", "uniform", "laplace", "bernoulli"]:
        raise ValueError("Noise type must be 'gaussian', 'uniform', 'laplace', or 'bernoulli'.")
      if noise_type == "bernoulli" and component != "reward":
        raise ValueError("Bernoulli noise is only supported for rewards.")
      if not -1 <= entropy_level <= 1:
        raise ValueError("entropy_level must be between -1 and 1.")

  def _add_obs_noise(self, obs):
    for config in self.noise_configs:
      if config["component"] == "obs":
        noise_type = config["type"]
        entropy_level = abs(config["entropy_level"])
        if noise_type == "gaussian":
          std = entropy_level * self.base_std
          return obs + np.random.normal(0, std, size=obs.shape)
        elif noise_type == "uniform":
          range_val = entropy_level * self.base_range
          return obs + np.random.uniform(-range_val, range_val, size=obs.shape)
        elif noise_type == "laplace":
          scale = entropy_level * self.base_scale
          return obs + np.random.laplace(0, scale, size=obs.shape)
    return obs

  def _add_reward_noise(self, reward):
    for config in self.noise_configs:
      if config["component"] == "reward":
        noise_type = config["type"]
        entropy_level = abs(config["entropy_level"])
        if noise_type == "gaussian":
          std = entropy_level * self.base_std
          return reward + np.random.normal(0, std)
        elif noise_type == "uniform":
          range_val = entropy_level * self.base_range
          return reward + np.random.uniform(-range_val, range_val)
        elif noise_type == "laplace":
          scale = entropy_level * self.base_scale
          return reward + np.random.laplace(0, scale)
        elif noise_type == "bernoulli":
          p = entropy_level * self.base_p
          return 0 if np.random.uniform() < p else reward
    return reward

  def _add_action_noise(self, action):
    for config in self.noise_configs:
      if config["component"] == "action":
        noise_type = config["type"]
        entropy_level = abs(config["entropy_level"])
        if noise_type == "gaussian":
          std = entropy_level * self.base_std
          return action + np.random.normal(0, std, size=action.shape)
        elif noise_type == "uniform":
          range_val = entropy_level * self.base_range
          return action + np.random.uniform(-range_val, range_val, size=action.shape)
        elif noise_type == "laplace":
          scale = entropy_level * self.base_scale
          return action + np.random.laplace(0, scale, size=action.shape)
    return action

  def reset(self, **kwargs):
    obs, info = self.env.reset(**kwargs)
    noisy_obs = self._add_obs_noise(obs)
    obs = np.clip(noisy_obs, self.observation_space.low, self.observation_space.high)
    return obs, info

  def step(self, action):
    noisy_action = self._add_action_noise(action)
    action_to_use = np.clip(noisy_action, self.action_space.low, self.action_space.high)
    obs, reward, terminated, truncated, info = self.env.step(action_to_use)
    noisy_obs = self._add_obs_noise(obs)
    obs = np.clip(noisy_obs, self.observation_space.low, self.observation_space.high)
    reward = self._add_reward_noise(reward)
    return obs, reward, terminated, truncated, info


def generate_step_configs(components, noise_type, steps, min_level=-1.0, max_level=1.0):
  if steps < 1 or min_level >= max_level:
    raise ValueError("Invalid steps or range.")
  entropy_levels = np.linspace(min_level, max_level, steps)
  configs = []
  for level in entropy_levels:
    config_list = [{"component": comp, "type": noise_type, "entropy_level": float(level)} for comp in components]
    configs.append(config_list)
  return configs


def run_training(model_class, env, config, total_timesteps, num_runs, dry_run=False):
  run_rewards = []
  run_entropies = []
  for run in range(num_runs):
    callback = TrainingDataCallback(verbose=1)
    model_config = config.copy()
    model_config["env"] = env
    model = model_class(**model_config)
    model.learn(total_timesteps=total_timesteps, callback=callback)
    rewards = callback.rewards if callback.rewards else [0]
    entropies = callback.entropies if callback.entropies else [0]
    run_rewards.append(rewards)
    run_entropies.append(entropies)
    print(f"Run {run+1}/{num_runs} completed.")
  max_reward_len = max(len(r) for r in run_rewards)
  max_entropy_len = max(len(e) for e in run_entropies)
  padded_rewards = [np.pad(r, (0, max_reward_len - len(r)), mode="edge") for r in run_rewards]
  padded_entropies = [np.pad(e, (0, max_entropy_len - len(e)), mode="edge") for e in run_entropies]
  return np.mean(padded_rewards, axis=0).tolist(), np.mean(padded_entropies, axis=0).tolist()


def smooth_data(training_data, window_size=3):
  smoothed_data = []
  for data in training_data:
    padded_rewards = np.pad(data["rewards"], (0, max(0, window_size - len(data["rewards"]))), mode="constant")
    padded_entropies = np.pad(data["entropies"], (0, max(0, window_size - len(data["entropies"]))), mode="constant")
    smoothed_rewards = (
      uniform_filter1d(padded_rewards, size=window_size, mode="nearest")[: len(data["rewards"])] if len(data["rewards"]) > 1 else data["rewards"]
    )
    smoothed_entropies = (
      uniform_filter1d(padded_entropies, size=window_size, mode="nearest")[: len(data["entropies"])] if len(data["entropies"]) > 1 else data["entropies"]
    )
    smoothed_data.append({"label": data["label"], "rewards": smoothed_rewards.tolist(), "entropies": smoothed_entropies.tolist(), "model": data["model"]})
  return smoothed_data


def plot_results(smoothed_results, run_date, model_name, total_timesteps, num_runs):
  fig, axes = plt.subplots(nrows=2, ncols=len(smoothed_results), figsize=(24 * len(smoothed_results), 20), sharex=True, sharey="row")
  if len(smoothed_results) == 1:
    axes = axes.reshape(2, 1)
  colors = ["b", "g", "r", "c", "m", "y", "k", "orange", "purple", "brown", "pink"]
  for col_idx, result in enumerate(smoothed_results):
    noise_type = result["noise_type"]
    smoothed_data = result["smoothed_data"]
    ax1, ax2 = axes[0, col_idx], axes[1, col_idx]
    for i, data in enumerate(smoothed_data):
      x = np.arange(len(data["rewards"]))
      ax1.plot(x, data["rewards"], label=data["label"], color=colors[i % len(colors)], linewidth=2)
      entropies_len = len(data["entropies"])
      padded_entropies = np.pad(data["entropies"], (0, len(x) - entropies_len), mode="edge") if entropies_len < len(x) else data["entropies"][: len(x)]
      ax2.plot(x, padded_entropies, label=data["label"], color=colors[i % len(colors)], linewidth=2)
    ax1.set_title(f"{noise_type.capitalize()} Noise - Rewards (Avg of {num_runs} Runs)")
    ax1.set_ylabel("Mean Reward")
    ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax1.grid(True)
    ax2.set_title(f"{noise_type.capitalize()} Noise - Entropy (Avg of {num_runs} Runs)")
    ax2.set_xlabel("Rollout")
    ax2.set_ylabel("Entropy")
    ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax2.grid(True)
  plt.tight_layout()
  os.makedirs(f".noise/{run_date}", exist_ok=True)
  plot_path = f".noise/{run_date}/{model_name}_{total_timesteps}_noise_levels_{num_runs}_runs.png"
  plt.savefig(plot_path)
  plt.close()
  return plot_path


def get_all_combinations(components):
  return list(chain.from_iterable(combinations(components, r) for r in range(1, len(components) + 1)))


def save_partial_results(run_date, model_name, total_timesteps, num_runs, noise_type, training_data):
  os.makedirs(f".noise/{run_date}/partial", exist_ok=True)
  temp_path = f".noise/{run_date}/partial/{model_name}_{noise_type}_{total_timesteps}_temp.yml"
  with open(temp_path, "w") as file:
    yaml.dump({"noise_type": noise_type, "training_data": training_data}, file)
  return temp_path


def update_summary(run_date, summary_data):
  os.makedirs(f".noise/{run_date}", exist_ok=True)
  temp_path = f".noise/{run_date}/summary_temp.yml"
  with open(temp_path, "w") as file:
    yaml.dump(summary_data, file)


def load_config_from_env(default_path=".noise/config.yml"):
  """Load config from the path specified in NOISE env var, fallback to default."""
  config_path = os.getenv("NOISE", default_path)
  config = {}
  try:
    with open(config_path, "r") as file:
      config = yaml.safe_load(file)
      print(f"Loaded config from {config_path}")
  except FileNotFoundError:
    print(f"Config file not found at {config_path}, using defaults")
  except Exception as e:
    print(f"Error loading config from {config_path}: {e}, using defaults")
  return config


if __name__ == "__main__":
  # Default values
  dry_run = False
  env_name = "Humanoid-v5"
  total_timesteps = 100000
  steps = 20
  min_level = -1
  max_level = 1
  num_runs = 5

  run_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

  # Load config from env var NOISE or default path
  config = load_config_from_env()

  # Set defaults, overridden by config if present
  models = [TRPOR, PPO, TRPO]
  dry_run = config.get("dry_run", dry_run)
  env_name = config.get("env_name", env_name)
  total_timesteps = config.get("total_timesteps", total_timesteps)
  steps = config.get("steps", steps)
  min_level = config.get("min_level", min_level)
  max_level = config.get("max_level", max_level)
  num_runs = config.get("num_runs", num_runs)

  VALID_NOISE_CONFIGS = config.get(
    "noise_configs",
    {
      "obs": ["gaussian", "uniform", "laplace"],
      "reward": ["gaussian", "uniform", "laplace", "bernoulli"],
      "action": ["gaussian", "uniform", "laplace"],
    },
  )

  ALL_COMPONENTS = config.get("components", ["obs", "reward", "action"])

  all_configs_results = []
  summary = {
    "run_date": run_date,
    "total_timesteps": total_timesteps,
    "num_runs": num_runs,
    "models_tested": [],
    "best_reward_config": None,
    "best_entropy_config": None,
  }
  max_reward_improvement = float("-inf")
  max_entropy_reduction = float("-inf")
  baseline_dict = {}

  # Save initial config
  os.makedirs(f".noise/{run_date}", exist_ok=True)
  used_noise_configs = VALID_NOISE_CONFIGS if not dry_run else {"none": ["none"]}
  config_data = {
    "total_timesteps": total_timesteps,
    "steps": steps,
    "min_level": min_level,
    "max_level": max_level,
    "noise_configs": used_noise_configs,
    "component_combinations": [list(combo) for combo in get_all_combinations(ALL_COMPONENTS)],
    "num_runs": num_runs,
    "env_name": env_name,
    "dry_run": dry_run,
  }
  with open(f".noise/{run_date}/config.yml", "w") as file:
    yaml.dump(config_data, file)

  for model_class in models:
    with open(f".hyperparameters/{model_class.__name__.lower()}.yml", "r") as file:
      model_hyperparameters = yaml.safe_load(file.read())
    all_results = []
    env_base = gym.make(env_name, render_mode=None)

    # Baseline run
    baseline_rewards, baseline_entropies = run_training(model_class, env_base, model_hyperparameters[env_name], total_timesteps, num_runs, dry_run)
    baseline_data = [{"label": "Baseline", "rewards": baseline_rewards, "entropies": baseline_entropies, "model": model_class.__name__}]
    all_results.append({"noise_type": "none", "training_data": baseline_data})
    save_partial_results(run_date, model_class.__name__, total_timesteps, num_runs, "none", baseline_data)
    baseline_dict[model_class.__name__] = {
      "final_reward": baseline_rewards[-1] if baseline_rewards else 0,
      "initial_entropy": baseline_entropies[0] if baseline_entropies else 0,
    }
    print(f"Baseline completed and saved for {model_class.__name__}")

    if not dry_run:
      # Single-component tests
      for component, noise_types in VALID_NOISE_CONFIGS.items():
        for noise_type in noise_types:
          configs = generate_step_configs([component], noise_type, steps, min_level, max_level)
          training_data = []
          for config_list in configs:
            env = EntropyInjectionWrapper(env_base, noise_configs=config_list)
            avg_rewards, avg_entropies = run_training(model_class, env, model_hyperparameters[env_name], total_timesteps, num_runs, dry_run)
            label = f"{config_list[0]['component']}_{config_list[0]['type']} ({config_list[0]['entropy_level']:.2f})"
            training_data.append({"label": label, "rewards": avg_rewards, "entropies": avg_entropies, "model": model_class.__name__})
            print(f"Averaged {num_runs} runs for {component} with {noise_type}: {label}")
          all_results.append({"noise_type": f"{component}_{noise_type}", "training_data": training_data})
          save_partial_results(run_date, model_class.__name__, total_timesteps, num_runs, f"{component}_{noise_type}", training_data)

      # Combination tests
      component_combinations = get_all_combinations(ALL_COMPONENTS)
      for combo in component_combinations:
        if len(combo) == 1:
          continue
        for noise_type in set.intersection(*[set(VALID_NOISE_CONFIGS[c]) for c in combo]):
          configs = generate_step_configs(combo, noise_type, steps, min_level, max_level)
          training_data = []
          for config_list in configs:
            env = EntropyInjectionWrapper(env_base, noise_configs=config_list)
            avg_rewards, avg_entropies = run_training(model_class, env, model_hyperparameters[env_name], total_timesteps, num_runs, dry_run)
            label = "+".join([f"{cfg['component']}_{cfg['type']} ({cfg['entropy_level']:.2f})" for cfg in config_list])
            training_data.append({"label": label, "rewards": avg_rewards, "entropies": avg_entropies, "model": model_class.__name__})
            print(f"Averaged {num_runs} runs for combo {combo} with {noise_type}: {label}")
          all_results.append({"noise_type": f"{'_'.join(combo)}_{noise_type}", "training_data": training_data})
          save_partial_results(run_date, model_class.__name__, total_timesteps, num_runs, f"{'_'.join(combo)}_{noise_type}", training_data)

    # Process and save model results
    smoothed_results = [{"noise_type": r["noise_type"], "smoothed_data": smooth_data(r["training_data"])} for r in all_results]
    plot_path = plot_results(smoothed_results, run_date, model_class.__name__, total_timesteps, num_runs)
    with open(f".noise/{run_date}/{model_class.__name__}_{total_timesteps}_noise_levels_{num_runs}_runs.yml", "w") as file:
      yaml.dump(smoothed_results, file)
    all_configs_results.extend(smoothed_results)

    # Update summary with partial best configs
    summary["models_tested"].append(model_class.__name__)
    for result in smoothed_results:
      if result["noise_type"] == "none":
        continue
      for data in result["smoothed_data"]:
        model_name = data["model"]
        baseline_final_reward = baseline_dict.get(model_name, {}).get("final_reward", 0)
        baseline_initial_entropy = baseline_dict.get(model_name, {}).get("initial_entropy", 0)
        final_reward = data["rewards"][-1] if data["rewards"] else 0
        final_entropy = data["entropies"][-1] if data["entropies"] else 0
        reward_improvement = final_reward - baseline_final_reward
        entropy_reduction = baseline_initial_entropy - final_entropy

        if reward_improvement > max_reward_improvement:
          max_reward_improvement = reward_improvement
          summary["best_reward_config"] = {"config": data["label"], "model": model_name, "improvement": float(reward_improvement)}
        if entropy_reduction > max_entropy_reduction:
          max_entropy_reduction = entropy_reduction
          summary["best_entropy_config"] = {"config": data["label"], "model": model_name, "reduction": float(entropy_reduction)}

    update_summary(run_date, summary)
    print(f"Model {model_class.__name__} completed and partial results saved")

  # Finalize summary
  with open(f".noise/{run_date}/summary.yml", "w") as file:
    yaml.dump(summary, file)
  if os.path.exists(f".noise/{run_date}/summary_temp.yml"):
    os.remove(f".noise/{run_date}/summary_temp.yml")

  print(f"Final summary saved: Best reward config = {summary['best_reward_config']}, Best entropy config = {summary['best_entropy_config']}")
