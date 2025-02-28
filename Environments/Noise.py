import os
from datetime import datetime

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
    # Called after each environment step; not used for rollout data here
    return True

  def _on_rollout_end(self) -> None:
    # Called after each rollout is collected and before training
    if hasattr(self.model, "rollout_buffer"):
      rewards = self.model.rollout_buffer.rewards
      if rewards.size > 0:
        mean_reward = np.mean(rewards)
        self.rewards.append(mean_reward)
        print(f"Rollout end: Mean reward = {mean_reward}")

      # Attempt to get entropy from the policy distribution in the rollout buffer
      # Get the generator from rollout_buffer.get()
      for rollout_data in self.model.rollout_buffer.get(batch_size=None):
        observations = rollout_data.observations
        distribution = self.model.policy.get_distribution(observations)
        entropy_mean = distribution.entropy().mean().item()

        self.entropies.append(entropy_mean)


class EntropyInjectionWrapper(gym.Wrapper):
  def __init__(self, env, noise_config=None):
    """
        A wrapper to inject entropy into one component with a single entropy_level parameter.

        Args:
            env: Gymnasium environment with continuous action and observation spaces.
            noise_config: Dict with 'component' (obs, reward, action), 'type' (gaussian, uniform, laplace, bernoulli),
                          and 'entropy_level' (-1 to 1). If None, no noise is applied.
                          Example: {'component': 'obs', 'type': 'gaussian', 'entropy_level': 0.5}
        """
    super().__init__(env)
    if not (isinstance(self.action_space, gym.spaces.Box) and isinstance(self.observation_space, gym.spaces.Box)):
      raise ValueError("This wrapper is designed for continuous action and observation spaces only.")

    self.noise_config = noise_config if noise_config is not None else {}
    self._validate_config()

    # Base noise parameters (scaled by entropy_level)
    self.base_std = 1.0  # Gaussian
    self.base_range = 1.0  # Uniform
    self.base_scale = 1.0  # Laplace
    self.base_p = 0.5  # Bernoulli

  def _validate_config(self):
    if not self.noise_config:
      return
    if "component" not in self.noise_config or "type" not in self.noise_config or "entropy_level" not in self.noise_config:
      raise ValueError("noise_config must include 'component', 'type', and 'entropy_level'.")

    component = self.noise_config["component"]
    noise_type = self.noise_config["type"]
    entropy_level = self.noise_config["entropy_level"]

    if component not in ["obs", "reward", "action"]:
      raise ValueError("Component must be 'obs', 'reward', or 'action'.")
    if noise_type not in ["gaussian", "uniform", "laplace", "bernoulli"]:
      raise ValueError("Noise type must be 'gaussian', 'uniform', 'laplace', or 'bernoulli'.")
    if noise_type == "bernoulli" and component != "reward":
      raise ValueError("Bernoulli noise is only supported for rewards.")
    if not -1 <= entropy_level <= 1:
      raise ValueError("entropy_level must be between -1 and 1.")

  def _add_obs_noise(self, obs):
    if not self.noise_config or self.noise_config.get("component") != "obs":
      return obs
    noise_type = self.noise_config["type"]
    entropy_level = abs(self.noise_config["entropy_level"])  # Use absolute value for magnitude

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
    if not self.noise_config or self.noise_config.get("component") != "reward":
      return reward
    noise_type = self.noise_config["type"]
    entropy_level = abs(self.noise_config["entropy_level"])

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
      return 0 if np.random.uniform() < p else reward  # Fixed flipped_value=0
    return reward

  def _add_action_noise(self, action):
    if not self.noise_config or self.noise_config.get("component") != "action":
      return action
    noise_type = self.noise_config["type"]
    entropy_level = abs(self.noise_config["entropy_level"])

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
    """Handle seed and other kwargs passed by gymnasium and Stable Baselines3."""
    obs, info = self.env.reset(**kwargs)  # Unpack tuple from gymnasium reset
    if self.noise_config and self.noise_config.get("component") == "obs":
      noisy_obs = self._add_obs_noise(obs)
      obs = np.clip(noisy_obs, self.observation_space.low, self.observation_space.high)
    return obs, info  # Return tuple for SB3 compatibility

  def step(self, action):
    if self.noise_config and self.noise_config.get("component") == "action":
      noisy_action = self._add_action_noise(action)
      action_to_use = np.clip(noisy_action, self.action_space.low, self.action_space.high)
    else:
      action_to_use = action

    obs, reward, terminated, truncated, info = self.env.step(action_to_use)

    if self.noise_config and self.noise_config.get("component") == "obs":
      noisy_obs = self._add_obs_noise(obs)
      obs = np.clip(noisy_obs, self.observation_space.low, self.observation_space.high)

    reward = self._add_reward_noise(reward)

    return obs, reward, terminated, truncated, info


def generate_step_configs(component, noise_type, steps, min_level=-1.0, max_level=1.0):
  if steps < 1:
    raise ValueError("Number of steps must be at least 1.")
  if min_level >= max_level:
    raise ValueError("min_level must be less than max_level.")

  # Generate evenly spaced entropy levels
  entropy_levels = np.linspace(min_level, max_level, steps)

  # Create config list
  configs = [{"component": component, "type": noise_type, "entropy_level": float(level)} for level in entropy_levels]

  return configs


def run_training(model_class, env, config, total_timesteps, num_runs, dry_run=False):
  """Run training or simulate dry run for a given model and noise config."""
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

  # Average across runs
  max_reward_len = max(len(r) for r in run_rewards)
  max_entropy_len = max(len(e) for e in run_entropies)
  padded_rewards = [np.pad(r, (0, max_reward_len - len(r)), mode="edge") for r in run_rewards]
  padded_entropies = [np.pad(e, (0, max_entropy_len - len(e)), mode="edge") for e in run_entropies]

  avg_rewards = np.mean(padded_rewards, axis=0).tolist()
  avg_entropies = np.mean(padded_entropies, axis=0).tolist()

  return avg_rewards, avg_entropies


def smooth_data(training_data, window_size=3):
  """Smooth the rewards and entropies data."""
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
    smoothed_data.append({"label": data["label"], "rewards": smoothed_rewards.tolist(), "entropies": smoothed_entropies.tolist()})
  return smoothed_data


def plot_results(smoothed_results, run_date, model_name, total_timesteps, num_runs):
  """Generate and save tiled plots for rewards and entropies."""
  fig, axes = plt.subplots(nrows=2, ncols=len(smoothed_results), figsize=(24 * len(smoothed_results), 20), sharex=True, sharey="row")
  if len(smoothed_results) == 1:
    axes = axes.reshape(2, 1)

  colors = ["b", "g", "r", "c", "m", "y", "k", "orange", "purple", "brown", "pink"]

  for col_idx, result in enumerate(smoothed_results):
    noise_type = result["noise_type"]
    smoothed_data = result["smoothed_data"]

    ax1 = axes[0, col_idx]  # Rewards
    ax2 = axes[1, col_idx]  # Entropy

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


if __name__ == "__main__":
  models = [TRPOR, PPO, TRPO]
  dry_run = False
  run_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  env_name = "Humanoid-v5"

  total_timesteps = 100000
  steps = 10
  min_level = 0.01
  max_level = 0.1
  num_runs = 5

  # Define valid noise configurations as a dictionary
  VALID_NOISE_CONFIGS = {
    "obs": ["gaussian", "uniform", "laplace"],
    "reward": ["gaussian", "uniform", "laplace", "bernoulli"],
    "action": ["gaussian", "uniform", "laplace"],
  }

  for model_class in models:
    with open(f".hyperparameters/{model_class.__name__.lower()}.yml", "r") as file:
      model_hyperparameters = yaml.safe_load(file.read())

    all_results = []
    env_base = gym.make(env_name, render_mode=None)

    if not dry_run:
      # Iterate over valid component-noise combinations
      for component, noise_types in VALID_NOISE_CONFIGS.items():
        for noise_type in noise_types:
          configs = generate_step_configs(component, noise_type, steps, min_level, max_level)
          training_data = []

          for i, config in enumerate(configs):
            env = EntropyInjectionWrapper(env_base, noise_config=config)
            avg_rewards, avg_entropies = run_training(model_class, env, model_hyperparameters[env_name], total_timesteps, num_runs, dry_run)
            label = f"{config['component']}_{config['type']} ({config['entropy_level']:.2f})"
            training_data.append({"label": label, "rewards": avg_rewards, "entropies": avg_entropies})
            print(f"Averaged {num_runs} runs for {component} with {noise_type}: {label}")

          all_results.append({"noise_type": f"{component}_{noise_type}", "training_data": training_data})
    else:
      training_data = []
      avg_rewards, avg_entropies = run_training(model_class, env_base, model_hyperparameters[env_name], total_timesteps, num_runs, dry_run)
      training_data.append({"label": "Baseline", "rewards": avg_rewards, "entropies": avg_entropies})
      all_results.append({"noise_type": "none", "training_data": training_data})
      print(f"Dry run: Averaged {num_runs} baseline runs")

    # Smooth and plot
    smoothed_results = [{"noise_type": r["noise_type"], "smoothed_data": smooth_data(r["training_data"])} for r in all_results]
    plot_path = plot_results(smoothed_results, run_date, model_class.__name__, total_timesteps, num_runs)

    # Save data
    with open(f".noise/{run_date}/{model_class.__name__}_{total_timesteps}_noise_levels_{num_runs}_runs.yml", "w") as file:
      yaml.dump(smoothed_results, file)

  # Dump config with the actual noise types used
  used_noise_configs = {component: noise_types for component, noise_types in VALID_NOISE_CONFIGS.items()} if not dry_run else {"none": ["none"]}
  with open(f".noise/{run_date}/config.yml", "w") as file:
    yaml.dump(
      {
        "total_timesteps": total_timesteps,
        "steps": steps,
        "min_level": min_level,
        "max_level": max_level,
        "noise_configs": used_noise_configs,
        "num_runs": num_runs,
      },
      file,
    )
