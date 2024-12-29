
import numpy as np
import gymnasium as gym


def make_lunar_lander(reward_strategy="default", render_mode=None, continuous=True):
  env = gym.make("LunarLanderContinuous-v3" if continuous else "LunarLander-v3", render_mode=render_mode)
  env = LunarLanderRewardWrapper(env, reward_strategy)
  env.make_func_name = "make_frozen_lake" if continuous else "make_lunar_lander_continuous"
  return env

def make_lunar_lander_continuous(reward_strategy="default", render_mode=None):
  return make_lunar_lander(reward_strategy, render_mode, continuous=True)

class LunarLanderRewardWrapper(gym.Wrapper):
  def __init__(self, env, reward_strategy):
    super().__init__(env)

    self.reward_strategy = reward_strategy

    if reward_strategy not in REWARD_STRATEGIES:
      raise ValueError(f"Unknown reward strategy: {reward_strategy}")

    self.reward_fn = REWARD_STRATEGIES[reward_strategy]

  def step(self, action):
    try:
      state, reward, terminated, truncated, info = self.env.step(action)
      custom_reward = self.reward_fn(state, reward, action, terminated or truncated, info)
      return state, custom_reward, terminated, truncated, info
    except Exception as e:
      print(f"Action: {action}")
      raise e


def default_reward(state, reward, action, done, info):
  """
  Default reward strategy from the environment.
  """
  info["landed"] = False if "landed" not in info else info["landed"]

  if done:
    success = check_success(state, done)
    if success:
      info["landed"] = True
      reward = max(reward + 200.0, 200.0)

  return reward


def proximity_reward(state, reward, action, done, info):
  """
  Reward strategy prioritizing proximity to the target and low velocity.
  """
  info["landed"] = False if "landed" not in info else info["landed"]

  x_position = state[0]

  if done:
    success = check_success(state, done)
    if success:
      info["landed"] = True
      award = 200.0 + -x_position**2
      reward = max(reward + award, 200.0)
    else:
      reward = reward - 100.0

  return reward


def energy_efficient_reward(state, reward, action, done, info):
  """
  Reward strategy prioritizing energy efficiency during landing.
  """
  info["landed"] = False if "landed" not in info else info["landed"]
  fuel_usage = np.linalg.norm(action)  # Action magnitude as a proxy for fuel usage
  x_position = state[0]

  if fuel_usage > 1.0:  # Penalize only excessive fuel usage
    reward -= 0.1 * (fuel_usage - 1.0)

  if done:
    success = check_success(state, done)
    if success:
      info["landed"] = True

      award = 200.0 + -x_position**2 * 0.1 * (fuel_usage - 1.0)

      reward = max(reward + award, 200.0)
    else:
      reward = reward - 100.0

  return reward


def combined_reward(state, reward, action, done, info):
  """
  Combines proximity and energy efficiency strategies.
  """
  proximity = proximity_reward(state, reward, action, done, info)
  efficiency = energy_efficient_reward(state, reward, action, done, info)

  # Adjust weights dynamically based on proximity
  distance_to_target = np.linalg.norm(state[:2])
  proximity_weight = 0.7 if distance_to_target > 1.0 else 0.9
  efficiency_weight = 1 - proximity_weight

  return proximity_weight * proximity + efficiency_weight * efficiency


import numpy as np


def check_success(observation,
                  terminated,
                  angle_threshold=0.1,
                  landing_pad_threshold=0.2,
                  x_velocity_threshold=0.01,
                  y_velocity_threshold=0.01):
  """
  Determine if the agent has successfully landed in the Continuous Lunar Lander environment.

  Parameters:
  - observation: list or array containing the state values from the environment.
  - terminated: boolean indicating if the episode has ended in a terminal state.
  - angle_threshold: maximum allowable angle (in radians) for success.
  - landing_pad_threshold: maximum allowable horizontal distance from the landing pad center.
  - x_velocity_threshold: maximum allowable horizontal velocity.
  - y_velocity_threshold: maximum allowable vertical velocity.

  Returns:
  - success: boolean indicating if the agent has successfully landed.
  """
  # Extract values from the observation
  x_position = observation[0]
  y_position = observation[1]
  x_velocity = observation[2]
  y_velocity = observation[3]
  angle = observation[4]
  angular_velocity = observation[5]
  leg_contact_left = observation[6]
  leg_contact_right = observation[7]

  # Define success criteria
  lander_is_upright = abs(angle) < angle_threshold
  lander_within_landing_pad = abs(x_position) < landing_pad_threshold
  velocities_within_limits = abs(x_velocity) < x_velocity_threshold and abs(y_velocity) < y_velocity_threshold
  legs_in_contact = leg_contact_left and leg_contact_right

  # Success condition
  success = terminated and lander_is_upright and lander_within_landing_pad and velocities_within_limits and legs_in_contact and y_position >= 0

  return success


# Registry for reward strategies
REWARD_STRATEGIES = {
    "default": default_reward,
    "proximity": proximity_reward,
    "energy": energy_efficient_reward,
    "combined": combined_reward,
}
