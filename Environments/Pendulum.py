import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import numpy as np


class PendulumWrapper(gym.Wrapper):
  def __init__(self, env):
    super().__init__(env)
    self.name = "Pendulum-v1"
    self.step_counter = 0
    self.total_reward = 0
    self.last_angle = None
    self.last_velocity = None

  def reset(self, **kwargs):
    self.step_counter = 0
    self.total_reward = 0
    self.last_angle = None
    self.last_velocity = None
    return self.env.reset(**kwargs)

  def step(self, action):
    state, reward, terminated, truncated, info = self.env.step(action)

    # Extract pendulum angle (theta) and angular velocity (omega)
    angle = np.arctan2(state[1], state[0])  # Theta (angle) from cosine and sine
    velocity = state[2]  # Angular velocity (omega)

    success = is_success(state, action)
    info["success"] = success
    reward += 10 if success else 0
    reward = min(0, reward) 

    # Update tracking variables
    self.last_angle = angle
    self.last_velocity = velocity
    self.total_reward += reward
    self.step_counter += 1

    return state, reward, terminated, truncated, info


def is_success(state, action, angle_threshold=0.2, velocity_threshold=0.5, torque_threshold=0.1):
  """
  Determines if the pendulum is in a successful state.

  Parameters:
  - state (ndarray): The observation from the environment, [cos(theta), sin(theta), angular_velocity].
  - action (ndarray): The action taken, [torque].
  - angle_threshold (float): Maximum allowable deviation from the upright position (in radians).
  - velocity_threshold (float): Maximum allowable angular velocity.
  - torque_threshold (float): Maximum allowable torque.

  Returns:
  - bool: True if the pendulum is in a success state, False otherwise.
  """
  x, y, angular_velocity = state
  torque = action[0]

  # Calculate the angle from cos(theta) and sin(theta)
  angle = np.arctan2(y, x)

  # Check success criteria
  is_upright = abs(angle) < angle_threshold
  is_stable = abs(angular_velocity) < velocity_threshold
  is_low_torque = abs(torque) < torque_threshold

  return is_upright and is_stable and is_low_torque


def make_pendulum(render_mode=None, max_episode_steps=200):
  env = gym.make("Pendulum-v1", render_mode=render_mode)
  env = TimeLimit(env, max_episode_steps=max_episode_steps)
  env = PendulumWrapper(env)
  env.make_func_name = "make_pendulum"
  return env
