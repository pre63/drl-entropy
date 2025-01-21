import gymnasium as gym
import numpy as np
from gymnasium.wrappers import TimeLimit


class AcrobotContinuousWrapper(gym.Wrapper):
  def __init__(self, env):
    super().__init__(env)
    self.name = "AcrobotContinuous-v1"
    self.step_counter = 0
    self.total_reward = 0
    self.last_state = None

  def reset(self, **kwargs):
    self.step_counter = 0
    self.total_reward = 0
    self.last_state = None
    return self.env.reset(**kwargs)

  def step(self, action):
    state, reward, terminated, truncated, info = self.env.step(action)

    # Extract angles and angular velocities from the state
    angle1 = np.arctan2(state[1], state[0])  # First joint angle
    angle2 = np.arctan2(state[3], state[2])  # Second joint angle
    angular_velocity1 = state[4]  # First joint angular velocity
    angular_velocity2 = state[5]  # Second joint angular velocity

    success = is_success(state, action)
    info["success"] = success
    info["is_success"] = success
    reward += 10 if success else 0
    reward = min(0, reward)
    if success:
      terminated = True

    # Update tracking variables
    self.last_state = state
    self.total_reward += reward
    self.step_counter += 1

    return state, reward, terminated, truncated, info


def is_success(state, action, angle_threshold=0.1, velocity_threshold=0.5, torque_threshold=0.1):
  """
    Determines if the acrobot is in a successful state.

    Parameters:
    - state (ndarray): The observation from the environment.
    - action (ndarray): The action taken.
    - angle_threshold (float): Maximum allowable deviation from the upright position (in radians).
    - velocity_threshold (float): Maximum allowable angular velocity.
    - torque_threshold (float): Maximum allowable torque.

    Returns:
    - bool: True if the acrobot is in a success state, False otherwise.
    """
  x1, y1, x2, y2, angular_velocity1, angular_velocity2 = state
  torque = action[0]

  # Calculate the angles
  angle1 = np.arctan2(y1, x1)
  angle2 = np.arctan2(y2, x2)

  # Check success criteria
  is_upright = abs(angle1) < angle_threshold and abs(angle2) < angle_threshold
  is_stable = abs(angular_velocity1) < velocity_threshold and abs(angular_velocity2) < velocity_threshold
  is_low_torque = abs(torque) < torque_threshold

  return is_upright and is_stable and is_low_torque


def make(render_mode=None, max_episode_steps=500):
  env = gym.make("Acrobot-v1", render_mode=render_mode)
  env = TimeLimit(env, max_episode_steps=max_episode_steps)
  env = AcrobotContinuousWrapper(env)
  env.make_func_name = "make"
  env.max_reward = 0
  return env
