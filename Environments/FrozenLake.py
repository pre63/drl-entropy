import gymnasium as gym
from gymnasium.wrappers import TimeLimit


class FrozenLakeWrapper(gym.Wrapper):
  def __init__(self, env):
    super().__init__(env)
    self.name = "FrozenLake-v1"
    self.step_counter = 0
    self.observation = None
    self.nrow = self.env.unwrapped.nrow
    self.ncol = self.env.unwrapped.ncol

  def reset(self):
    self.step_counter = 0
    self.observation = None
    return self.env.reset()

  def step(self, action):
    try:
      state, reward, terminated, truncated, info = self.env.step(action)
    except Exception as e:
      print("An error occurred, action:", action)
      raise e

    # Add success info to info dict
    info["success"] = reward == 1

    # Penalty for not earning reward
    if reward == 0:
      reward = -0.001

    # Penalty for falling into the hole
    if terminated and state != 15:
      reward = -1

    # Penalty for not moving
    if self.observation is not None and self.observation == state:
      reward = -1

    self.observation = state
    self.step_counter += 1

    return state, reward, terminated, truncated, info


def make_frozen_lake(render_mode=None, max_episode_steps=50, desc=None, map_name="4x4", is_slippery=True):
  env = gym.make("FrozenLake-v1", render_mode=render_mode, desc=desc, map_name=map_name, is_slippery=is_slippery)
  env = TimeLimit(env, max_episode_steps=max_episode_steps)
  env = FrozenLakeWrapper(env)
  env.make_func_name = "make_frozen_lake"
  return env
