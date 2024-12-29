"""
MIT, based on Miguel Morales' implementation:
https://github.com/mimoralea/gym-walk
"""

import sys
import numpy as np
from six import StringIO
from string import ascii_uppercase
from typing import Optional
import matplotlib.pyplot as plt
import seaborn as sns
import pygame

import gymnasium as gym
from gymnasium import spaces, utils
from gymnasium.envs.toy_text.utils import categorical_sample
WEST, EAST = 0, 1


class WalkEnv(gym.Env):
  """
  Random Walk Environment for Reinforcement Learning.

  The environment represents a 1D grid with a start state, terminal states at both ends,
  and intermediate navigable states. The agent can move left, right, or stay in the same position.
  Rewards are provided for reaching the terminal states.

  Parameters:
  - n_states (int): Number of intermediate states (excluding start and terminal states).
  - p_stay (float): Probability of staying in the same state.
  - p_backward (float): Probability of moving backward.
  - max_blocks (int): Maximum number of steps allowed per episode.
  - render_mode (str): Rendering mode ("human", "ansi", or None).
  - verbose (int): Verbosity level for logging messages (0 for no logging).
  """

  metadata = {"render_modes": [None, "human", "ansi"], "render_fps": 5}

  def __init__(self, n_states=7, p_stay=0.0, p_backward=0.5, max_blocks=100, render_mode=None, verbose=0):
    # Set the verbosity level for logging messages; higher values may provide more detailed output.
    self.verbose = verbose

    # Specify the number of intermediate states in the environment.
    self.n_states = n_states

    # Define the number of possible actions:
    # - 0 corresponds to moving backward (WEST).
    # - 1 corresponds to moving forward (EAST).
    self.n_actions = 2

    # Define the grid shape, which includes intermediate states and terminal states.
    # The grid size is (1, n_states + 2), accounting for the start and terminal states.
    self.shape = (1, n_states + 2)

    # The agent starts at the center of the grid.
    self.start_state_index = self.shape[1] // 2

    # Assign a descriptive name to the environment.
    self.name = "RandomWalk"

    # Define the maximum number of blocks (steps) allowed during an episode.
    self.max_blocks = max_blocks

    # Initialize a counter to track the current step number within an episode.
    self.current_step = 0

    # Extract the number of rows and columns from the grid's shape.
    # Since this is a 1D environment, nrow will always be 1.
    self.nrow, self.ncol = self.shape

    # Define the action space as a discrete space with n_actions possible actions.
    self.action_space = spaces.Discrete(self.n_actions)

    # Define the observation space as a discrete space representing all possible grid positions.
    self.observation_space = spaces.Discrete(np.prod(self.shape))

    # Rendering settings
    self.render_mode = render_mode
    self._pygame_initialized = False

    if render_mode == "human":
      self._initialize_pygame()

    # State and action counts
    self.nS = np.prod(self.shape)
    self.nA = self.n_actions

    # Initialize transitions and sampling cache lazily
    self.P = None
    self.cached_samples = None

    # Define the initial state distribution (starting at the center).
    self.isd = np.zeros(self.nS)
    self.isd[self.start_state_index] = 1.0

    # Set the transition probabilities for the environment.
    self.p_stay = p_stay
    self.p_backward = p_backward

    # Placeholder for the last action taken by the agent.
    self.lastaction = None

    # Reset the environment to the initial state.
    self.reset()

  def _initialize_pygame(self):
    """
    Initialize Pygame for rendering the environment in human mode.
    """
    pygame.init()
    self._pygame_initialized = True
    self.screen_size = (800, 200)
    self.cell_width = self.screen_size[0] // (self.shape[1] - 2)
    self.screen = pygame.display.set_mode(self.screen_size)
    pygame.display.set_caption("Random Walk")
    self.clock = pygame.time.Clock()
    self.screen.fill((0, 0, 0))
    pygame.display.flip()

  def _initialize_transitions(self):
    """
    Precompute transition probabilities for all states and actions.
    This method is called lazily to reduce initialization overhead.
    """
    self.P = {}
    for s in range(self.nS):
      self.P[s] = {}
      for a in range(self.nA):
        p_forward = 1.0 - self.p_stay - self.p_backward
        s_forward = np.clip(s - 1 if a == WEST else s + 1, 0, self.nS - 1)
        s_backward = np.clip(s + 1 if a == WEST else s - 1, 0, self.nS - 1)
        r_forward = 1.0 if s == self.nS - 2 and s_forward == self.nS - 1 else 0.0
        r_backward = 1.0 if s == self.nS - 2 and s_backward == self.nS - 1 else 0.0
        d_forward = (s >= self.nS - 2 and s_forward == self.nS - 1) or (s <= 1 and s_forward == 0)
        d_backward = (s >= self.nS - 2 and s_backward == self.nS - 1) or (s <= 1 and s_backward == 0)

        self.P[s][a] = [
            (p_forward, s_forward, r_forward, d_forward),
            (self.p_stay, s, 0.0, s in [0, self.nS - 1]),
            (self.p_backward, s_backward, r_backward, d_backward),
        ]

  def _initialize_categorical_samples(self):
    """
    Precompute cumulative probabilities for transitions to speed up sampling during steps.
    """
    if not self.P:
      self._initialize_transitions()

    self.cached_samples = {}
    for s in range(self.nS):
      for a in range(self.nA):
        probs = [t[0] for t in self.P[s][a]]
        self.cached_samples[(s, a)] = np.cumsum(probs)

  def step(self, action):
    """
    Execute a step in the environment based on the provided action.

    Returns:
    - next_state (int): The next state of the agent.
    - reward (float): The reward received after taking the action.
    - terminated (bool): Whether the episode has terminated.
    - truncated (bool): Whether the episode was truncated due to max steps.
    - info (dict): Additional information about the step.
    """
    self.current_step += 1

    if not self.action_space.contains(action):
      return self.s, 0.0, True, True, {"success": False}

    if not self.cached_samples:
      self._initialize_categorical_samples()

    cumulative_probs = self.cached_samples[(self.s, action)]
    transitions = self.P[self.s][action]
    i = np.searchsorted(cumulative_probs, self.np_random.uniform(0, 1))
    prob, next_state, reward, terminated = transitions[i]

    self.s = next_state
    self.lastaction = action
    truncated = self.current_step >= self.max_blocks
    terminated = terminated or truncated

    info = {"prob": prob, "success": terminated and self.s in [0, self.nS - 1]}
    return int(next_state), reward, terminated, truncated, info

  def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
    """
    Reset the environment to its initial state.

    Parameters:
    - seed (int, optional): Random seed for reproducibility.
    - options (dict, optional): Additional options for the reset.

    Returns:
    - initial_state (int): The initial state of the environment.
    - info (dict): Additional information about the reset.
    """
    if seed is not None:
      super().reset(seed=seed)
      self.np_random, _ = gym.utils.seeding.np_random(seed)

    self.s = int(categorical_sample(self.isd, self.np_random))
    self.lastaction = None
    self.current_step = 0
    return int(self.s), {"prob": self.isd[self.s]}

  def render(self):
    if self.render_mode == "ansi":
      return self._ansi_render()
    elif self.render_mode == "human":
      self._human_render()

  def _ansi_render(self):
    """
    Render the environment in text mode.
    """
    outfile = StringIO()
    desc = np.asarray(["[" + ascii_uppercase[: self.shape[1] - 2] + "]"], dtype="c").tolist()
    desc = [[c.decode("utf-8") for c in line] for line in desc]
    color = "red" if self.s == 0 else "green" if self.s == self.nS - 1 else "yellow"
    desc[0][self.s] = utils.colorize(desc[0][self.s], color, highlight=True)
    outfile.write("\n")
    outfile.write("\n".join("".join(line) for line in desc) + "\n")
    return outfile.getvalue()

  def _human_render(self):
    """
    Render the environment using Pygame.
    """
    if self.verbose > 0:
      print("Rendering the environment with Pygame...")

    BG_COLOR = (30, 30, 30)
    TERMINAL_COLOR = (0, 255, 0)
    LEFT_TERMINAL_COLOR = (255, 0, 0)
    AGENT_COLOR = (255, 255, 0)
    CELL_COLOR = (200, 200, 200)

    pygame.event.pump()
    self.screen.fill(BG_COLOR)

    surface = pygame.Surface(self.screen_size)
    for i in range(self.shape[1] - 2):
      cell_x = i * self.cell_width
      cell_rect = pygame.Rect(cell_x, 50, self.cell_width, 100)
      color = (
          LEFT_TERMINAL_COLOR if i == 0 else
          TERMINAL_COLOR if i == self.nS - 1 else
          AGENT_COLOR if i == self.s else
          CELL_COLOR
      )
      pygame.draw.rect(surface, color, cell_rect)
      pygame.draw.rect(surface, (0, 0, 0), cell_rect, 2)

    self.screen.blit(surface, (0, 0))
    pygame.display.flip()
    self.clock.tick(self.metadata["render_fps"])


class ContinuousWalkEnv(WalkEnv):
  """
  Continuous Random Walk Environment for Reinforcement Learning.
  Extends the base WalkEnv to use continuous state and action spaces.
  """

  def __init__(self, n_states, p_stay, p_backward, max_blocks, render_mode, verbose):
    super().__init__(n_states, p_stay, p_backward, max_blocks, render_mode, verbose)
    self.observation_space = spaces.Box(
        low=0.0, high=float(n_states + 1), shape=(1,), dtype=np.float32
    )
    self.action_space = spaces.Box(
        low=-1.0, high=1.0, shape=(1,), dtype=np.float32
    )

  def step(self, action):
    """
    Override the step function to handle continuous actions.
    """
    self.current_step += 1

    if not self.action_space.contains(action):
      action = np.clip(action, self.action_space.low, self.action_space.high)

    # Continuous action controls movement
    action = action[0]  # Convert single-element array to scalar
    move = np.sign(action)  # Determine direction: -1 (left), +1 (right), 0 (stay)
    step_size = min(abs(action), 1.0)  # Limit step size to 1.0

    if move < 0:
      next_state = max(0, self.s - int(step_size))
    elif move > 0:
      next_state = min(self.nS - 1, self.s + int(step_size))
    else:
      next_state = self.s

    reward = 1.0 if next_state == self.nS - 1 else 0.0
    terminated = next_state in [0, self.nS - 1]
    truncated = self.current_step >= self.max_blocks
    self.s = next_state

    info = {"success": terminated and next_state == self.nS - 1}
    return int(self.s), reward, terminated, truncated, info


def make_random_walk(n_states=19, p_stay=0.1, p_backward=0.4, max_blocks=40, render_mode=None, verbose=0, continuous=False):
  """
  Create a Random Walk environment.

  Parameters:
  - n_states (int): Number of intermediate states.
  - p_stay (float): Probability of staying in the same state.
  - p_backward (float): Probability of moving backward.
  - max_blocks (int): Maximum steps per episode.
  - render_mode (str): Rendering mode.
  - verbose (int): Verbosity level for logging.
  - continuous (bool): Whether to use a continuous observation and action space.

  Returns:
  - env (WalkEnv or ContinuousWalkEnv): The configured Random Walk environment.
  """
  if continuous:
    env = ContinuousWalkEnv(n_states, p_stay, p_backward, max_blocks, render_mode, verbose)
  else:
    env = WalkEnv(n_states, p_stay, p_backward, max_blocks, render_mode, verbose)
  env.make_func_name = "make_random_walk"
  return env


def plot_transition_matrix(env):
  # Adjust for terminal states
  n_states = env.shape[1] - 2  # Exclude start and terminal states
  n_actions = env.n_actions
  transition_probs = np.zeros((n_states, n_actions, n_states))
  rewards = np.zeros((n_states, n_actions, n_states))

  for s in range(n_states):
    for a in range(n_actions):
      for prob, next_s, reward, terminated in env.P[s + 1][a]:  # Offset to skip start state
        if 0 <= next_s - 1 < n_states:  # Adjust indices for valid range
          transition_probs[s, a, next_s - 1] += prob
          rewards[s, a, next_s - 1] += reward

  # Plot transition probabilities for each action
  fig, axes = plt.subplots(1, n_actions, figsize=(15, 5))
  for a in range(n_actions):
    sns.heatmap(
        transition_probs[:, a, :],
        annot=True,
        fmt=".2f",
        cmap="Blues",
        cbar=True,
        ax=axes[a]
    )
    axes[a].set_title(f"Action {a}: Transition Probabilities")
    axes[a].set_xlabel("Next State")
    axes[a].set_ylabel("Current State")

  plt.tight_layout()
  plt.show()


def plot_rewards(env):
    # Adjust for terminal states
  n_states = env.shape[1] - 2  # Exclude start and terminal states
  n_actions = env.n_actions
  rewards = np.zeros((n_states, n_actions, n_states))

  for s in range(n_states):
    for a in range(n_actions):
      for _, next_s, reward, _ in env.P[s + 1][a]:  # Offset to skip start state
        if 0 <= next_s - 1 < n_states:  # Adjust indices for valid range
          rewards[s, a, next_s - 1] = reward

  # Aggregate and plot rewards
  aggregated_rewards = rewards.sum(axis=1)  # Aggregate rewards over actions
  plt.figure(figsize=(10, 6))
  sns.heatmap(
      aggregated_rewards,
      annot=True,
      fmt=".2f",
      cmap="YlGnBu",
      cbar=True
  )
  plt.title("Rewards Matrix (Aggregated over Actions)")
  plt.xlabel("Next State")
  plt.ylabel("Current State")
  plt.show()


def estimate_goal_probability(env, num_simulations=1000):
  """
  Estimate the probability of reaching the goal in a random walk MDP.

  Parameters:
  - env: The random walk environment.
  - num_simulations: Number of simulations to run.

  Returns:
  - success_probability: Estimated probability of reaching the goal.
  """
  successes = 0

  for _ in range(num_simulations):
    env.reset()
    done = False

    while not done:
      action = env.action_space.sample()  # Take a random action
      _, _, terminated, truncated, info = env.step(action)
      done = terminated or truncated

      if terminated and info.get("success", False):
        successes += 1
        break

  success_probability = successes / num_simulations
  print(f"Estimated probability of reaching the goal: {success_probability:.4f}")


if __name__ == "__main__":
  env = make_random_walk(n_states=19, p_stay=0.0, p_backward=0.5, max_blocks=50)
  estimate_goal_probability(env)
  plot_transition_matrix(env)
  plot_rewards(env)
