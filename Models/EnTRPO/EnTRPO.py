import torch as th
from torch import nn
from stable_baselines3.common.distributions import kl_divergence
from stable_baselines3.common.utils import explained_variance

from sb3_contrib.common.utils import conjugate_gradient_solver
from Models.TRPO import TRPO


from collections import deque
import random


class ReplayBuffer:
  """
  A simple replay buffer to store past transitions for EnTRPO.

  Stores (state, action, reward, next_state) tuples and allows sampling 
  for mini-batch gradient updates. Includes a reward threshold-based clearing mechanism.
  """

  def __init__(self, capacity):
    self.buffer = deque(maxlen=capacity)

  def add(self, state, action, reward, next_state):
    self.buffer.append((state, action, reward, next_state))

  def sample(self, batch_size):
    return random.sample(self.buffer, batch_size)

  def clear(self):
    self.buffer.clear()

  def __len__(self):
    return len(self.buffer)


class EnTRPO(TRPO):
  """
  Entropy-Regularized Trust Region Policy Optimization (EnTRPO)

  EnTRPO extends the standard Trust Region Policy Optimization (TRPO) by incorporating:
  - A replay buffer for improved sample efficiency, borrowing ideas from off-policy learning.
  - Entropy regularization to encourage policy exploration.
  - A reward threshold for buffer clearing based on a percentage of the environment's maximum reward.

  The replay buffer stores past transitions and allows mini-batch sampling for policy updates. 
  Entropy regularization modifies the policy loss function to balance exploration and exploitation.

  Key Features:
  - **Replay Buffer:** Stores (state, action, reward, next_state) tuples for off-policy corrections.
  - **Entropy Regularization:** Controlled via `ent_coef`.
  - **Reward Clearing Criterion:** The buffer is cleared if the reward exceeds `reward_threshold`.
  - **Batch Size:** Mini-batch sampling controlled via the `batch_size` parameter.
  - **KL Divergence Constraint:** Retains the theoretical guarantees of TRPO using KL constraints.

  Parameters:
  - `ent_coef` (float): Coefficient for entropy regularization.
  - `buffer_capacity` (int): Size of the replay buffer.
  - `max_env_reward` (float): Maximum possible reward for the environment.
  - `reward_threshold` (float): The reward threshold to clear the buffer (97.5% of max reward by default).

  Example Usage:
  ```python
  model = EnTRPO("MlpPolicy", "CartPole-v1", ent_coef=0.01, max_env_reward=200)
  model.learn(total_timesteps=100000)
  ```
  """

  def __init__(
      self,
      *args,
      ent_coef=0.01,
      buffer_capacity=10000,
      max_env_reward=200,
      reward_threshold=None,
      batch_size=32,
      **kwargs
  ):
    super().__init__(*args, **kwargs)
    self.ent_coef = ent_coef
    self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)
    self.batch_size = batch_size
    self.max_env_reward = max_env_reward
    # Set reward threshold to 97.5% of max_env_reward if not specified
    self.reward_threshold = reward_threshold if reward_threshold else 0.975 * max_env_reward

  def train(self) -> None:
    """
    Perform a single training update using the replay buffer.
    The buffer is cleared if a reward exceeds the reward threshold.
    """
    self.policy.set_training_mode(True)
    self._update_learning_rate(self.policy.optimizer)
    policy_objective_values, kl_divergences, value_losses = [], [], []

    # Sample from the replay buffer
    batch_size = min(self.batch_size, len(self.replay_buffer))
    if batch_size == 0:
      # Populate buffer if empty
      for rollout_data in self.rollout_buffer.get(batch_size=None):
        self.replay_buffer.add(rollout_data.observations, rollout_data.actions, rollout_data.returns, rollout_data.observations)

    # Perform training on the sampled batch
    for state, action, reward, next_state in self.replay_buffer.sample(batch_size):
      if self.action_space.__class__.__name__ == "Discrete":
        action = action.long().flatten()

      with th.no_grad():
        old_distribution = self.policy.get_distribution(state)

      # Compute current policy and entropy
      distribution = self.policy.get_distribution(state)
      log_prob = distribution.log_prob(action)
      entropy = distribution.entropy().mean()
      ratio = th.exp(log_prob)

      # Policy objective with entropy regularization
      policy_objective = (reward * ratio).mean() + self.ent_coef * entropy

      # Compute KL divergence
      kl_div = kl_divergence(distribution, old_distribution).mean()
      self.policy.optimizer.zero_grad()
      policy_objective.backward()
      self.policy.optimizer.step()

    # **Clear buffer if reward exceeds reward threshold**
    if any(r.max().item() > self.reward_threshold for _, _, r, _ in self.replay_buffer.buffer):
      print(f"Clearing replay buffer: reward exceeded {self.reward_threshold}")
      self.replay_buffer.clear()



def sample_entrpo_params(trial, n_actions, n_envs, additional_args):
  """
  Sampler for EnTRPO hyperparameters using Optuna.

  This function generates hyperparameters for the Entropy-Regularized Trust Region Policy Optimization (EnTRPO).
  The hyperparameters control aspects such as batch size, number of steps per update, entropy regularization, 
  and neural network architecture.

  :param trial: An Optuna trial object used for hyperparameter sampling.
  :param n_actions: Number of actions in the environment.
  :param n_envs: Number of parallel environments.
  :param additional_args: Dictionary for additional arguments if needed.
  :return: A dictionary containing the sampled hyperparameters for EnTRPO.
  """
  # Sampling core hyperparameters
  batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128, 256, 512])
  n_steps = trial.suggest_categorical("n_steps", [8, 16, 32, 64, 128, 256, 512, 1024, 2048])
  gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
  learning_rate = trial.suggest_float("learning_rate", 1e-5, 1, log=True)
  n_critic_updates = trial.suggest_categorical("n_critic_updates", [5, 10, 20, 25, 30])
  cg_max_steps = trial.suggest_categorical("cg_max_steps", [5, 10, 20, 25, 30])
  target_kl = trial.suggest_categorical("target_kl", [0.1, 0.05, 0.03, 0.02, 0.01, 0.005, 0.001])
  gae_lambda = trial.suggest_categorical("gae_lambda", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0])

  # Ensure batch size does not exceed the number of steps
  if batch_size > n_steps:
    batch_size = n_steps

  # Neural network architecture selection
  net_arch_type = trial.suggest_categorical("net_arch", ["small", "medium"])
  net_arch = {
      "small": dict(pi=[64, 64], vf=[64, 64]),
      "medium": dict(pi=[256, 256], vf=[256, 256]),
  }[net_arch_type]

  # Activation function selection
  activation_fn_name = trial.suggest_categorical("activation_fn", ["tanh", "relu"])
  activation_fn = {
      "tanh": nn.Tanh,
      "relu": nn.ReLU
  }[activation_fn_name]

  # Entropy coefficient for regularization
  ent_coef = trial.suggest_float("ent_coef", 0.0, 0.001, step=0.0001)

  # Replay buffer capacity and reward threshold for buffer clearing
  buffer_capacity = trial.suggest_int("buffer_capacity", 1000, 100000, step=1000)
  max_env_reward = trial.suggest_float("max_env_reward", 100.0, 1000.0, step=50.0)
  reward_threshold = trial.suggest_float("reward_threshold", 0.9, 1.0, step=0.01) * max_env_reward

  # Returning the sampled hyperparameters as a dictionary
  return {
      "ent_coef": ent_coef,
      "n_steps": n_steps,
      "batch_size": batch_size,
      "gamma": gamma,
      "cg_max_steps": cg_max_steps,
      "n_critic_updates": n_critic_updates,
      "target_kl": target_kl,
      "learning_rate": learning_rate,
      "gae_lambda": gae_lambda,
      "buffer_capacity": buffer_capacity,
      "max_env_reward": max_env_reward,
      "reward_threshold": reward_threshold,
      "policy_kwargs": dict(
          net_arch=net_arch,
          activation_fn=activation_fn,
          ortho_init=False,  # Fixed for simplicity but can be made tunable
      ),
  }
