import copy
import random
import warnings
from collections import deque
from functools import partial
from typing import Any, ClassVar, Optional, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from sb3_contrib.common.utils import conjugate_gradient_solver, flat_grad
from sb3_contrib.trpo.policies import CnnPolicy, MlpPolicy, MultiInputPolicy
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.distributions import kl_divergence
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticPolicy, BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, RolloutBufferSamples, Schedule
from stable_baselines3.common.utils import explained_variance
from torch import nn
from torch.nn import functional as F

from Models.SB3 import TRPO


class ReplayBuffer:
  """
    A replay buffer with optional Prioritized Experience Replay (PER).
    """

  def __init__(self, capacity: int, use_per: bool = False):
    """
        Initialize the replay buffer.

        Args:
            capacity (int): Maximum number of items the buffer can hold.
            use_per (bool): Whether to use Prioritized Experience Replay (PER).
        """
    self.capacity = capacity
    self.use_per = use_per
    self.observations = deque(maxlen=capacity)
    self.actions = deque(maxlen=capacity)
    self.returns = deque(maxlen=capacity)
    self.advantages = deque(maxlen=capacity)
    self.old_log_prob = deque(maxlen=capacity)
    self.cumulative_rewards = deque(maxlen=capacity)  # Required only for PER

  def add(self, observations: th.Tensor, actions: th.Tensor, returns: th.Tensor, advantages: th.Tensor, old_log_prob: th.Tensor):
    """
        Add a batch of transitions to the buffer. Stores cumulative rewards if PER is enabled.

        Args:
            observations, actions, returns, advantages, old_log_prob (th.Tensor): Tensors for the experience batch.
        """
    batch_size = observations.shape[0]
    cumulative_reward = returns.sum().item()  # Calculate cumulative reward

    for i in range(batch_size):
      self.observations.append(observations[i].detach().clone())
      self.actions.append(actions[i].detach().clone())
      self.returns.append(returns[i].detach().clone())
      self.advantages.append(advantages[i].detach().clone())
      self.old_log_prob.append(old_log_prob[i].detach().clone())
      if self.use_per:
        self.cumulative_rewards.append(cumulative_reward)

  def sample_extend(self, batch_size: int, observations: th.Tensor, actions: th.Tensor, returns: th.Tensor, advantages: th.Tensor, old_log_prob: th.Tensor):
    """
        Sample a batch of transitions and extend the provided tensors with the sampled batch.

        If PER is enabled, experiences are sampled based on cumulative rewards.
        If PER is disabled, experiences are randomly sampled.

        Args:
            batch_size (int): Number of samples to retrieve.
        """
    if len(self.observations) < batch_size:
      return observations, actions, returns, advantages, old_log_prob

    # Use PER if enabled, otherwise random sampling
    if self.use_per:
      rewards_tensor = th.tensor(self.cumulative_rewards, dtype=th.float32)
      sampling_probs = th.softmax(rewards_tensor, dim=0).numpy()  # Softmax for stability
      indices = np.random.choice(len(self.observations), batch_size, p=sampling_probs)
    else:
      indices = random.sample(range(len(self.observations)), batch_size)

    # Collect the sampled experiences
    sampled_observations = th.stack([self.observations[i] for i in indices])
    sampled_actions = th.stack([self.actions[i] for i in indices])
    sampled_returns = th.stack([self.returns[i] for i in indices])
    sampled_advantages = th.stack([self.advantages[i] for i in indices])
    sampled_old_log_prob = th.stack([self.old_log_prob[i] for i in indices])

    # Concatenate with the incoming batch
    observations = th.cat([observations, sampled_observations], dim=0)
    actions = th.cat([actions, sampled_actions], dim=0)
    returns = th.cat([returns, sampled_returns], dim=0)
    advantages = th.cat([advantages, sampled_advantages], dim=0)
    old_log_prob = th.cat([old_log_prob, sampled_old_log_prob], dim=0)

    return observations, actions, returns, advantages, old_log_prob

  def clear(self):
    """Clear all stored data in the buffer."""
    self.observations.clear()
    self.actions.clear()
    self.returns.clear()
    self.advantages.clear()
    self.old_log_prob.clear()
    self.cumulative_rewards.clear()

  def __len__(self):
    """Return the current size of the buffer."""
    return len(self.observations)


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
    - Replay Buffer: Stores transitions for sequential processing.
    - Entropy Regularization: Controlled via `ent_coef`.
    - Reward Clearing Criterion: The buffer is cleared if the reward exceeds `reward_threshold`.
    - Batch Size: Mini-batch sampling controlled via the `batch_size` parameter.
    - KL Divergence Constraint: Retains the theoretical guarantees of TRPO using KL constraints.

    """

  def __init__(
    self,
    *args,
    epsilon=0.2,
    ent_coef=0.01,
    buffer_capacity=10000,
    reward_threshold=None,
    replay_strategy="EnTRPO",
    replay_strategy_threshold=0.0,
    batch_size=32,
    use_per=False,
    **kwargs
  ):
    super().__init__(*args, **kwargs)

    self.epsilon = epsilon
    self.ent_coef = ent_coef
    self.replay_buffer = ReplayBuffer(capacity=buffer_capacity, use_per=use_per)
    self.batch_size = batch_size
    self.reward_threshold = reward_threshold
    self.replay_strategy = replay_strategy
    self.replay_strategy_threshold = replay_strategy_threshold

    if reward_threshold is None:
      raise ValueError("reward_threshold must be provided for EnTRPO")

  def train(self) -> None:
    """
        Update policy using the currently gathered rollout buffer.
        """
    # Switch to train mode (this affects batch norm / dropout)
    self.policy.set_training_mode(True)
    # Update optimizer learning rate
    self._update_learning_rate(self.policy.optimizer)

    policy_objective_values = []
    kl_divergences = []
    line_search_results = []
    value_losses = []

    # This will only loop once (get all data in one go)
    for rollout_data in self.rollout_buffer.get(batch_size=None):
      # Optional: sub-sample data for faster computation
      if self.sub_sampling_factor > 1:
        rollout_data = RolloutBufferSamples(
          rollout_data.observations[:: self.sub_sampling_factor],
          rollout_data.actions[:: self.sub_sampling_factor],
          None,  # type: ignore[arg-type]  # old values, not used here
          rollout_data.old_log_prob[:: self.sub_sampling_factor],
          rollout_data.advantages[:: self.sub_sampling_factor],
          None,  # type: ignore[arg-type]  # returns, not used here
        )

      observations = rollout_data.observations
      actions = rollout_data.actions
      returns = rollout_data.returns
      advantages = rollout_data.advantages
      old_log_prob = rollout_data.old_log_prob

      # Clear the replay buffer if the reward threshold is exceeded
      returns_sum = th.sum(returns)
      if returns_sum >= self.reward_threshold:
        self.replay_buffer.clear()

      # Check if the reward threshold is exceeded and activate the replay strategy
      distribution = self.policy.get_distribution(observations)
      entropy_mean = distribution.entropy().mean().item()

      # Here we implmnet hte alernate strategies, not part of EnTRPO specs initially
      # depending on the 'direction' of the exploration based onthe entropy we add the
      # experience to from the replay buffer to the current batch
      # the premise is that if the environment is unknown, we should replay more
      # or vice versa depending on low or high direction
      activate_high = self.replay_strategy_threshold > entropy_mean and self.replay_strategy == "HIGH"
      activate_low = self.replay_strategy_threshold < entropy_mean and self.replay_strategy == "LOW"

      if activate_high or activate_low:
        # Sample data from the replay buffer and
        # concatenate with the current rollout data,
        # effectively doubling the batch size
        data = self.replay_buffer.sample_extend(self.batch_size, observations, actions, returns, advantages, old_log_prob)

        observations, actions, returns, advantages, old_log_prob = data

      # Add the current rollout data to the replay buffer for next iteration
      self.replay_buffer.add(rollout_data.observations, rollout_data.actions, rollout_data.returns, rollout_data.advantages, rollout_data.old_log_prob)

      if isinstance(self.action_space, spaces.Discrete):
        # Convert discrete action from float to long
        actions = actions.long().flatten()

      with th.no_grad():
        # Note: is copy enough, no need for deepcopy?
        # If using gSDE and deepcopy, we need to use `old_distribution.distribution`
        # directly to avoid Pyth errors.
        old_distribution = copy.copy(self.policy.get_distribution(observations))

      distribution = self.policy.get_distribution(observations)
      log_prob = distribution.log_prob(actions)

      if self.normalize_advantage:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

      # ratio between old and new policy, should be one at the first iteration
      ratio = th.exp(log_prob - old_log_prob)

      # surrogate policy objective
      if self.replay_strategy == "EnTRPO":
        # Entropy regularization on the advantage
        # this is the EnTRPO implementation
        policy_objective = (advantages * ratio).mean() + self.ent_coef * distribution.entropy().mean()
      else:
        # Alternative way of using entropy os to use it to guide
        # experience replay sampling based on how unknown or known
        # the environment is represented by the distribution
        policy_objective = (advantages * ratio).mean()

      # KL divergence
      kl_div = kl_divergence(distribution, old_distribution).mean()

      # Surrogate & KL gradient
      self.policy.optimizer.zero_grad()

      actor_params, policy_objective_gradients, grad_kl, grad_shape = self._compute_actor_grad(kl_div, policy_objective)

      # Hessian-vector dot product function used in the conjugate gradient step
      hessian_vector_product_fn = partial(self.hessian_vector_product, actor_params, grad_kl)

      # Computing search direction
      search_direction = conjugate_gradient_solver(
        hessian_vector_product_fn,
        policy_objective_gradients,
        max_iter=self.cg_max_steps,
      )

      # Maximal step length
      line_search_max_step_size = 2 * self.target_kl
      line_search_max_step_size /= th.matmul(search_direction, hessian_vector_product_fn(search_direction, retain_graph=False))
      line_search_max_step_size = th.sqrt(line_search_max_step_size)  # type: ignore[assignment, arg-type]

      line_search_backtrack_coeff = 1.0
      original_actor_params = [param.detach().clone() for param in actor_params]

      is_line_search_success = False
      with th.no_grad():
        # Line-search (backtracking)
        for _ in range(self.line_search_max_iter):
          start_idx = 0
          # Applying the scaled step direction
          for param, original_param, shape in zip(actor_params, original_actor_params, grad_shape):
            n_params = param.numel()
            param.data = original_param.data + line_search_backtrack_coeff * line_search_max_step_size * search_direction[
              start_idx : (start_idx + n_params)
            ].view(shape)
            start_idx += n_params

          # Recomputing the policy log-probabilities
          distribution = self.policy.get_distribution(observations)
          log_prob = distribution.log_prob(actions)

          # New policy objective
          ratio = th.exp(log_prob - old_log_prob)
          new_policy_objective = (advantages * ratio).mean()

          # New KL-divergence
          kl_div = kl_divergence(distribution, old_distribution).mean()

          # Constraint criteria:
          # we need to improve the surrogate policy objective
          # while being close enough (in term of kl div) to the old policy
          if (kl_div < self.target_kl) and (new_policy_objective > policy_objective):
            is_line_search_success = True
            break

          # Reducing step size if line-search wasn't successful
          line_search_backtrack_coeff *= self.line_search_shrinking_factor

        line_search_results.append(is_line_search_success)

        if not is_line_search_success:
          # If the line-search wasn't successful we revert to the original parameters
          for param, original_param in zip(actor_params, original_actor_params):
            param.data = original_param.data.clone()

          policy_objective_values.append(policy_objective.item())
          kl_divergences.append(0.0)
        else:
          policy_objective_values.append(new_policy_objective.item())
          kl_divergences.append(kl_div.item())

    # Critic update
    for _ in range(self.n_critic_updates):
      for rollout_data in self.rollout_buffer.get(self.batch_size):
        values_pred = self.policy.predict_values(observations)
        value_loss = F.mse_loss(returns, values_pred.flatten())
        value_losses.append(value_loss.item())

        self.policy.optimizer.zero_grad()
        value_loss.backward()
        # Removing gradients of parameters shared with the actor
        # otherwise it defeats the purposes of the KL constraint
        for param in actor_params:
          param.grad = None
        self.policy.optimizer.step()

    self._n_updates += 1
    explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

    # Logs
    self.logger.record("train/policy_objective", np.mean(policy_objective_values))
    self.logger.record("train/value_loss", np.mean(value_losses))
    self.logger.record("train/kl_divergence_loss", np.mean(kl_divergences))
    self.logger.record("train/explained_variance", explained_var)
    self.logger.record("train/is_line_search_success", np.mean(line_search_results))
    if hasattr(self.policy, "log_std"):
      self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

    self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")


class EnTRPOLow(EnTRPO):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.replay_strategy = "LOW"


class EnTRPOHigh(EnTRPO):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.replay_strategy = "HIGH"


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
  n_steps = trial.suggest_categorical("n_steps", [8, 16, 32, 64, 128, 256, 512, 1024, 2048])
  gamma = trial.suggest_categorical("gamma", [0.8, 0.85, 0.9, 0.95, 0.99])
  learning_rate = trial.suggest_float("learning_rate", 1e-5, 1, log=True)
  n_critic_updates = trial.suggest_categorical("n_critic_updates", [5, 10, 20, 25, 30])
  cg_max_steps = trial.suggest_categorical("cg_max_steps", [5, 10, 20, 25, 30])
  target_kl = trial.suggest_categorical("target_kl", [0.1, 0.05, 0.03, 0.02, 0.01, 0.005, 0.001])
  gae_lambda = trial.suggest_categorical("gae_lambda", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0])

  batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128, 256, 512, 1024])

  # Neural network architecture selection
  net_arch_type = trial.suggest_categorical("net_arch", ["small", "medium"])
  net_arch = {
    "small": dict(pi=[64, 64], vf=[64, 64]),
    "medium": dict(pi=[256, 256], vf=[256, 256]),
  }[net_arch_type]

  # Activation function selection
  activation_fn_name = trial.suggest_categorical("activation_fn", ["tanh", "relu"])
  activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation_fn_name]

  # Entropy coefficient for regularization
  ent_coef = trial.suggest_float("ent_coef", 0.0, 0.001, step=0.0001)

  # Replay buffer capacity and reward threshold for buffer clearing
  buffer_capacity = trial.suggest_int("buffer_capacity", 1000, 100000, step=1000)

  reward_threshold = trial.suggest_float("reward_threshold", 200, 500, step=50)

  # Strategy:
  # EnTRPO: replay all untill buffer clears
  # HIGH: replay when the entropy exceeeds a threshold
  # LOW: replay when the entropy is below a threshold
  replay_strategy_threshold = trial.suggest_float("replay_strategy_threshold", -10, 10)

  epsilon = trial.suggest_float("epsilon", 0.1, 0.9, step=0.05)

  use_per = trial.suggest_categorical("use_per", [True, False])

  orthogonal_init = trial.suggest_categorical("orthogonal_init", [True, False])

  n_timesteps = trial.suggest_int("n_timesteps", 100000, 1000000, step=100000)
  n_envs_choice = [2, 4, 6, 8, 10]
  n_envs = trial.suggest_categorical("n_envs", n_envs_choice)

  # Returning the sampled hyperparameters as a dictionary
  return {
    "policy": "MlpPolicy",
    "n_timesteps": n_timesteps,
    "n_envs": n_envs,
    "epsilon": epsilon,
    "use_per": use_per,
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
    "reward_threshold": reward_threshold,
    "replay_strategy_threshold": replay_strategy_threshold,
    "policy_kwargs": dict(
      net_arch=net_arch,
      activation_fn=activation_fn,
      ortho_init=orthogonal_init,
    ),
    **additional_args,
  }
