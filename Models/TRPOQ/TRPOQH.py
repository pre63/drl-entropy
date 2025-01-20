import copy
from functools import partial
from typing import List, Type, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from sb3_contrib.common.utils import conjugate_gradient_solver
from stable_baselines3.common.distributions import kl_divergence
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.common.utils import explained_variance
from torch import nn
from torch.nn import functional as F

from Models.SB3 import TRPO
from Models.TRPOQ.Network import QuantileValueNetwork, optimize_hyperparameters

SelfTRPO = TypeVar("SelfTRPO", bound="TRPO")


class TRPOQH(TRPO):
  """
  Trust Region Policy Optimization with Quantile-Based Value Estimation (TRPOQ-Hybrid).

  TRPOQ-Hybrid extends standard TRPO by introducing dual critics, adaptive truncation,
  and corrective penalties to control overestimation bias in continuous control tasks.

  Key Features:
  - **Dual Critics:** Maintains separate critics for standard and truncated quantile estimations.
  - **Adaptive Truncation:** Dynamically adjusts the truncation threshold based on the variance of the critic's quantile outputs.
  - **Corrective Penalty:** Applies a penalty based on the difference between standard and truncated critic values to compensate for conservative bias.
  - **Multiple Value Networks:** Utilizes multiple critic networks for both standard and truncated values to reduce variance and improve stability.
  - **KL-Divergence Constraint:** Retains the core TRPO constraint to ensure policy stability.
  """

  def __init__(
      self,
      policy: Union[str, type[ActorCriticPolicy]],
      env: Union[GymEnv, str],
      learning_rate: Union[float, Schedule] = 1e-3,
      n_steps: int = 2048,
      batch_size: int = 128,
      gamma: float = 0.99,
      n_quantiles: int = 25,
      truncation_threshold: int = 5,
      n_value_networks: int = 2,
      adaptive_truncation: bool = True,
      penalty_coef: float = 0.01,
      net_arch: List[int] = [64, 64],
      activation_fn: Type[nn.Module] = nn.ReLU,
      **kwargs
  ):
    super().__init__(
        policy=policy,
        env=env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        gamma=gamma,
        **kwargs
    )
    self.n_quantiles = n_quantiles
    self.truncation_threshold = truncation_threshold
    self.n_value_networks = n_value_networks
    self.adaptive_truncation = adaptive_truncation
    self.penalty_coef = penalty_coef

    # Shared network backbone for both standard and truncated critics
    self.shared_value_network = QuantileValueNetwork(
        state_dim=env.observation_space.shape[0],
        n_quantiles=n_quantiles,
        net_arch=net_arch,
        activation_fn=activation_fn
    )

    # Separate heads for standard and truncated critics
    self.truncated_value_networks = nn.ModuleList([copy.deepcopy(self.shared_value_network) for _ in range(n_value_networks)])
    self.standard_value_networks = nn.ModuleList([copy.deepcopy(self.shared_value_network) for _ in range(n_value_networks)])

    if callable(learning_rate):
      learning_rate = learning_rate(0)

    # Optimizers for each set of critics
    self.value_optimizers = [th.optim.Adam(v.parameters(), lr=learning_rate) for v in self.standard_value_networks + self.truncated_value_networks]

  def _compute_truncated_value(self, states):
    all_quantiles = th.cat([v(states) for v in self.truncated_value_networks], dim=1)
    sorted_quantiles, _ = th.sort(all_quantiles, dim=1)

    if self.adaptive_truncation:
      # Adaptive truncation based on variance
      variances = th.var(sorted_quantiles, dim=1)
      adaptive_truncation_threshold = th.clamp((variances.mean() * self.truncation_threshold).long(), 1, self.n_quantiles)
    else:
      adaptive_truncation_threshold = self.truncation_threshold

    truncated_values = sorted_quantiles[:, :adaptive_truncation_threshold].mean(dim=1)
    return truncated_values

  def _compute_standard_value(self, states):
    all_quantiles = th.cat([v(states) for v in self.standard_value_networks], dim=1)
    return all_quantiles.mean(dim=1)

  def train(self):
    self.policy.set_training_mode(True)
    self._update_learning_rate(self.policy.optimizer)

    for rollout_data in self.rollout_buffer.get(batch_size=None):
      actions = rollout_data.actions
      if isinstance(self.action_space, spaces.Discrete):
        actions = rollout_data.actions.long().flatten()

      with th.no_grad():
        old_distribution = copy.deepcopy(self.policy.get_distribution(rollout_data.observations))

      distribution = self.policy.get_distribution(rollout_data.observations)
      log_prob = distribution.log_prob(actions)

      truncated_values = self._compute_truncated_value(rollout_data.observations)
      standard_values = self._compute_standard_value(rollout_data.observations)

      # Adaptive Penalty Calculation
      variance_penalty = (truncated_values - standard_values).abs().mean()
      penalty = self.penalty_coef * variance_penalty

      advantages = rollout_data.returns - truncated_values - penalty

      if self.normalize_advantage:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

      ratio = th.exp(log_prob - rollout_data.old_log_prob)
      policy_objective = (advantages * ratio).mean()
      kl_div = kl_divergence(distribution, old_distribution).mean()

      # Trust Region Optimization (Same as TRPO core)
      self.policy.optimizer.zero_grad()
      actor_params, policy_objective_gradients, grad_kl, grad_shape = self._compute_actor_grad(kl_div, policy_objective)

      search_direction = conjugate_gradient_solver(
          partial(self.hessian_vector_product, actor_params, grad_kl),
          policy_objective_gradients,
          max_iter=self.cg_max_steps
      )

      step_size = 2 * self.target_kl / (th.matmul(search_direction, self.hessian_vector_product(actor_params, grad_kl, search_direction)))
      step_size = th.sqrt(step_size)

      line_search_success = False
      original_actor_params = [param.detach().clone() for param in actor_params]

      for _ in range(self.line_search_max_iter):
        start_idx = 0
        for param, original_param, shape in zip(actor_params, original_actor_params, grad_shape):
          n_params = param.numel()
          param.data = (
              original_param.data + step_size * search_direction[start_idx: start_idx + n_params].view(shape)
          )
          start_idx += n_params

        distribution = self.policy.get_distribution(rollout_data.observations)
        log_prob = distribution.log_prob(actions)
        ratio = th.exp(log_prob - rollout_data.old_log_prob)
        new_policy_objective = (advantages * ratio).mean()
        kl_div = kl_divergence(distribution, old_distribution).mean()

        if kl_div < self.target_kl and new_policy_objective > policy_objective:
          line_search_success = True
          break
        step_size *= self.line_search_shrinking_factor

      # Value Update with Quantile Loss
      for value_network, value_optimizer in zip(self.standard_value_networks + self.truncated_value_networks, self.value_optimizers):
        predicted_quantiles = value_network(rollout_data.observations)
        target_quantiles = rollout_data.returns.unsqueeze(1).repeat(1, self.n_quantiles)
        quantile_loss = F.smooth_l1_loss(predicted_quantiles, target_quantiles)
        value_optimizer.zero_grad()
        quantile_loss.backward()
        value_optimizer.step()


class TRPOQHO(TRPO):
  """
  Trust Region Policy Optimization with Quantile-Based Value Estimation (TRPOQ-Hybrid Optimized).

  TRPOQ-Hybrid Optimized is an advanced policy optimization algorithm that combines elements of
  TRPO and quantile regression with several efficiency improvements. It is designed to reduce overestimation
  bias while maintaining computational efficiency and policy stability.

  Key Features:
  - **Single Critic with Dual Heads:** Uses a shared neural network for both standard and truncated quantile estimates, reducing parameter count and forward passes.
  - **Soft Truncation:** Applies a weighted average of lower quantiles instead of hard truncation, reducing the need for sorting operations.
  - **KL-Regularized Optimization:** Replaces TRPO's line search with a KL-divergence penalty term in the objective function for simpler and faster optimization.
  - **Gradient Clipping:** Implements gradient clipping for improved numerical stability during updates.
  - **Reduced Value Network Complexity:** Combines standard and truncated quantile critics into a single model with two output heads, reducing memory and computation costs.

  """

  def __init__(
      self,
      policy: Union[str, type[ActorCriticPolicy]],
      env: Union[GymEnv, str],
      learning_rate: Union[float, Schedule] = 1e-3,
      n_steps: int = 2048,
      batch_size: int = 128,
      gamma: float = 0.99,
      n_quantiles: int = 25,
      kl_coef: float = 0.01,  # Replaces explicit line search
      net_arch: List[int] = [64, 64],
      activation_fn: Type[nn.Module] = nn.ReLU,
      **kwargs
  ):
    super().__init__(
        policy=policy,
        env=env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        gamma=gamma,
        **kwargs
    )
    self.n_quantiles = n_quantiles
    self.kl_coef = kl_coef

    # Shared Critic Network with Dual Heads
    self.critic = QuantileValueNetwork(
        state_dim=env.observation_space.shape[0],
        n_quantiles=n_quantiles,
        net_arch=net_arch,
        activation_fn=activation_fn
    )

    if callable(learning_rate):
      learning_rate = learning_rate(0)

    # Optimizer for the shared critic
    self.critic_optimizer = th.optim.Adam(self.critic.parameters(), lr=learning_rate)

  def _compute_truncated_value(self, states):
    quantiles = self.critic(states)
    # Soft Truncation: Weighted sum of lower quantiles without sorting
    weights = th.linspace(1, 0, steps=self.n_quantiles, device=states.device)
    truncated_values = (quantiles * weights).sum(dim=1) / weights.sum()
    return truncated_values

  def train(self):
    self.policy.set_training_mode(True)
    self._update_learning_rate(self.policy.optimizer)

    policy_objective_values = []
    kl_divergences = []
    value_losses = []

    for rollout_data in self.rollout_buffer.get(batch_size=None):
      actions = rollout_data.actions
      if isinstance(self.action_space, spaces.Discrete):
        actions = rollout_data.actions.long().flatten()

      with th.no_grad():
        old_distribution = copy.deepcopy(self.policy.get_distribution(rollout_data.observations))

      distribution = self.policy.get_distribution(rollout_data.observations)
      log_prob = distribution.log_prob(actions)

      truncated_values = self._compute_truncated_value(rollout_data.observations)
      advantages = rollout_data.returns - truncated_values

      # Normalize Advantage
      if self.normalize_advantage:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

      ratio = th.exp(log_prob - rollout_data.old_log_prob)
      policy_objective = (advantages * ratio).mean()
      kl_div = kl_divergence(distribution, old_distribution).mean()

      # KL Regularization instead of Line Search
      policy_loss = -policy_objective + self.kl_coef * kl_div

      self.policy.optimizer.zero_grad()
      policy_loss.backward()
      th.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
      self.policy.optimizer.step()

      # Value Network Update
      predicted_quantiles = self.critic(rollout_data.observations)
      target_quantiles = rollout_data.returns.unsqueeze(1).repeat(1, self.n_quantiles)
      quantile_loss = F.smooth_l1_loss(predicted_quantiles, target_quantiles)

      self.critic_optimizer.zero_grad()
      quantile_loss.backward()
      self.critic_optimizer.step()

      # Log results
      policy_objective_values.append(policy_objective.item())
      kl_divergences.append(kl_div.item())
      value_losses.append(quantile_loss.item())

    explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())
    self.logger.record("train/policy_objective", np.mean(policy_objective_values))
    self.logger.record("train/value_loss", np.mean(value_losses))
    self.logger.record("train/kl_divergence_loss", np.mean(kl_divergences))
    self.logger.record("train/explained_variance", explained_var)
    if hasattr(self.policy, "log_std"):
      self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
    self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")


def sample_trpoqho_params(trial, n_actions, n_envs, additional_args):
  """
  Hyperparameter sampler for TRPOQHybridOptimized using Optuna.

  This method samples hyperparameters for the TRPOQHybridOptimized model, focusing on efficient
  value estimation with dual-head critics, soft truncation, and KL-regularization.

  :param trial: Optuna trial object
  :param n_actions: Number of actions in the environment
  :param n_envs: Number of parallel environments
  :param additional_args: Additional arguments for hyperparameter sampling
  :return: Dictionary of sampled hyperparameters for TRPOQHybridOptimized
  """
  n_steps = trial.suggest_categorical("n_steps", [512, 1024, 2048])
  gamma = trial.suggest_categorical("gamma", [0.98, 0.99, 0.995])
  learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)

  # KL and Quantile Specific Hyperparameters
  kl_coef = trial.suggest_float("kl_coef", 0.001, 0.1, log=True)
  n_quantiles = trial.suggest_categorical("n_quantiles", [25, 50, 75])

  # Neural network architecture selection
  net_arch_type = trial.suggest_categorical("net_arch", ["small", "medium", "large"])
  net_arch = {
      "small": [64, 64],
      "medium": [256, 256],
      "large": [512, 512]
  }[net_arch_type]

  # Activation function selection
  activation_fn_name = trial.suggest_categorical("activation_fn", ["tanh", "relu"])
  activation_fn = {
      "tanh": nn.Tanh,
      "relu": nn.ReLU
  }[activation_fn_name]

  batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128, 256, 512, 1024])

  return {
      "policy": "MlpPolicy",
      "n_steps": n_steps,
      "batch_size": batch_size,
      "gamma": gamma,
      "learning_rate": learning_rate,
      "kl_coef": kl_coef,
      "n_quantiles": n_quantiles,
      "net_arch": net_arch,
      "activation_fn": activation_fn
  }


def sample_trpoqh_params(trial, n_actions, n_envs, additional_args):
  """
  Hyperparameter sampler for TRPOQHybrid using Optuna.

  This method samples hyperparameters for TRPOQHybrid, focusing on dual critics,
  adaptive truncation, and corrective penalties for enhanced policy stability.

  :param trial: Optuna trial object
  :param n_actions: Number of actions in the environment
  :param n_envs: Number of parallel environments
  :param additional_args: Additional arguments for hyperparameter sampling
  :return: Dictionary of sampled hyperparameters for TRPOQHybrid
  """

  n_steps = trial.suggest_categorical("n_steps", [512, 1024, 2048])
  gamma = trial.suggest_categorical("gamma", [0.98, 0.99, 0.995])
  learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)

  # Critic and Truncation Hyperparameters
  n_critic_updates = trial.suggest_categorical("n_critic_updates", [5, 10, 20])
  n_value_networks = trial.suggest_categorical("n_value_networks", [2, 3, 5])
  penalty_coef = trial.suggest_float("penalty_coef", 0.001, 0.1, log=True)
  adaptive_truncation = trial.suggest_categorical("adaptive_truncation", [True, False])

  # Neural network architecture selection
  net_arch_type = trial.suggest_categorical("net_arch", ["small", "medium", "large"])
  net_arch = {
      "small": [64, 64],
      "medium": [256, 256],
      "large": [512, 512]
  }[net_arch_type]

  # Activation function selection
  activation_fn_name = trial.suggest_categorical("activation_fn", ["tanh", "relu"])
  activation_fn = {
      "tanh": nn.Tanh,
      "relu": nn.ReLU
  }[activation_fn_name]

  batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128, 256, 512, 1024])

  return {
      "policy": "MlpPolicy",
      "n_steps": n_steps,
      "batch_size": batch_size,
      "gamma": gamma,
      "learning_rate": learning_rate,
      "n_critic_updates": n_critic_updates,
      "n_value_networks": n_value_networks,
      "penalty_coef": penalty_coef,
      "adaptive_truncation": adaptive_truncation,
      "net_arch": net_arch,
      "activation_fn": activation_fn
  }
