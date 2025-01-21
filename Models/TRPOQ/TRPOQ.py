import copy
from functools import partial
from typing import Any, Dict, List, Type, TypeVar, Union

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


class TRPOQ(TRPO):
  """
    Trust Region Policy Optimization with Quantile-Based Value Estimation (TRPOQ).

    TRPOQ extends the standard TRPO algorithm by incorporating a quantile-based critic architecture for
    value estimation. This method aims to reduce overestimation bias and stabilize advantage calculations
    during policy optimization, making it suitable for continuous control tasks.

    Key Features:
    - Quantile-Based Value Function Estimation: Uses an ensemble of critics outputting multiple quantiles instead of point estimates.
    - Conservative Truncation Strategy: The highest quantiles are discarded during advantage estimation to reduce overestimation bias.
    - Multiple Value Networks: Maintains multiple value networks for improved stability and reduced variance.
    - Trust Region Constraint: The standard TRPO KL-divergence constraint ensures stable policy updates.

    Args:
        policy (Union[str, type[ActorCriticPolicy]]): The policy model to use (e.g., "MlpPolicy").
        env (Union[GymEnv, str]): The environment for training. Can be a gym environment or an environment ID string.
        learning_rate (Union[float, Schedule], optional): Learning rate for the policy and value network optimization. Defaults to 1e-3.
        n_steps (int, optional): Number of steps to collect per iteration. Defaults to 2048.
        batch_size (int, optional): Mini-batch size for policy updates. Defaults to 128.
        gamma (float, optional): Discount factor for reward calculation. Defaults to 0.99.
        n_quantiles (int, optional): Number of quantiles used for value estimation. Defaults to 25.
        truncation_threshold (int, optional): Number of lower quantiles retained for conservative advantage estimation. Defaults to 5.
        n_value_networks (int, optional): Number of independent value networks used for ensemble estimation. Defaults to 3.
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
    n_value_networks: int = 3,
    net_arch: List[int] = [64, 64],
    activation_fn: Type[nn.Module] = nn.ReLU,
    **kwargs
  ):
    super().__init__(policy=policy, env=env, learning_rate=learning_rate, n_steps=n_steps, batch_size=batch_size, gamma=gamma, **kwargs)
    self.n_quantiles = n_quantiles
    self.truncation_threshold = truncation_threshold
    self.n_value_networks = n_value_networks

    # Initialize multiple quantile-based value functions
    self.value_networks = nn.ModuleList(
      [
        QuantileValueNetwork(state_dim=env.observation_space.shape[0], n_quantiles=n_quantiles, net_arch=net_arch, activation_fn=activation_fn)
        for _ in range(n_value_networks)
      ]
    )

    if callable(learning_rate):
      learning_rate = learning_rate(0)

    self.value_optimizers = [th.optim.Adam(v.parameters(), lr=learning_rate) for v in self.value_networks]

  def _compute_truncated_value(self, states):
    all_quantiles = th.cat([v(states) for v in self.value_networks], dim=1)
    sorted_quantiles, _ = th.sort(all_quantiles, dim=1)
    truncated_values = sorted_quantiles[:, : self.truncation_threshold].mean(dim=1)
    return truncated_values

  def train(self):
    self.policy.set_training_mode(True)
    self._update_learning_rate(self.policy.optimizer)

    policy_objective_values = []
    kl_divergences = []
    line_search_results = []
    value_losses = []

    for rollout_data in self.rollout_buffer.get(batch_size=None):
      actions = rollout_data.actions
      if isinstance(self.action_space, spaces.Discrete):
        actions = rollout_data.actions.long().flatten()

      with th.no_grad():
        old_distribution = copy.copy(self.policy.get_distribution(rollout_data.observations))

      distribution = self.policy.get_distribution(rollout_data.observations)
      log_prob = distribution.log_prob(actions)

      truncated_values = self._compute_truncated_value(rollout_data.observations)
      advantages = rollout_data.returns - truncated_values

      if self.normalize_advantage:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

      ratio = th.exp(log_prob - rollout_data.old_log_prob)
      policy_objective = (advantages * ratio).mean()
      kl_div = kl_divergence(distribution, old_distribution).mean()

      self.policy.optimizer.zero_grad()
      actor_params, policy_objective_gradients, grad_kl, grad_shape = self._compute_actor_grad(kl_div, policy_objective)

      search_direction = conjugate_gradient_solver(
        partial(self.hessian_vector_product, actor_params, grad_kl), policy_objective_gradients, max_iter=self.cg_max_steps
      )

      step_size = 2 * self.target_kl / (th.matmul(search_direction, self.hessian_vector_product(actor_params, grad_kl, search_direction)))
      step_size = th.sqrt(step_size)

      line_search_success = False
      original_actor_params = [param.detach().clone() for param in actor_params]

      for _ in range(self.line_search_max_iter):
        start_idx = 0
        for param, original_param, shape in zip(actor_params, original_actor_params, grad_shape):
          n_params = param.numel()
          param.data = original_param.data + step_size * search_direction[start_idx : start_idx + n_params].view(shape)
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

      if not line_search_success:
        for param, original_param in zip(actor_params, original_actor_params):
          param.data = original_param.data.clone()

      # Value function update with quantile regression loss
      for _ in range(self.n_critic_updates):
        for value_network, value_optimizer in zip(self.value_networks, self.value_optimizers):
          predicted_quantiles = value_network(rollout_data.observations)
          sorted_predicted, _ = th.sort(predicted_quantiles, dim=1)
          truncated_target = rollout_data.returns.unsqueeze(1).repeat(1, self.n_quantiles)
          quantile_loss = F.smooth_l1_loss(sorted_predicted, truncated_target)
          value_optimizer.zero_grad()
          quantile_loss.backward()
          value_optimizer.step()

      policy_objective_values.append(policy_objective.item())
      kl_divergences.append(kl_div.item())

    self._n_updates += 1
    explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

    self.logger.record("train/policy_objective", np.mean(policy_objective_values))
    self.logger.record("train/value_loss", np.mean(value_losses))
    self.logger.record("train/kl_divergence_loss", np.mean(kl_divergences))
    self.logger.record("train/explained_variance", explained_var)
    self.logger.record("train/is_line_search_success", np.mean(line_search_results))
    if hasattr(self.policy, "log_std"):
      self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
    self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")


def sample_trpoq_params(trial, n_actions, n_envs, additional_args):
  """
    Sampler for TRPO with Quantile Value Estimation hyperparameters.

    :param trial: Optuna trial object
    :param n_actions: Number of actions in the environment
    :param n_envs: Number of parallel environments
    :param additional_args: Additional arguments for sampling
    :return: Dictionary of sampled hyperparameters
    """
  n_steps = trial.suggest_categorical("n_steps", [8, 16, 32, 64, 128, 256, 512, 1024, 2048])
  gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
  learning_rate = trial.suggest_float("learning_rate", 1e-5, 1, log=True)

  batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128, 256, 512, 1024])

  n_critic_updates = trial.suggest_categorical("n_critic_updates", [5, 10, 20, 25, 30])
  cg_max_steps = trial.suggest_categorical("cg_max_steps", [5, 10, 20, 25, 30])
  target_kl = trial.suggest_categorical("target_kl", [0.1, 0.05, 0.03, 0.02, 0.01, 0.005, 0.001])

  # New hyperparameters for quantile-based value estimation
  truncation_threshold = trial.suggest_categorical("truncation_threshold", [5, 10, 20])
  n_value_networks = trial.suggest_categorical("n_value_networks", [3, 5, 7])

  # Adjust batch size if it exceeds n_steps
  if batch_size > n_steps:
    batch_size = n_steps

  network_params = optimize_hyperparameters(trial)

  n_timesteps = trial.suggest_int("n_timesteps", 100000, 1000000, step=100000)
  n_envs_choice = [2, 4, 6, 8, 10]
  n_envs = trial.suggest_categorical("n_envs", n_envs_choice)

  return {
    "policy": "MlpPolicy",
    "n_timesteps": n_timesteps,
    "n_envs": n_envs,
    "n_steps": n_steps,
    "batch_size": batch_size,
    "gamma": gamma,
    "cg_max_steps": cg_max_steps,
    "n_critic_updates": n_critic_updates,
    "target_kl": target_kl,
    "learning_rate": learning_rate,
    "truncation_threshold": truncation_threshold,
    "n_value_networks": n_value_networks,
    **network_params,
  }
