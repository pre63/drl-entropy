"""
[I 2025-01-05 01:04:03,668] Trial 21 finished with value: 264.79548 and parameters: {'batch_size': 128, 'n_steps': 64, 'gamma': 0.9999, 
'learning_rate': 0.00041735117386320454, 'n_critic_updates': 30, 'cg_max_steps': 5, 'target_kl': 0.005, 
'gae_lambda': 0.99, 'net_arch': 'small', 'activation_fn': 'relu', 'n_quantiles': 50, 
'truncation_threshold': 20, 'n_value_networks': 5}. Best is trial 21 with value: 264.79548.

[I 2025-01-05 16:04:38,557] Trial 209 finished with value: 265.7609142 and parameters: {'batch_size': 64, 'n_steps': 1024, 'gamma': 0.99, 'learning_rate': 0.0009913127150303867, 'n_critic_updates': 25, 'cg_max_steps': 30, 'target_kl': 0.005, 'gae_lambda': 0.99, 'net_arch': 'medium', 'activation_fn': 'tanh', 'n_quantiles': 10, 'truncation_threshold': 10, 'n_value_networks': 5}. Best is trial 209 with value: 265.7609142.
Number of finished trials:  501
Best trial:
Value:  266.082071
Params: 
    batch_size: 256
    n_steps: 1024
    gamma: 0.99
    learning_rate: 0.0001193459439628414
    n_critic_updates: 20
    cg_max_steps: 30
    target_kl: 0.03
    gae_lambda: 0.99
    net_arch: small
    activation_fn: relu
    n_quantiles: 25
    truncation_threshold: 20
    n_value_networks: 3
Writing report to logs/trpoq2/report_LunarLanderContinuous-v3_500-trials-100000-tpe-median_1736117622
[I 2025-01-05 10:49:17,654] Trial 20 finished with value: 234.45948559999997 and parameters: 
{'batch_size': 128, 'n_steps': 512, 'gamma': 0.995, 'learning_rate': 0.0015285313400231586, 
'n_critic_updates': 25, 'cg_max_steps': 20, 'target_kl': 0.001, 'gae_lambda': 1.0, 'net_arch': 'medium', 
'activation_fn': 'tanh', 'n_quantiles': 10, 'truncation_threshold': 20, 'n_value_networks': 3}.
 Best is trial 20 with value: 234.45948559999997.
 
 [I 2025-01-05 13:03:09,205] Trial 52 finished with value: 261.5463718 and parameters: {'batch_size': 256, 'n_steps': 512, 'gamma': 0.9999, 'learning_rate': 0.0011772965012798017, 'n_critic_updates': 25, 'cg_max_steps': 5, 'target_kl': 0.001, 'gae_lambda': 0.98, 'net_arch': 'medium', 'activation_fn': 'relu', 'n_quantiles': 50, 'truncation_threshold': 20, 'n_value_networks': 7}. Best is trial 52 with value: 261.5463718.


"""

import copy
from functools import partial
from typing import TypeVar, Union

import torch as th
from gymnasium import spaces
from stable_baselines3.common.distributions import kl_divergence
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from torch import nn
from torch.nn import functional as F

from sb3_contrib.common.utils import conjugate_gradient_solver
from Models.TRPO import TRPO

SelfTRPO = TypeVar("SelfTRPO", bound="TRPO")


class QuantileValueNetwork(nn.Module):
  def __init__(self, state_dim, n_quantiles=25):
    super().__init__()
    self.n_quantiles = n_quantiles
    self.network = nn.Sequential(
        nn.Linear(state_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, n_quantiles)
    )

  def forward(self, state):
    return self.network(state)


class TRPOQ2(TRPO):
  """
  Trust Region Policy Optimization with Quantile Regression (TRPO-Q) implementation with guarantees.
  - Corrective Penalties: Compensates for conservative bias with a penalty term.
  - Dual Critics: Maintains both standard and truncated critics.
  - Adaptive Truncation: Dynamically adjusts truncation based on value variance.
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
      adaptive_truncation: bool = True,
      penalty_coef: float = 0.01,
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

    # Initialize dual critics
    self.truncated_value_networks = nn.ModuleList([
        QuantileValueNetwork(state_dim=env.observation_space.shape[0], n_quantiles=n_quantiles)
        for _ in range(n_value_networks)
    ])
    self.standard_value_networks = nn.ModuleList([
        QuantileValueNetwork(state_dim=env.observation_space.shape[0], n_quantiles=n_quantiles)
        for _ in range(n_value_networks)
    ])

    self.truncated_value_optimizers = [th.optim.Adam(v.parameters(), lr=learning_rate) for v in self.truncated_value_networks]
    self.standard_value_optimizers = [th.optim.Adam(v.parameters(), lr=learning_rate) for v in self.standard_value_networks]

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
        old_distribution = copy.copy(self.policy.get_distribution(rollout_data.observations))

      distribution = self.policy.get_distribution(rollout_data.observations)
      log_prob = distribution.log_prob(actions)

      truncated_values = self._compute_truncated_value(rollout_data.observations)
      standard_values = self._compute_standard_value(rollout_data.observations)

      # Advantage estimation with penalty for conservative bias
      advantages = rollout_data.returns - truncated_values
      penalty = self.penalty_coef * (truncated_values - standard_values).mean()
      advantages -= penalty

      if self.normalize_advantage:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

      ratio = th.exp(log_prob - rollout_data.old_log_prob)
      policy_objective = (advantages * ratio).mean()
      kl_div = kl_divergence(distribution, old_distribution).mean()

      self.policy.optimizer.zero_grad()
      actor_params, policy_objective_gradients, grad_kl, grad_shape = self._compute_actor_grad(kl_div, policy_objective)

      search_direction = conjugate_gradient_solver(
          partial(self.hessian_vector_product, actor_params, grad_kl),
          policy_objective_gradients,
          max_iter=self.cg_max_steps
      )

      step_size = 2 * self.target_kl / (th.matmul(search_direction, self.hessian_vector_product(actor_params, grad_kl, search_direction)))
      step_size = th.sqrt(step_size)

      # Line search with KL constraint
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

      if not line_search_success:
        for param, original_param in zip(actor_params, original_actor_params):
          param.data = original_param.data.clone()

      # Value function update with dual critics and quantile regression
      for _ in range(self.n_critic_updates):
        for networks, optimizers in [
            (self.truncated_value_networks, self.truncated_value_optimizers),
            (self.standard_value_networks, self.standard_value_optimizers)
        ]:
          for value_network, value_optimizer in zip(networks, optimizers):
            predicted_quantiles = value_network(rollout_data.observations)
            sorted_predicted, _ = th.sort(predicted_quantiles, dim=1)
            truncated_target = rollout_data.returns.unsqueeze(1).repeat(1, self.n_quantiles)
            quantile_loss = F.smooth_l1_loss(sorted_predicted, truncated_target)
            value_optimizer.zero_grad()
            quantile_loss.backward()
            value_optimizer.step()



def sample_trpoq2_params(trial, n_actions, n_envs, additional_args):
  """
  Sampler for TRPO with Quantile Value Estimation hyperparameters.

  :param trial: Optuna trial object
  :param n_actions: Number of actions in the environment
  :param n_envs: Number of parallel environments
  :param additional_args: Additional arguments for sampling
  :return: Dictionary of sampled hyperparameters
  """
  batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128, 256, 512])
  n_steps = trial.suggest_categorical("n_steps", [8, 16, 32, 64, 128, 256, 512, 1024, 2048])
  gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
  learning_rate = round(trial.suggest_float("learning_rate", 1e-5, 1, log=True), 6)

  n_critic_updates = trial.suggest_categorical("n_critic_updates", [5, 10, 20, 25, 30])
  cg_max_steps = trial.suggest_categorical("cg_max_steps", [5, 10, 20, 25, 30])
  target_kl = trial.suggest_categorical("target_kl", [0.1, 0.05, 0.03, 0.02, 0.01, 0.005, 0.001])

  # New hyperparameters for quantile-based value estimation
  n_quantiles = trial.suggest_categorical("n_quantiles", [10, 25, 50, 100])
  truncation_threshold = trial.suggest_categorical("truncation_threshold", [5, 10, 20])
  n_value_networks = trial.suggest_categorical("n_value_networks", [3, 5, 7])

  # Adjust batch size if it exceeds n_steps
  if batch_size > n_steps:
    batch_size = n_steps

  adaptive_truncation = trial.suggest_categorical("adaptive_truncation", [True, False])
  penalty_coef = trial.suggest_float("penalty_coef", 0.001, 0.1, log=True)

  return {
      "n_steps": n_steps,
      "batch_size": batch_size,
      "gamma": gamma,
      "cg_max_steps": cg_max_steps,
      "n_critic_updates": n_critic_updates,
      "target_kl": target_kl,
      "learning_rate": learning_rate,
      "n_quantiles": n_quantiles,
      "truncation_threshold": truncation_threshold,
      "n_value_networks": n_value_networks,
      "adaptive_truncation": adaptive_truncation,
      "penalty_coef": penalty_coef
  }