import copy
from functools import partial
from typing import Any, ClassVar, Optional, TypeVar, Union

import torch as th
from gymnasium import spaces
from sb3_contrib.common.utils import conjugate_gradient_solver
from stable_baselines3.common.distributions import kl_divergence
from stable_baselines3.common.type_aliases import RolloutBufferSamples
from stable_baselines3.common.utils import explained_variance
from torch import nn
from torch.nn import functional as F

from Models.SB3 import TRPO

SelfTRPOR = TypeVar("SelfTRPOR", bound="TRPOR")


class TRPOR(TRPO):
  """
    Trust Region Policy Optimization with Entropy Regularization (TRPO-R) implementation.
    """

  def __init__(
    self,
    *args,
    ent_coef: float = 0.01,
    **kwargs,
  ):
    super().__init__(*args, **kwargs)
    self.ent_coef = ent_coef
    self.tb_log_name = "TRPOR"

  def learn(self, n_timesteps, callback, **learn_kwargs):
    learn_kwargs["tb_log_name"] = self.tb_log_name
    return super().learn(n_timesteps, callback, **learn_kwargs)

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

      actions = rollout_data.actions
      if isinstance(self.action_space, spaces.Discrete):
        # Convert discrete action from float to long
        actions = rollout_data.actions.long().flatten()

      with th.no_grad():
        # Note: is copy enough, no need for deepcopy?
        # If using gSDE and deepcopy, we need to use `old_distribution.distribution`
        # directly to avoid PyTorch errors.
        old_distribution = copy.copy(self.policy.get_distribution(rollout_data.observations))

      distribution = self.policy.get_distribution(rollout_data.observations)
      log_prob = distribution.log_prob(actions)
      entropy = distribution.entropy().mean()

      advantages = rollout_data.advantages
      if self.normalize_advantage:
        advantages = (advantages - advantages.mean()) / (rollout_data.advantages.std() + 1e-8)

      # ratio between old and new policy, should be one at the first iteration
      ratio = th.exp(log_prob - rollout_data.old_log_prob)

      # surrogate policy objective, add entropy term
      policy_objective = (advantages * ratio).mean() + self.ent_coef * entropy

      # KL divergence
      kl_div = kl_divergence(distribution, old_distribution).mean()

      # Surrogate & KL gradient
      self.policy.optimizer.zero_grad()

      actor_params, policy_objective_gradients, grad_kl, grad_shape = self._compute_actor_grad(kl_div, policy_objective)

      # Hessian-vector dot product function used in the conjugate gradient step
      hessian_vector_product_fn = partial(self.hessian_vector_product, actor_params, grad_kl)

      # Computing search direction
      search_direction = conjugate_gradient_solver(hessian_vector_product_fn, policy_objective_gradients, max_iter=self.cg_max_steps)

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
          distribution = self.policy.get_distribution(rollout_data.observations)
          log_prob = distribution.log_prob(actions)

          # New policy objective
          ratio = th.exp(log_prob - rollout_data.old_log_prob)

          # Adding entropy term for regularization
          new_entropy = distribution.entropy().mean()
          new_policy_objective = (advantages * ratio).mean() + self.ent_coef * new_entropy

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
        values_pred = self.policy.predict_values(rollout_data.observations)
        value_loss = F.mse_loss(rollout_data.returns, values_pred.flatten())
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

    def mean(value_list):
      mean_ = 0.0
      for value in value_list:
        mean_ += value
      return float(mean_ / len(value_list))

    # Logs
    self.logger.record("train/policy_objective", mean(policy_objective_values))
    self.logger.record("train/value_loss", mean(value_losses))
    self.logger.record("train/kl_divergence_loss", mean(kl_divergences))
    self.logger.record("train/explained_variance", explained_var)
    self.logger.record("train/is_line_search_success", mean(line_search_results))
    if hasattr(self.policy, "log_std"):
      self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

    self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")


def sample_trpor_params(trial, n_actions, n_envs, additional_args):
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

  n_critic_updates = trial.suggest_categorical("n_critic_updates", [5, 10, 20, 25, 30])
  cg_max_steps = trial.suggest_categorical("cg_max_steps", [5, 10, 20, 25, 30])
  target_kl = trial.suggest_categorical("target_kl", [0.1, 0.05, 0.03, 0.02, 0.01, 0.005, 0.001])
  gae_lambda = trial.suggest_categorical("gae_lambda", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0])
  net_arch_type = trial.suggest_categorical("net_arch", ["small", "medium"])
  activation_fn_name = trial.suggest_categorical("activation_fn", ["tanh", "relu"])

  # New hyperparameters for quantile-based value estimation

  batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128, 256, 512, 1024])

  # Neural network architecture configuration
  net_arch = {
    "small": dict(pi=[64, 64], vf=[64, 64]),
    "medium": dict(pi=[256, 256], vf=[256, 256]),
  }[net_arch_type]

  activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation_fn_name]

  ent_coef = trial.suggest_float("ent_coef", 0.0, 0.001, step=0.0001)

  n_timesteps = trial.suggest_int("n_timesteps", 100000, 1000000, step=100000)
  n_envs_choice = [2, 4, 6, 8, 10]
  n_envs = trial.suggest_categorical("n_envs", n_envs_choice)

  return {
    "policy": "MlpPolicy",
    "n_timesteps": n_timesteps,
    "n_envs": n_envs,
    "ent_coef": ent_coef,
    "n_steps": n_steps,
    "batch_size": batch_size,
    "gamma": gamma,
    "cg_max_steps": cg_max_steps,
    "n_critic_updates": n_critic_updates,
    "target_kl": target_kl,
    "learning_rate": learning_rate,
    "gae_lambda": gae_lambda,
    "policy_kwargs": dict(
      net_arch=net_arch,
      activation_fn=activation_fn,
    ),
  }
