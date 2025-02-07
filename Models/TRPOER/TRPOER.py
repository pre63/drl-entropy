import bisect
import copy
from functools import partial

import numpy as np
import torch
import torch as th
from gymnasium import spaces
from sb3_contrib.common.utils import conjugate_gradient_solver
from stable_baselines3.common.distributions import kl_divergence
from stable_baselines3.common.utils import explained_variance
from torch import nn
from torch.nn import functional as F

from Models.SB3 import TRPO
from Models.Strategy import sampling_strategy


class PrioritizedReplayBuffer:
  def __init__(self, capacity, alpha=0.6):
    # Initializes the prioritized replay buffer with a fixed capacity and prioritization exponent alpha.
    # Transitions are stored in a sorted list as tuples: (negative_priority, counter, transition).
    # Including a counter prevents comparing transitions when priorities are equal.
    self.capacity = capacity
    self.alpha = alpha
    self.data = []  # Sorted list: highest priority (lowest negative value) comes first.
    self.max_priority = 1.0
    self.counter = 0  # Unique counter to break ties in priority

  def add(self, observations, actions, returns, advantages, old_log_prob):
    # Adds a batch of transitions to the buffer, inserting each transition in sorted order by priority.
    batch_size = observations.shape[0]
    for i in range(batch_size):
      obs = observations[i]
      act = actions[i]
      ret = returns[i]
      adv = advantages[i]
      lp = old_log_prob[i]

      # Compute the priority for the new transition and ensure it is a Python float.
      priority = float(self.max_priority**self.alpha)
      transition = (obs, act, ret, adv, lp)

      # Insert the transition into the sorted list.

      # The tuple is (negative_priority, counter, transition) to ensure proper ordering.
      bisect.insort(self.data, (-priority, self.counter, transition))
      self.counter += 1

      # If capacity is exceeded, remove the transition with the lowest priority (last in the sorted list).
      if len(self.data) > self.capacity:
        self.data.pop()

  def sample(self, sample_size):
    # Returns the top 'sample_size' transitions based on priority.
    actual_sample_size = min(sample_size, len(self.data))
    if actual_sample_size == 0:
      raise ValueError("Not enough samples in buffer to sample.")

    # Since the list is sorted (highest priority first), select the first actual_sample_size elements.
    selected = self.data[:actual_sample_size]

    # Unpack transitions from each tuple (ignoring the negative priority and counter).
    obs_list, act_list, ret_list, adv_list, lp_list = zip(*[entry[2] for entry in selected])
    return (
      torch.stack(obs_list),
      torch.stack(act_list),
      torch.stack(ret_list),
      torch.stack(adv_list),
      torch.stack(lp_list),
    )


class TRPOER(TRPO):
  """
    Trust Region Policy Optimization with Prioritized Experience Replay (TRPOER) with entropy regularization on policy objective.
    """

  def __init__(
    self, epsilon=0.2, entropy_coef=0.01, buffer_capacity=10000, batch_size=32, normalized_advantage=False, buffer_alpha=0.6, sampling_coef=0.5, **kwargs
  ):
    # Initializes TRPO with replay buffer integration.
    super().__init__(**kwargs)
    self.epsilon = epsilon
    self.entropy_coef = entropy_coef
    self.replay_buffer = PrioritizedReplayBuffer(buffer_capacity, buffer_alpha)
    self.batch_size = batch_size
    self.normalize_advantage = normalized_advantage
    self.sampling_coef = sampling_coef
    self.buffer_alpha = buffer_alpha

  def train(self) -> None:
    # Updates the policy using on-policy rollouts augmented with prioritized replay samples.
    self.policy.set_training_mode(True)
    self._update_learning_rate(self.policy.optimizer)

    policy_objective_values = []
    kl_divergences = []
    line_search_results = []
    value_losses = []

    # Gather all on-policy rollout data and add each batch to the replay buffer.
    rollout_data_list = list(self.rollout_buffer.get())
    on_policy_obs = th.cat([rd.observations for rd in rollout_data_list])
    on_policy_actions = th.cat([rd.actions for rd in rollout_data_list])
    on_policy_returns = th.cat([rd.returns for rd in rollout_data_list])
    on_policy_advantages = th.cat([rd.advantages for rd in rollout_data_list])
    on_policy_old_log_prob = th.cat([rd.old_log_prob for rd in rollout_data_list])
    for rd in rollout_data_list:
      self.replay_buffer.add(rd.observations, rd.actions, rd.returns, rd.advantages, rd.old_log_prob)

    # Compute entropy and determine how many replay samples to mix in.
    distribution = self.policy.get_distribution(on_policy_obs)
    entropy_mean = distribution.entropy().mean().item()
    num_replay_samples = sampling_strategy(entropy=entropy_mean, sampling_coef=self.sampling_coef, min_samples=0, max_samples=self.batch_size)

    if num_replay_samples > 0:
      replay_obs, replay_actions, replay_returns, replay_advantages, replay_old_log_prob = self.replay_buffer.sample(num_replay_samples)
      replay_obs = replay_obs.to(self.device)
      replay_actions = replay_actions.to(self.device)
      replay_returns = replay_returns.to(self.device)
      replay_advantages = replay_advantages.to(self.device)
      replay_old_log_prob = replay_old_log_prob.to(self.device)

      # Concatenate replay samples with the current on-policy batch.
      on_policy_obs = th.cat([on_policy_obs, replay_obs])
      on_policy_actions = th.cat([on_policy_actions, replay_actions])
      on_policy_returns = th.cat([on_policy_returns, replay_returns])
      on_policy_advantages = th.cat([on_policy_advantages, replay_advantages])
      on_policy_old_log_prob = th.cat([on_policy_old_log_prob, replay_old_log_prob])

    if isinstance(self.action_space, spaces.Discrete):
      on_policy_actions = on_policy_actions.long().flatten()

    with th.no_grad():
      old_distribution = copy.copy(self.policy.get_distribution(on_policy_obs))

    distribution = self.policy.get_distribution(on_policy_obs)
    log_prob = distribution.log_prob(on_policy_actions)

    if self.normalize_advantage:
      on_policy_advantages = (on_policy_advantages - on_policy_advantages.mean()) / (on_policy_advantages.std() + 1e-8)

    ratio = th.exp(log_prob - on_policy_old_log_prob)
    policy_objective = (on_policy_advantages * ratio).mean() + self.entropy_coef * distribution.entropy().mean()
    kl_div = kl_divergence(distribution, old_distribution).mean()

    self.policy.optimizer.zero_grad()
    actor_params, policy_objective_gradients, grad_kl, grad_shape = self._compute_actor_grad(kl_div, policy_objective)
    hessian_vector_product_fn = partial(self.hessian_vector_product, actor_params, grad_kl)
    search_direction = conjugate_gradient_solver(
      hessian_vector_product_fn,
      policy_objective_gradients,
      max_iter=self.cg_max_steps,
    )
    line_search_max_step_size = 2 * self.target_kl
    line_search_max_step_size /= th.matmul(search_direction, hessian_vector_product_fn(search_direction, retain_graph=False))
    line_search_max_step_size = th.sqrt(line_search_max_step_size)
    line_search_backtrack_coeff = 1.0
    original_actor_params = [param.detach().clone() for param in actor_params]
    is_line_search_success = False
    with th.no_grad():
      for _ in range(self.line_search_max_iter):
        start_idx = 0
        for param, original_param, shape in zip(actor_params, original_actor_params, grad_shape):
          n_params = param.numel()
          param.data = original_param.data + line_search_backtrack_coeff * line_search_max_step_size * search_direction[
            start_idx : (start_idx + n_params)
          ].view(shape)
          start_idx += n_params
        distribution = self.policy.get_distribution(on_policy_obs)
        log_prob = distribution.log_prob(on_policy_actions)
        ratio = th.exp(log_prob - on_policy_old_log_prob)
        new_policy_objective = (on_policy_advantages * ratio).mean()
        kl_div = kl_divergence(distribution, old_distribution).mean()
        if (kl_div < self.target_kl) and (new_policy_objective > policy_objective):
          is_line_search_success = True
          break
        line_search_backtrack_coeff *= self.line_search_shrinking_factor
      line_search_results.append(is_line_search_success)
      if not is_line_search_success:
        for param, original_param in zip(actor_params, original_actor_params):
          param.data = original_param.data.clone()
        policy_objective_values.append(policy_objective.item())
        kl_divergences.append(0.0)
      else:
        policy_objective_values.append(new_policy_objective.item())
        kl_divergences.append(kl_div.item())

    for _ in range(self.n_critic_updates):
      values_pred = self.policy.predict_values(on_policy_obs)
      value_loss = F.mse_loss(on_policy_returns, values_pred.flatten())
      value_losses.append(value_loss.item())
      self.policy.optimizer.zero_grad()
      value_loss.backward()
      for param in actor_params:
        param.grad = None
      self.policy.optimizer.step()

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


def sample_trpoer_params(trial, n_actions, n_envs, additional_args):
  # Sampling core hyperparameters
  n_steps = trial.suggest_categorical("n_steps", [8, 16, 32, 64, 128, 256, 512, 1024, 2048])
  gamma = trial.suggest_categorical("gamma", [0.8, 0.85, 0.9, 0.95, 0.99])
  learning_rate = trial.suggest_float("learning_rate", 1e-5, 1, log=True)
  n_critic_updates = trial.suggest_categorical("n_critic_updates", [5, 10, 20, 25, 30])
  cg_max_steps = trial.suggest_categorical("cg_max_steps", [5, 10, 20, 25, 30])
  target_kl = trial.suggest_categorical("target_kl", [0.1, 0.05, 0.03, 0.02, 0.01, 0.005, 0.001])
  gae_lambda = trial.suggest_categorical("gae_lambda", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0])

  batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024, 2048])

  # Neural network architecture selection
  net_arch_type = trial.suggest_categorical("net_arch", ["small", "medium", "large"])
  net_arch = {
    "small": dict(pi=[64, 64], vf=[64, 64]),
    "medium": dict(pi=[256, 256], vf=[256, 256]),
    "large": dict(pi=[400, 300], vf=[400, 300]),
  }[net_arch_type]

  # Activation function selection
  activation_fn_name = trial.suggest_categorical("activation_fn", ["tanh", "relu"])
  activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation_fn_name]

  # Entropy coefficient for regularization
  entropy_coef = trial.suggest_float("entropy_coef", -1, 1, step=0.01)
  sampling_coef = trial.suggest_float("sampling_coef", -1, 1, step=0.01)

  # Replay buffer capacity and reward threshold for buffer clearing
  buffer_capacity = trial.suggest_int("buffer_capacity", 10000, 100000, step=1000)

  epsilon = trial.suggest_float("epsilon", 0.1, 0.9, step=0.05)

  orthogonal_init = trial.suggest_categorical("orthogonal_init", [True, False])

  n_timesteps = trial.suggest_int("n_timesteps", 100000, 1000000, step=100000)
  n_envs_choice = [2, 4, 6, 8, 10]
  n_envs = trial.suggest_categorical("n_envs", n_envs_choice)

  normalized_advantage = trial.suggest_categorical("normalize_advantage", [True, False])

  buffer_alpha = trial.suggest_float("buffer_alpha", 0.1, 1, log=True)

  # Returning the sampled hyperparameters as a dictionary
  return {
    "policy": "MlpPolicy",
    "n_timesteps": n_timesteps,
    "n_envs": n_envs,
    "epsilon": epsilon,
    "entropy_coef": entropy_coef,
    "sampling_coef": sampling_coef,
    "n_steps": n_steps,
    "batch_size": batch_size,
    "gamma": gamma,
    "cg_max_steps": cg_max_steps,
    "n_critic_updates": n_critic_updates,
    "target_kl": target_kl,
    "learning_rate": learning_rate,
    "gae_lambda": gae_lambda,
    "buffer_capacity": buffer_capacity,
    "policy_kwargs": dict(
      net_arch=net_arch,
      activation_fn=activation_fn,
      ortho_init=orthogonal_init,
    ),
    "normalize_advantage": normalized_advantage,
    "buffer_alpha": buffer_alpha,
    **additional_args,
  }
