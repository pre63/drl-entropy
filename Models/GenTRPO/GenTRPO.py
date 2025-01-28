import copy
import random
from functools import partial
from typing import Any, ClassVar, Optional, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from sb3_contrib.common.utils import conjugate_gradient_solver, flat_grad
from sb3_contrib.trpo.policies import CnnPolicy, MlpPolicy, MultiInputPolicy
from stable_baselines3.common.distributions import kl_divergence
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, RolloutBufferSamples, Schedule
from stable_baselines3.common.utils import explained_variance
from torch import nn
from torch.nn import functional as F

from Models.SB3 import TRPO


class GenerativeReplayBuffer:
  def __init__(self, real_capacity, synthetic_capacity, relevance_function, generative_model, batch_size):
    """
        Replay buffer with generative replay inspired by PGR.

        Args:
            real_capacity: Maximum capacity for the real buffer.
            synthetic_capacity: Maximum capacity for the synthetic buffer.
            relevance_function: Function to prioritize transitions.
            generative_model: Generative model for creating synthetic transitions.
            batch_size: Number of transitions to sample for training.
        """
    self.real_capacity = real_capacity
    self.synthetic_capacity = synthetic_capacity
    self.real_buffer = []
    self.synthetic_buffer = []
    self.relevance_function = relevance_function
    self.generative_model = generative_model
    self.batch_size = batch_size

  def add_real(self, transition):
    """
        Adds a real transition to the real buffer.

        Args:
            transition: A tuple (state, action, reward, next_state, done).
        """
    if len(self.real_buffer) >= self.real_capacity:
      self.real_buffer.pop(0)
    self.real_buffer.append(transition)

  def generate_synthetic(self):
    """
        Generates synthetic transitions and adds them to the synthetic buffer.
        """
    if len(self.real_buffer) == 0:
      return  # No real transitions to condition on

    # Sample a subset of the real buffer to guide generation
    sampled_real = random.sample(self.real_buffer, min(10, len(self.real_buffer)))

    # Compute relevance scores
    relevance_scores = [self.relevance_function(t) for t in sampled_real]

    # Generate synthetic transitions using the generative model
    synthetic_transitions = self.generative_model.generate(sampled_real, relevance_scores)

    # Add generated transitions to the synthetic buffer
    for transition in synthetic_transitions:
      if len(self.synthetic_buffer) >= self.synthetic_capacity:
        self.synthetic_buffer.pop(0)
      self.synthetic_buffer.append(transition)

  def sample(self, num_samples):
    """
        Samples a batch of transitions from both real and synthetic buffers.

        Returns:
            A batch of transitions.
        """
    real_sample_size = num_samples // 2
    synthetic_sample_size = num_samples // 2

    real_sample = random.sample(self.real_buffer, min(real_sample_size, len(self.real_buffer)))
    synthetic_sample = random.sample(self.synthetic_buffer, min(synthetic_sample_size, len(self.synthetic_buffer)))

    return real_sample + synthetic_sample


def compute_sampling_parameters_gradual_linear(entropy, entropy_coeff, min_samples=0, max_samples=10000):
  """
    Computes the number of samples to draw based on entropy and binary coefficient behavior,
    ensuring a more gradual slope for linear adjustment for both positive and negative coefficients.

    Args:
        entropy: Scalar or array representing the entropy of observations (unnormalized).
        entropy_coeff: Scalar in range [-1, 1], non-zero, that determines max/min behavior.
        min_samples: Minimum number of samples.
        max_samples: Maximum number of samples.

    Returns:
        Number of samples to draw.
    """
  samples_range = max_samples - min_samples

  if entropy_coeff > 0:
    # Gradual scaling with max samples at zero entropy for positive coefficients
    factor = 1 - np.abs(entropy * (1 / (abs(entropy_coeff) * 10)))  # Gradual slope adjustment
    samples = samples_range * factor
  else:
    # Gradual scaling with min samples at zero entropy for negative coefficients
    factor = np.abs(entropy * (1 / (abs(entropy_coeff) * 10)))  # Gradual slope adjustment
    samples = samples_range * factor

  # Ensure samples stay within the min and max bounds
  return int(np.clip(samples + min_samples, min_samples, max_samples))


class GenTRPO(TRPO):

  def __init__(self, *args, epsilon=0.2, buffer_capacity=10000, batch_size=32, entropy_coeff=0.1, **kwargs):
    super().__init__(*args, **kwargs)

    self.epsilon = epsilon
    self.replay_buffer = GenerativeReplayBuffer(
      real_capacity=buffer_capacity,
      synthetic_capacity=buffer_capacity,
      relevance_function=self._compute_relevance,
      generative_model=self.policy,
      batch_size=batch_size,
    )
    self.batch_size = batch_size
    self.entropy_coeff = entropy_coeff

  def train(self) -> None:
      """
            Core TRPO training loop with rollout buffer usage.
            """
      self.policy.set_training_mode(True)
      self._update_learning_rate(self.policy.optimizer)

      policy_objective_values = []
      kl_divergences = []
      line_search_results = []
      value_losses = []

      for rollout_data in self.rollout_buffer.get(batch_size=None):
        observations = rollout_data.observations
        actions = rollout_data.actions
        returns = rollout_data.returns
        advantages = rollout_data.advantages
        old_log_prob = rollout_data.old_log_prob

        # Get the policy distribution and compute log probabilities
        distribution = self.policy.get_distribution(observations)
        log_prob = distribution.log_prob(actions)

        if self.normalize_advantage:
          advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Compute the policy objective and KL divergence
        ratio = th.exp(log_prob - old_log_prob)
        policy_objective = (advantages * ratio).mean() + self.entropy_coeff * distribution.entropy().mean()
        kl_div = kl_divergence(distribution, self.policy.get_distribution(observations)).mean()

        self.policy.optimizer.zero_grad()
        actor_params, policy_grad, grad_kl, grad_shape = self._compute_actor_grad(kl_div, policy_objective)

        # Hessian-vector product for conjugate gradient
        hessian_vector_product_fn = partial(self.hessian_vector_product, actor_params, grad_kl)
        search_direction = conjugate_gradient_solver(hessian_vector_product_fn, policy_grad, max_iter=self.cg_max_steps)

        # Line search to satisfy KL constraint
        max_step = th.sqrt(2 * self.target_kl / th.matmul(search_direction, hessian_vector_product_fn(search_direction)))
        line_search_backtrack_coeff = 1.0
        original_actor_params = [param.detach().clone() for param in actor_params]
        is_successful = False

        with th.no_grad():
          for _ in range(self.line_search_max_iter):
            start_idx = 0
            for param, original_param, shape in zip(actor_params, original_actor_params, grad_shape):
              param.data = original_param.data + line_search_backtrack_coeff * max_step * search_direction[start_idx : start_idx + param.numel()].view(shape)
              start_idx += param.numel()

            # Recompute the KL divergence and policy objective
            distribution = self.policy.get_distribution(observations)
            log_prob = distribution.log_prob(actions)
            ratio = th.exp(log_prob - old_log_prob)
            new_policy_objective = (advantages * ratio).mean()
            kl_div = kl_divergence(distribution, self.policy.get_distribution(observations)).mean()

            if (kl_div < self.target_kl) and (new_policy_objective > policy_objective):
              is_successful = True
              break
            line_search_backtrack_coeff *= self.line_search_shrinking_factor

        if not is_successful:
          # Revert to original parameters
          for param, original_param in zip(actor_params, original_actor_params):
            param.data = original_param.data.clone()

        line_search_results.append(is_successful)
        policy_objective_values.append(new_policy_objective.item() if is_successful else policy_objective.item())
        kl_divergences.append(kl_div.item())

      # Critic update
      for _ in range(self.n_critic_updates):
        for rollout_data in self.rollout_buffer.get(self.batch_size):
          values_pred = self.policy.predict_values(rollout_data.observations)
          value_loss = F.mse_loss(rollout_data.returns, values_pred.flatten())
          value_losses.append(value_loss.item())

          self.policy.optimizer.zero_grad()
          value_loss.backward()
          self.policy.optimizer.step()

      # Logging
      self._n_updates += 1
      explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())
      self.logger.record("train/policy_objective", np.mean(policy_objective_values))
      self.logger.record("train/value_loss", np.mean(value_losses))
      self.logger.record("train/kl_divergence", np.mean(kl_divergences))
      self.logger.record("train/explained_variance", explained_var)
      self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")

  def _compute_relevance(self, transition):
    """
        Computes the relevance of a transition based on the advantage estimate.

        Args:
            transition: A tuple (state, action, reward, _, done).

        Returns:
            A relevance score (float) for the given transition.
        """
    state, action, reward, _, done = transition
    with th.no_grad():
      value = self.policy.predict_values(th.tensor(state, dtype=th.float32).to(self.device)).item()

    advantage = reward - value  # Reward minus baseline (value estimate)
    return abs(advantage)  # Higher advantage magnitude indicates higher relevance


def sample_gentrpo_params(trial, n_actions, n_envs, additional_args):

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

  # Entropy coefficient for regularization, it affect the number of samples to draw and the exploration behavior
  entropy_coeff = trial.suggest_float("entropy_coeff", -1, 1, step=0.01)

  # Replay buffer capacity and reward threshold for buffer clearing
  buffer_capacity = trial.suggest_int("buffer_capacity", 1000, 100000, step=1000)

  epsilon = trial.suggest_float("epsilon", 0.1, 0.9, step=0.05)

  orthogonal_init = trial.suggest_categorical("orthogonal_init", [True, False])

  n_timesteps = trial.suggest_int("n_timesteps", 100000, 1000000, step=100000)
  n_envs = trial.suggest_categorical("n_envs", [2, 4, 6, 8, 10])

  # Returning the sampled hyperparameters as a dictionary
  return {
    "policy": "MlpPolicy",
    "n_timesteps": n_timesteps,
    "n_envs": n_envs,
    "epsilon": epsilon,
    "entropy_coeff": entropy_coeff,
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
    **additional_args,
  }
