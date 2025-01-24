import copy
from functools import partial

import numpy as np
import torch as th
from gymnasium import spaces
from sb3_contrib.common.utils import conjugate_gradient_solver
from stable_baselines3.common.distributions import kl_divergence
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, RolloutBufferSamples, Schedule
from stable_baselines3.common.utils import explained_variance
from torch import nn
from torch.nn import functional as F

from Models.SB3 import TRPO


class TRPOR(TRPO):
  """
    TRPOR: Entropy-Regularized Trust Region Policy Optimization with Reinforcement Learning

    This is an extension of the standard Trust Region Policy Optimization (TRPO) algorithm
    that incorporates entropy regularization into the policy objective. The entropy bonus
    encourages exploration and prevents premature convergence to deterministic policies.

    Key Features:
    - Adds an entropy bonus term to the policy objective to promote exploration.
    - Retains the KL-divergence constraint from TRPO for stable policy updates.
    - The entropy coefficient (`ent_coef`) is tunable for balancing exploration and exploitation.
    - Suitable for environments with sparse rewards where additional exploration is needed.

    Mathematical Formulation:
    -------------------------
    Standard TRPO objective:
        L(θ) = E_t [ (π_θ(a_t | s_t) / π_θ_old(a_t | s_t)) * Â(s_t, a_t) ]

    TRPOR modified objective:
        L(θ) = E_t [ (π_θ(a_t | s_t) / π_θ_old(a_t | s_t)) * Â(s_t, a_t) + α * H(π_θ) ]

    where:
    - π_θ is the current policy.
    - π_θ_old is the old policy.
    - Â is the advantage function.
    - α (`ent_coef`) is the entropy coefficient.
    - H(π_θ) is the entropy of the policy.

    Parameters:
    -----------
    policy : Union[str, type[ActorCriticPolicy]]
        The policy model to be used (e.g., "MlpPolicy").
    env : Union[GymEnv, str]
        The environment to learn from.
    ent_coef : float, optional
        Entropy coefficient controlling the strength of the entropy bonus (default: 0.01).
    learning_rate : Union[float, Schedule], optional
        Learning rate for the optimizer (default: 1e-3).
    n_steps : int, optional
        Number of steps to run per update (default: 2048).
    batch_size : int, optional
        Minibatch size for the value function updates (default: 128).
    gamma : float, optional
        Discount factor for the reward (default: 0.99).
    cg_max_steps : int, optional
        Maximum steps for the conjugate gradient solver (default: 10).
    target_kl : float, optional
        Target KL divergence for policy updates (default: 0.01).

    Differences from Standard TRPO:
    -------------------------------
    - **Entropy Bonus:** Adds entropy to the policy objective for better exploration.
    - **Policy Objective:** Modified to include the entropy coefficient (`ent_coef`).
    - **Line Search:** Considers the entropy term while checking policy improvement.
    - **Logging:** Logs entropy-regularized objectives and KL divergence values.

    """

  def __init__(self, *args, epsilon=0.2, ent_coef=0.01, batch_size=32, **kwargs):
    super().__init__(*args, **kwargs)

    self.epsilon = epsilon
    self.ent_coef = ent_coef
    self.batch_size = batch_size

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

      # Check if the reward threshold is exceeded and activate the replay strategy
      distribution = self.policy.get_distribution(observations)

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
      # Entropy regularization on the advantage
      policy_objective = (advantages * ratio).mean() + self.ent_coef * distribution.entropy().mean()

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


def sample_trpor_params(trial, n_actions, n_envs, additional_args):
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

  # Strategy:
  # EnTRPO: replay all untill buffer clears
  # HIGH: replay when the entropy exceeeds a threshold
  # LOW: replay when the entropy is below a threshold

  epsilon = trial.suggest_float("epsilon", 0.1, 0.9, step=0.05)

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
      ortho_init=orthogonal_init,
    ),
    **additional_args,
  }
