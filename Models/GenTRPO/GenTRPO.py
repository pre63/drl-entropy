import random

import numpy as np
import torch as th
from sb3_contrib.common.utils import conjugate_gradient_solver, flat_grad
from sb3_contrib.trpo.policies import CnnPolicy, MlpPolicy, MultiInputPolicy
from stable_baselines3.common.distributions import kl_divergence
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, RolloutBufferSamples, Schedule
from stable_baselines3.common.utils import explained_variance
from torch import nn
from torch.nn import functional as F

from Models.SB3 import TRPO


class ForwardDynamicsModel(nn.Module):
  def __init__(self, observation_space, action_space, hidden_dim=128):
    super().__init__()

    self.state_dim = np.prod(observation_space.shape)  # Compute total state dimension
    self.action_dim = action_space.shape[0]  # Continuous action space

    self.encoder = nn.Sequential(
        nn.Linear(self.state_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
    )

    self.forward_model = nn.Sequential(
        nn.Linear(hidden_dim + self.action_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
    )

  def forward(self, state, action):
    state = state.view(state.size(0), -1)  # Ensure correct shape
    h_s = self.encoder(state)
    action = action.view(action.size(0), -1)  # Ensure action is 2D
    x = th.cat([h_s, action], dim=-1)
    pred_h_next = self.forward_model(x)
    return h_s, pred_h_next


class GenerativeReplayBuffer:
  def __init__(self, real_capacity, synthetic_capacity, relevance_function, generative_model, batch_size):
    self.real_capacity = real_capacity
    self.synthetic_capacity = synthetic_capacity
    self.real_buffer = []
    self.synthetic_buffer = []
    self.relevance_function = relevance_function
    self.generative_model = generative_model
    self.batch_size = batch_size

  def add_real(self, transition):
    state, action, reward, next_state, done = transition

    # Ensure state and action are tensors before storing
    if not isinstance(state, th.Tensor):
      state = th.as_tensor(state, dtype=th.float32)
    if not isinstance(action, th.Tensor):
      action = th.as_tensor(action, dtype=th.float32)

    self.real_buffer.append((state, action, reward, next_state, done))
    if len(self.real_buffer) > self.real_capacity:
      self.real_buffer.pop(0)

  def generate_synthetic(self):
    if len(self.real_buffer) == 0:
      return

    # Compute relevance scores and sort
    scored_samples = sorted(self.real_buffer, key=self.relevance_function, reverse=True)

    # Select the top 10 most relevant transitions
    sampled_real = scored_samples[:10]

    synthetic_transitions = []
    for state, action, reward, next_state, done in sampled_real:
      with th.no_grad():
        action_distribution = self.generative_model.get_distribution(state)
        synthetic_action = action_distribution.sample()  # Sample new action
        synthetic_transitions.append((state, synthetic_action, reward, next_state, done))

    # Store synthetic samples
    self.synthetic_buffer.extend(synthetic_transitions)

    # Trim synthetic buffer if over capacity
    self.synthetic_buffer = self.synthetic_buffer[-self.synthetic_capacity:]

  def sample(self, num_samples):
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
    self.batch_size = batch_size
    self.entropy_coeff = entropy_coeff

    self.forward_dynamics_model = ForwardDynamicsModel(
        observation_space=self.observation_space,
        action_space=self.action_space
    ).to(self.device)

    self.replay_buffer = GenerativeReplayBuffer(
        real_capacity=buffer_capacity,
        synthetic_capacity=buffer_capacity,
        relevance_function=self._compute_relevance,
        generative_model=self.policy,
        batch_size=batch_size,
    )

  def _compute_relevance(self, transition):
    state, action, _, _, _ = transition

    with th.no_grad():
      h_s, pred_h_next = self.forward_dynamics_model(state, action)

    curiosity_score = 0.5 * th.norm(pred_h_next - h_s, p=2).item() ** 2
    return curiosity_score

  def train(self) -> None:
    self.policy.set_training_mode(True)
    self._update_learning_rate(self.policy.optimizer)

    policy_objective_values = []
    kl_divergences = []
    value_losses = []

    for rollout_data in self.rollout_buffer.get():
      observations = rollout_data.observations
      actions = rollout_data.actions
      returns = rollout_data.returns
      advantages = rollout_data.advantages
      old_log_prob = rollout_data.old_log_prob

      transition = (
          observations.cpu().numpy(),
          actions.cpu().numpy(),
          returns.cpu().numpy(),
          advantages.cpu().numpy(),
          old_log_prob.cpu().numpy(),
      )
      self.replay_buffer.add_real(transition)

    self.replay_buffer.generate_synthetic()

    old_distribution = self.policy.get_distribution(observations)
    entropy_mean = old_distribution.entropy().mean()
    avg_entropy = entropy_mean.item()

    num_replay_samples = compute_sampling_parameters_gradual_linear(
        entropy=avg_entropy, entropy_coeff=self.entropy_coeff, min_samples=0, max_samples=self.batch_size
    )

    if num_replay_samples > 0:
      replay_samples = self.replay_buffer.sample(num_replay_samples)
      replay_obs, replay_actions, replay_returns, replay_advantages, replay_old_log_prob = zip(*replay_samples)

      # Convert lists of tensors into a single tensor
      replay_obs = th.cat(replay_obs).to(self.device)
      replay_actions = th.cat(replay_actions).to(self.device)
      replay_returns = th.cat(replay_returns).to(self.device)
      replay_advantages = th.cat(replay_advantages).to(self.device)
      replay_old_log_prob = th.cat(replay_old_log_prob).to(self.device)

      # Concatenate replay samples with on-policy samples
      observations = th.cat([observations, replay_obs])
      actions = th.cat([actions, replay_actions])
      returns = th.cat([returns, replay_returns])
      advantages = th.cat([advantages, replay_advantages])
      old_log_prob = th.cat([old_log_prob, replay_old_log_prob])

    if self.normalize_advantage:
      advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    new_distribution = self.policy.get_distribution(observations)
    log_prob = new_distribution.log_prob(actions)
    ratio = th.exp(log_prob - old_log_prob)
    policy_objective = (advantages * ratio).mean() + self.entropy_coeff * new_distribution.entropy().mean()
    kl_div = kl_divergence(old_distribution, new_distribution).mean()

    self.policy.optimizer.zero_grad()
    policy_objective.backward()
    self.policy.optimizer.step()

    self.logger.record("train/policy_objective", np.mean(policy_objective_values))
    self.logger.record("train/value_loss", np.mean(value_losses))
    self.logger.record("train/kl_divergence", np.mean(kl_divergences))


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
