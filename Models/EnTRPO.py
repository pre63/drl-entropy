import torch
from torch.distributions import Normal

from Models.TRPO import TRPO
from Common.GaussianPolicy import GaussianPolicy


class EnTRPO(TRPO):
  """
      EnTRPO: TRPO with entropy regularization for exploration.
  """

  def __init__(self, device=None, **params):
    super(EnTRPO, self).__init__(device=device, **params)
    self.entropy_coeff = params.get("entropy_coeff", 0.01)  # Coefficient for entropy regularization

  def compute_entropy(self, states):
    """
    Compute entropy of the current policy for the given states.
    Entropy encourages exploration by regularizing the policy.
    """
    action_mean, action_std = self.actor(states)
    action_distribution = Normal(action_mean, action_std)
    entropy = action_distribution.entropy()
    return entropy.sum(dim=-1)  # Sum entropy over action dimensions

  def train_actor(self, states, actions, advantages):
    """
    Modified train_actor to include entropy regularization.
    """
    # Compute log probabilities
    action_mean, action_std = self.actor(states)
    action_distribution = Normal(action_mean, action_std)
    log_probabilities = action_distribution.log_prob(actions).sum(dim=-1)

    # Compute entropy for the current policy
    entropy = self.compute_entropy(states)

    # Compute surrogate loss with entropy regularization
    surrogate_loss = -(log_probabilities * advantages).mean()
    entropy_regularization = -self.entropy_coeff * entropy.mean()
    total_loss = surrogate_loss + entropy_regularization
    self.actor_loss = total_loss.item()

    # Compute gradients and apply updates
    gradients = torch.autograd.grad(total_loss, self.actor.parameters())
    flattened_loss_gradient = torch.cat([gradient.view(-1) for gradient in gradients]).detach()

    def fisher_vector_product(vector):
      kl_divergence = self.compute_kl_divergence(states).mean()
      self.kl_divergence_loss = kl_divergence.item()

      gradients = torch.autograd.grad(kl_divergence, self.actor.parameters(), create_graph=True)
      flattened_gradients = torch.cat([gradient.view(-1) for gradient in gradients])
      kl_vector = (flattened_gradients * vector).sum()
      second_gradients = torch.autograd.grad(kl_vector, self.actor.parameters())
      return torch.cat([gradient.view(-1) for gradient in second_gradients]).detach() + 1e-2 * vector

    # Conjugate gradient to find the search direction
    search_direction = self.conjugate_gradient_method(fisher_vector_product, -flattened_loss_gradient)
    shs = 0.5 * (search_direction * fisher_vector_product(search_direction)).sum(0, keepdim=True)
    lagrange_multiplier = torch.sqrt(shs / self.kl_divergence_threshold)
    full_step = search_direction / lagrange_multiplier[0]

    # Apply the policy update step
    self.apply_policy_step(full_step)

  def train(self, states, actions, rewards, next_states, dones, successes, **parameters):
    gamma = parameters.get("gamma", 0.99)
    lambda_value = parameters.get("lam", 0.95)

    states = states.to(self.device)
    next_states = next_states.to(self.device)
    actions = actions.to(self.device)
    rewards = rewards.to(self.device)
    dones = dones.to(self.device)
    successes = successes.to(self.device)

    # Compute values and advantages
    state_values = self.critic(states).squeeze()
    next_state_values = self.critic(next_states).squeeze()

    advantages = self.compute_advantages(rewards, state_values.detach(), next_state_values.detach(), dones, successes, gamma, lambda_value)
    targets = rewards + gamma * next_state_values * (1 - dones)

    # Train critic
    self.train_critic(states, targets)

    # Clone the current actor to old_actor
    self.clone_actor()

    # Train actor using the cloned old actor
    self.train_actor(states, actions, advantages)

    self.log_loss(self.actor_loss, self.critic_loss, self.kl_divergence_loss)


if __name__ == "__main__":
  import gymnasium as gym
  from Experiment import Experiment
  from Train import Train
  from itertools import product

  # Initialize environment
  from Environments.Pendulum import make_pendulum
  env = make_pendulum()
  state_dim = env.observation_space.shape[0]
  action_dim = env.action_space.shape[0]

  # Define parameter grid
  param_grid = {
      "gamma": [0.95, 0.99],
      "lam": [0.9, 0.95],
      "critic_lr": [1e-3, 1e-4],
      "hidden_sizes": [[64, 64], [128, 128]],
      "kl_threshold": [1e-2, 5e-3],
      "entropy_coeff": [0.01, 0.001],
      "state_dim": [state_dim],  # Fixed for the environment
      "action_dim": [action_dim],  # Fixed for the environment
  }

  # Generate all combinations of parameters
  param_combinations = list(product(*param_grid.values()))

  # Map parameter names to combinations
  param_keys = list(param_grid.keys())

  # Training parameters
  batch_size = 128
  episodes_per_batch = 10
  factor = 100

  for i, param_values in enumerate(param_combinations):
    # Create parameter dictionary for this combination
    model_params = dict(zip(param_keys, param_values))

    # Initialize model
    model = EnTRPO(**model_params)

    # Define total timesteps for training
    total_timesteps = batch_size * episodes_per_batch * factor

    print(f"Starting experiment {i + 1}/{len(param_combinations)} with params: {model_params}")

    # Train the model
    Train.batch(model, env, total_timesteps, batch_size, **model_params)

    # Save the experiment
    Experiment.save(model)

    print(f"Experiment {i + 1}/{len(param_combinations)} completed.")
