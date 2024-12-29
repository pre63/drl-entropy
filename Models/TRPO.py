import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, kl
from collections import namedtuple

from Common.GaussianPolicy import GaussianPolicy
from Specs import ModelSpec
from Train import Train


class TRPO(ModelSpec):
  """
      Trust Region Policy Optimization (TRPO) model.
  """

  def __init__(self, device=None, **params):
    super(TRPO, self).__init__(device=device, **params)

    self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.state_dim = params.get("state_dim")
    self.action_dim = params.get("action_dim")
    self.hidden_layer_sizes = params.get("hidden_sizes", [64, 64])
    self.kl_divergence_threshold = params.get("kl_threshold", 1e-2)
    self.critic_learning_rate = params.get("critic_lr", 1e-3)

    # Actor network
    self.actor = self.build_actor().to(self.device)

    # Critic network
    self.critic = self.build_critic().to(self.device)

    # Optimizer for critic
    self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_learning_rate)

  def build_actor(self):
    return GaussianPolicy(self.state_dim, self.hidden_layer_sizes, self.action_dim)

  def build_critic(self):
    layers = []
    input_dim = self.state_dim
    for layer_size in self.hidden_layer_sizes:
      layers.append(nn.Linear(input_dim, layer_size))
      layers.append(nn.Tanh())
      input_dim = layer_size
    layers.append(nn.Linear(input_dim, 1))
    return nn.Sequential(*layers)

  def select_action(self, state):
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
    with torch.no_grad():
      mean, std = self.actor(state_tensor)
      distribution = Normal(mean, std)
      selected_action = distribution.sample()
    return selected_action.cpu().numpy()[0]

  def compute_advantages(self, rewards, state_values, next_state_values, dones, successes, discount_factor, lambda_value):
    """
    Compute advantages using Generalized Advantage Estimation (GAE), factoring in success flags.
    """
    advantages = []
    generalized_advantage_estimation = 0
    for time_step in reversed(range(len(rewards))):
      # Temporal difference error with success influence
      temporal_difference_error = (
          rewards[time_step]
          + discount_factor * next_state_values[time_step] * (1 - dones[time_step])
          - state_values[time_step]
      )
      # Amplify GAE for successes, dampen for failures
      generalized_advantage_estimation = (
          temporal_difference_error
          + discount_factor * lambda_value * (1 - dones[time_step]) * generalized_advantage_estimation
      ) * (1 + successes[time_step])  # Scale based on success
      advantages.insert(0, generalized_advantage_estimation)
    return torch.tensor(advantages, dtype=torch.float32).to(self.device)

  def compute_kl_divergence(self, states):
    # Compute KL divergence between the current actor and the old actor
    current_action_mean, current_action_std = self.actor(states)
    current_distribution = Normal(current_action_mean, current_action_std)

    old_action_mean, old_action_std = self.old_actor(states)
    old_distribution = Normal(old_action_mean, old_action_std)

    return kl.kl_divergence(current_distribution, old_distribution)

  def train_critic(self, states, targets):
    self.critic_optimizer.zero_grad()
    predictions = self.critic(states).squeeze()

    loss = ((predictions - targets) ** 2).mean()
    self.critic_loss = loss.item()

    loss.backward()
    self.critic_optimizer.step()

  def apply_policy_step(self, full_step):
    index = 0
    for parameter in self.actor.parameters():
      number_of_elements = parameter.numel()
      parameter.data.add_(full_step[index: index + number_of_elements].view(parameter.size()))
      index += number_of_elements

  @staticmethod
  def conjugate_gradient_method(vector_product_function, vector_b, number_of_steps=10, residual_tolerance=1e-10):
    solution_vector = torch.zeros_like(vector_b)
    residual = vector_b.clone()
    direction = vector_b.clone()
    residual_dot_residual = torch.dot(residual, residual)
    for _ in range(number_of_steps):
      vector_product = vector_product_function(direction)
      alpha = residual_dot_residual / torch.dot(direction, vector_product)
      solution_vector += alpha * direction
      residual -= alpha * vector_product
      new_residual_dot_residual = torch.dot(residual, residual)
      if new_residual_dot_residual < residual_tolerance:
        break
      beta = new_residual_dot_residual / residual_dot_residual
      direction = residual + beta * direction
      residual_dot_residual = new_residual_dot_residual
    return solution_vector

  def compute_loss(self, states, actions, advantages):
      # Compute action mean and standard deviation using the current actor
    action_mean, action_standard_deviation = self.actor(states)
    action_distribution = Normal(action_mean, action_standard_deviation)
    log_probabilities = action_distribution.log_prob(actions).sum(dim=1)
    return -(log_probabilities * advantages).mean()

  def train_actor(self, states, actions, advantages, max_kl_divergence=1e-2, damping=1e-2):
    """
    Train the actor network using the trust region policy optimization step.
    """
    # Compute gradients of the loss
    loss = self.compute_loss(states, actions, advantages)
    self.actor_loss = loss.item()
    gradients = torch.autograd.grad(loss, self.actor.parameters())
    flattened_loss_gradient = torch.cat([gradient.view(-1) for gradient in gradients]).detach()

    def fisher_vector_product(vector):
      kl_divergence = self.compute_kl_divergence(states).mean()
      self.kl_divergence_loss = kl_divergence.item()

      gradients = torch.autograd.grad(kl_divergence, self.actor.parameters(), create_graph=True)
      flattened_gradients = torch.cat([gradient.view(-1) for gradient in gradients])
      kl_vector = (flattened_gradients * vector).sum()
      second_gradients = torch.autograd.grad(kl_vector, self.actor.parameters())
      return torch.cat([gradient.view(-1) for gradient in second_gradients]).detach() + damping * vector

    # Conjugate gradient to find the search direction
    search_direction = self.conjugate_gradient_method(fisher_vector_product, -flattened_loss_gradient)
    shs = 0.5 * (search_direction * fisher_vector_product(search_direction)).sum(0, keepdim=True)
    lagrange_multiplier = torch.sqrt(shs / max_kl_divergence)
    full_step = search_direction / lagrange_multiplier[0]

    # Apply the update step
    self.apply_policy_step(full_step)

  def train(self, states, actions, rewards, next_states, dones, successes, **parameters):
    discount_factor = parameters.get("gamma", 0.99)
    lambda_value = parameters.get("lam", 0.95)

    states = states.to(self.device)
    next_states = next_states.to(self.device)
    actions = actions.to(self.device)
    rewards = rewards.to(self.device)
    dones = dones.to(self.device)
    successes = successes.to(self.device)

    state_values = self.critic(states).squeeze()
    next_state_values = self.critic(next_states).squeeze()

    advantages = self.compute_advantages(rewards, state_values.detach(), next_state_values.detach(), dones, successes, discount_factor, lambda_value)
    targets = rewards + discount_factor * next_state_values * (1 - dones)

    self.train_critic(states, targets)

    # Create and copy the current actor to old_actor
    self.clone_actor()

    # Train the actor
    self.train_actor(states, actions, advantages)

    self.log_loss(self.actor_loss, self.critic_loss, self.kl_divergence_loss)

  def clone_actor(self):
    self.old_actor = GaussianPolicy(
        input_dim=self.state_dim,
        hidden_layer_sizes=self.hidden_layer_sizes,
        output_dim=self.action_dim
    ).to(self.device)
    self.old_actor.load_state_dict(self.actor.state_dict())


if __name__ == "__main__":
  import gymnasium as gym
  from Experiment import Experiment
  from itertools import product

  # Initialize environment
  from Environments.LunarLander import make
  env = make()

  state_dim = env.observation_space.shape[0]
  action_dim = env.action_space.shape[0]

  # Define parameter grid
  param_grid = {
      "gamma": [0.95, 0.99],
      "lam": [0.5, 0.9, 0.95],
      "critic_lr": [1e-3, 1e-4],
      "hidden_sizes": [[64, 64], [128, 128]],
      "kl_threshold": [1e-2, 5e-3],
      "state_dim": [state_dim],  # Fixed for the environment
      "action_dim": [action_dim],  # Fixed for the environment
  }

  # Generate all combinations of params
  param_combinations = list(product(*param_grid.values()))

  # Map parameter names to combinations
  param_keys = list(param_grid.keys())

  # Training params
  batch_size = 128
  episodes_per_batch = 10
  factor = 10

  for i, param_values in enumerate(param_combinations):
    # Create parameter dictionary for this combination
    model_params = dict(zip(param_keys, param_values))

    # Initialize model
    model = TRPO(**model_params)

    # Define total timesteps for training
    total_timesteps = batch_size * episodes_per_batch * factor

    print(f"Starting experiment {i + 1}/{len(param_combinations)} with params: {model_params}")

    # Train the model
    Train.batch(model, env, total_timesteps, batch_size, **model_params)

    # Save the experiment
    Experiment.save(model, env)

    print(f"Experiment {i + 1}/{len(param_combinations)} completed.")
