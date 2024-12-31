import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, kl

from collections import namedtuple

from Common.GaussianPolicy import GaussianPolicy
from Specs import ModelSpec
from Train import Train

debug = False # Set to True to enable debug prints


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
    self.critic_learning_rate = params.get("critic_alpha", 1e-3)
    self.lambd = params.get("lambd", 0.95)
    self.gamma = params.get("gamma", 0.99)

    # Actor network
    self.actor = self.build_actor().to(self.device)

    # Critic network
    self.critic = self.build_critic().to(self.device)

    # Optimizer for critic
    self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_learning_rate)

    # Old actor for computing KL divergence
    self.old_actor = self.build_actor().to(self.device)
    self.old_actor.load_state_dict(self.actor.state_dict())

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

  def compute_advantages(self, rewards, state_values, next_state_values, dones, successes, gamma, lambd):
    """
    Compute advantages using Generalized Advantage Estimation (GAE), factoring in success flags.
    """
    advantages = []
    generalized_advantage_estimation = 0
    for time_step in reversed(range(len(rewards))):
      # Temporal difference error with success influence
      temporal_difference_error = (
          rewards[time_step]
          + self.gamma * next_state_values[time_step] * (1 - dones[time_step])
          - state_values[time_step]
      )

      # Amplify GAE for successes, dampen for failures
      generalized_advantage_estimation = (
          temporal_difference_error
          + self.gamma * self.lambd * (1 - dones[time_step]) * generalized_advantage_estimation
      ) * (1 + successes[time_step])  # Scale based on success
      advantages.insert(0, generalized_advantage_estimation)

    advantages_tensor = torch.tensor(advantages, dtype=torch.float32).to(self.device)
    if debug:
      print(f"Advantages: {advantages_tensor.mean().item()}")
    return advantages_tensor

  def compute_kl_divergence(self, states):
    # Compute KL divergence between the current actor and the old actor
    current_action_mean, current_action_std = self.actor(states)
    current_distribution = Normal(current_action_mean, current_action_std)

    old_action_mean, old_action_std = self.old_actor(states)
    old_distribution = Normal(old_action_mean, old_action_std)

    kl_div = kl.kl_divergence(old_distribution, current_distribution)
    return kl_div

  def train_critic(self, states, targets):
    if debug:
      print("Training critic...")
    self.critic_optimizer.zero_grad()
    predictions = self.critic(states).squeeze()

    loss = ((predictions - targets) ** 2).mean()
    self.critic_loss = loss.item()

    loss.backward()
    self.critic_optimizer.step()

  def apply_policy_step(self, full_step):
    if debug:
      print("Applying policy step...")
    index = 0
    for parameter in self.actor.parameters():
      number_of_elements = parameter.numel()
      parameter.data.add_(full_step[index: index + number_of_elements].view(parameter.size()))
      index += number_of_elements

  def compute_loss(self, states, actions, advantages):
    # Compute action mean and standard deviation using the current actor
    action_mean, action_standard_deviation = self.actor(states)
    action_distribution = Normal(action_mean, action_standard_deviation)
    log_probabilities = action_distribution.log_prob(actions).sum(dim=1)
    return -(log_probabilities * advantages).mean()

  def clone_actor(self):
    if debug:
      print("Cloning actor...")
    self.old_actor.load_state_dict(self.actor.state_dict())

  def line_search(self, states, actions, advantages, full_step, max_kl, max_backtracks=10, backtrack_coeff=0.5):
    old_loss = self.compute_loss(states, actions, advantages).detach().item()
    old_actor_state = {k: v.clone() for k, v in self.actor.state_dict().items()}

    for step_idx in range(max_backtracks):
      step_fraction = backtrack_coeff ** step_idx
      self.apply_policy_step(step_fraction * full_step)

      kl_div = self.compute_kl_divergence(states).mean().item()
      new_loss = self.compute_loss(states, actions, advantages).detach().item()
      improvement = old_loss - new_loss

      print(f"Line search iteration {step_idx+1}, step_fraction={step_fraction:.4f}, kl_div={kl_div:.4f}, improvement={improvement:.4f}")
      if kl_div < max_kl and improvement > 0:
        # Accept this step
        # <-- Update your final KL metric here
        self.kl_divergence_loss = kl_div
        print(f"Line search successful at iteration {step_idx+1}")
        return

      # Otherwise revert and try a smaller step
      self.actor.load_state_dict(old_actor_state)

    print("Line search failed. Reverting to old actor.")
    # If everything got reverted, final KL is effectively 0
    self.kl_divergence_loss = 0.0

  def train_actor(self, states, actions, advantages, max_kl_divergence=1e-2, damping=1e-2):
    if debug:
      print("Training actor...")
    loss = self.compute_loss(states, actions, advantages)
    self.actor_loss = loss.item()
    gradients = torch.autograd.grad(loss, self.actor.parameters())
    flattened_loss_gradient = torch.cat([gradient.view(-1) for gradient in gradients]).detach()

    def fisher_vector_product(vector):
      kl_divergence = self.compute_kl_divergence(states).mean()
      grads = torch.autograd.grad(kl_divergence, self.actor.parameters(), create_graph=True)
      flat_grads = torch.cat([g.view(-1) for g in grads])
      kl_v = (flat_grads * vector).sum()
      grads_2 = torch.autograd.grad(kl_v, self.actor.parameters())
      return torch.cat([g.view(-1) for g in grads_2]).detach() + damping * vector

    search_direction = self.conjugate_gradient_method(fisher_vector_product, -flattened_loss_gradient)
    shs = 0.5 * (search_direction * fisher_vector_product(search_direction)).sum()
    step_size = torch.sqrt(2.0 * max_kl_divergence / (shs + 1e-8))
    full_step = search_direction * step_size

    # Instead of applying full_step directly, do a backtracking line search
    self.line_search(states, actions, advantages, full_step, max_kl_divergence)

  def train(self, states, actions, rewards, next_states, dones, successes, **parameters):
    if debug:
      print("Training model...")
    # Prepare data
    states = states.to(self.device)
    next_states = next_states.to(self.device)
    actions = actions.to(self.device)
    rewards = rewards.to(self.device)
    dones = dones.to(self.device)
    successes = successes.to(self.device)

    # Compute values
    state_values = self.critic(states).squeeze()
    next_state_values = self.critic(next_states).squeeze()

    # Compute advantages
    advantages = self.compute_advantages(
        rewards, state_values.detach(), next_state_values.detach(),
        dones, successes, self.gamma, self.lambd
    )
    targets = rewards + self.gamma * next_state_values * (1 - dones)

    # Train critic
    self.train_critic(states, targets)

    # Train actor (old_actor was copied at the end of the *previous* iteration)
    self.train_actor(states, actions, advantages)

    # Now that actor is updated, clone it so next iteration sees a different policy in old_actor
    self.clone_actor()

    # Logging
    self.log_loss(self.actor_loss, self.critic_loss, self.kl_divergence_loss)


def load_model_if_available(model_folder, model_class, **model_params):
  """
  Load a model from a file if provided, or initialize a new model otherwise.
  """
  model_file = os.path.join(model_folder, "model.pth") if model_folder else None
  config_file = os.path.join(model_folder, "config.json") if model_folder else None

  if model_file and os.path.exists(model_file):
    print(f"Loading model from {model_file}")

    if config_file and os.path.exists(config_file):
      print(f"Loading model configuration from {config_file}")
      try:
        with open(config_file, "r") as f:
          loaded_params = json.load(f)
          model_params.update(loaded_params)  # Merge loaded params with existing ones
      except json.JSONDecodeError as e:
        print(f"Error reading configuration file: {e}")
        raise

    model = model_class(**model_params)
    model.load_state_dict(torch.load(model_file))
    return model

  else:
    print("No model file provided or file not found. Initializing a new model.")
    return model_class(**model_params)


if __name__ == "__main__":
  import gymnasium as gym
  from Experiment import Experiment
  from itertools import product

  # Initialize environment
  from Environments.LunarLander import make
  env = make()

  model_folder = sys.argv[1] if len(sys.argv) > 1 else None

  state_dim = env.observation_space.shape[0]
  action_dim = env.action_space.shape[0]

  # Define parameter grid
  param_grid = {
      "gamma": [0.99],
      "lambd": [0.95],
      "critic_alpha": [1e-3],
      "hidden_sizes": [[64, 64]],
      "kl_threshold": [1e-2],
      "state_dim": [state_dim],  # Fixed for the environment
      "action_dim": [action_dim],  # Fixed for the environment
  }

  # Generate all combinations of params
  param_combinations = list(product(*param_grid.values()))

  # Map parameter names to combinations
  param_keys = list(param_grid.keys())

  # Training params
  batch_size = 128
  num_batches = 100000

  for i, param_values in enumerate(param_combinations):
    # Create parameter dictionary for this combination
    model_params = dict(zip(param_keys, param_values))

    # Initialize model
    model = load_model_if_available(model_folder, TRPO, **model_params)

    # Define total timesteps for training
    total_timesteps = batch_size * num_batches

    print(f"Starting experiment {i + 1}/{len(param_combinations)} with params: {model_params}")

    # Train the model
    Train.batch(model, env, total_timesteps, batch_size, **model_params)

    # Evaluate the model
    Train.eval(model, env, episodes=len(model) // 5)

    # Save the experiment
    Experiment.save(model, env)

    print(f"Experiment {i + 1}/{len(param_combinations)} completed.")
