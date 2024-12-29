import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, kl
from collections import namedtuple

from Specs import ModelSpec
from Common.GaussianPolicy import GaussianPolicy
from Common.RolloutBuffer import RolloutBuffer


class TRPO3(ModelSpec):
  """
    TRPO3 implementation based on Stable Baselines 3 Contrib
  """

  def __init__(self, device=None, **params):
    super(TRPO3, self).__init__(device=device, **params)
    self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.state_dim = params["state_dim"]
    self.action_dim = params["action_dim"]
    self.hidden_layer_sizes = params.get("hidden_sizes", [64, 64])
    self.kl_divergence_threshold = params.get("kl_threshold", 0.01)
    self.critic_learning_rate = params.get("critic_lr", 1e-3)
    self.n_critic_updates = params.get("n_critic_updates", 10)
    self.damping = params.get("damping", 1e-2)

    # Actor and Critic networks
    self.actor = self.build_actor().to(self.device)
    self.critic = self.build_critic().to(self.device)

    # Optimizer for critic
    self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_learning_rate)

    # Rollout Buffer
    self.rollout_buffer = RolloutBuffer(
        buffer_size=params.get("buffer_size", 2048),
        state_dim=self.state_dim,
        action_dim=self.action_dim,
        device=self.device,
    )

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
      action = distribution.sample()
    return action.cpu().numpy()[0]

  def compute_advantages(self, rewards, values, next_values, dones, gamma, lam):
    advantages = []
    generalized_advantage_estimation = 0
    for t in reversed(range(len(rewards))):
      delta = rewards[t] + gamma * next_values[t] * (1 - dones[t]) - values[t]
      generalized_advantage_estimation = delta + gamma * lam * (1 - dones[t]) * generalized_advantage_estimation
      advantages.insert(0, generalized_advantage_estimation)
    return torch.tensor(advantages, dtype=torch.float32).to(self.device)

  def normalize_advantages(self, advantages):
    return (advantages - advantages.mean()) / torch.clamp(advantages.std(unbiased=False), min=1e-8)

  def hessian_vector_product(self, states, vector):
    kl = self.compute_kl_divergence(states).mean()
    self.kl_divergence_loss = kl.item()

    gradients = torch.autograd.grad(kl, self.actor.parameters(), create_graph=True)
    flat_grad = torch.cat([grad.view(-1) for grad in gradients])
    kl_v = (flat_grad * vector).sum()
    grads = torch.autograd.grad(kl_v, self.actor.parameters())
    flat_grads = torch.cat([grad.view(-1) for grad in grads]).detach()
    return flat_grads + self.damping * vector

  def conjugate_gradient(self, fisher_vector_product_fn, b, max_iterations=10, residual_tol=1e-10):
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)
    for _ in range(max_iterations):
      Avp = fisher_vector_product_fn(p)
      alpha = rdotr / torch.dot(p, Avp)
      x += alpha * p
      r -= alpha * Avp
      new_rdotr = torch.dot(r, r)
      if new_rdotr < residual_tol:
        break
      beta = new_rdotr / rdotr
      p = r + beta * p
      rdotr = new_rdotr
    return x

  def compute_kl_divergence(self, states):
    # Compute KL divergence between the current actor and the old actor
    current_action_mean, current_action_std = self.actor(states)
    current_distribution = Normal(current_action_mean, current_action_std)

    old_action_mean, old_action_std = self.old_actor(states)
    old_distribution = Normal(old_action_mean, old_action_std)

    return kl.kl_divergence(current_distribution, old_distribution)

  def clone_actor(self):
    self.old_actor = GaussianPolicy(
        input_dim=self.state_dim,
        hidden_layer_sizes=self.hidden_layer_sizes,
        output_dim=self.action_dim
    ).to(self.device)
    self.old_actor.load_state_dict(self.actor.state_dict())

  def train_critic(self, states, targets):
    self.critic_optimizer.zero_grad()
    predictions = self.critic(states).squeeze()
    targets = targets.detach()  # Detach the targets to avoid graph dependencies
    loss = ((predictions - targets) ** 2).mean()
    self.critic_loss = loss.item()
    loss.backward()
    self.critic_optimizer.step()

  def train_actor(self, states, actions, advantages):
    self.clone_actor()
    loss = self.compute_loss(states, actions, advantages)
    self.actor_loss = loss.item()

    gradients = torch.autograd.grad(loss, self.actor.parameters())
    flat_grad = torch.cat([grad.view(-1) for grad in gradients]).detach()

    search_direction = self.conjugate_gradient(
        lambda v: self.hessian_vector_product(states, v), -flat_grad
    )
    shs = 0.5 * (search_direction * self.hessian_vector_product(states, search_direction)).sum(0)
    step_size = torch.sqrt(2 * self.kl_divergence_threshold / shs)
    self.apply_policy_step(search_direction * step_size)

  def apply_policy_step(self, full_step):
    index = 0
    for parameter in self.actor.parameters():
      number_of_elements = parameter.numel()
      parameter.data.add_(full_step[index: index + number_of_elements].view(parameter.size()))
      index += number_of_elements

  def compute_loss(self, states, actions, advantages):
    mean, std = self.actor(states)
    distribution = Normal(mean, std)
    log_probs = distribution.log_prob(actions).sum(dim=1)
    return -(log_probs * advantages).mean()

  def train(self, states, actions, rewards, next_states, dones, **params):
    gamma = params.get("gamma", 0.99)
    lambda_value = params.get("lam", 0.95)

    # Move tensors to device
    states = states.to(self.device)
    next_states = next_states.to(self.device)
    actions = actions.to(self.device)
    rewards = rewards.to(self.device)
    dones = dones.to(self.device)

    # Compute values
    state_values = self.critic(states).squeeze()
    next_state_values = self.critic(next_states).squeeze()

    # Compute advantages and targets
    advantages = self.compute_advantages(rewards, state_values.detach(), next_state_values.detach(), dones, gamma, lambda_value)
    advantages = advantages.detach()  # Ensure it's detached
    targets = rewards + gamma * next_state_values * (1 - dones.float())

    # Train critic
    self.train_critic(states, targets)

    # Clone the actor network for KL divergence computation
    self.clone_actor()

    # Train actor
    self.train_actor(states, actions, advantages)

    self.log_loss(self.actor_loss, self.critic_loss, self.kl_divergence_loss)


if __name__ == "__main__":
  import gymnasium as gym
  from itertools import product

  from Train import Train
  from Experiment import Experiment

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
  factor = 1000

  for i, param_values in enumerate(param_combinations):
    # Create parameter dictionary for this combination
    model_params = dict(zip(param_keys, param_values))

    # Initialize model
    model = TRPO3(**model_params)

    # Define total timesteps for training
    total_timesteps = batch_size * episodes_per_batch * factor

    print(f"Starting experiment {i + 1}/{len(param_combinations)} with params: {model_params}")

    # Train the model
    Train.batch(model, env, total_timesteps, batch_size, **model_params)

    # Save the experiment
    Experiment.save(model, env)

    print(f"Experiment {i + 1}/{len(param_combinations)} completed.")
