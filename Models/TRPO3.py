import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, kl
from collections import namedtuple

from Specs import ModelSpec
from Common.GaussianPolicy import GaussianPolicy
from Common.RolloutBuffer import RolloutBuffer


class TRPO3(ModelSpec):
  def __init__(self, device=None, **params):
    super(TRPO3, self).__init__(device=device, **params)
    self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.state_dim = params.get("state_dim")
    self.action_dim = params.get("action_dim")
    self.hidden_layer_sizes = params.get("hidden_sizes", [64, 64])
    self.gamma = params.get("gamma", 0.99)
    self.lambd = params.get("gae_lambda", 0.95)
    self.kl_threshold = params.get("kl_threshold", 1e-2)
    self.critic_alpha = params.get("critic_alpha", 1e-3)
    self.n_steps = params.get("n_steps", 2048)

    self.actor = self.build_actor().to(self.device)
    self.old_actor = self.build_actor().to(self.device)
    self.old_actor.load_state_dict(self.actor.state_dict())

    self.critic = self.build_critic().to(self.device)
    self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_alpha)

    self.rollout_buffer = RolloutBuffer(
        device=self.device
    )

    self.actor_loss = 0.0
    self.critic_loss = 0.0
    self.kl_divergence_loss = 0.0

  def build_actor(self):
    return GaussianPolicy(self.state_dim, self.hidden_layer_sizes, self.action_dim)

  def build_critic(self):
    layers = []
    input_dim = self.state_dim
    for size in self.hidden_layer_sizes:
      layers.append(nn.Linear(input_dim, size))
      layers.append(nn.Tanh())
      input_dim = size
    layers.append(nn.Linear(input_dim, 1))
    return nn.Sequential(*layers)

  def compute_advantages(self, rewards, state_values, next_state_values, dones, successes):
    advantages = []
    gae = 0
    for t in reversed(range(len(rewards))):
      delta = rewards[t] + self.gamma * next_state_values[t] * (1 - dones[t]) - state_values[t]
      gae = (delta + self.gamma * self.lambd * (1 - dones[t]) * gae) * (1 + successes[t])
      advantages.insert(0, gae)
    return torch.tensor(advantages, dtype=torch.float32, device=self.device)

  def select_action(self, state, deterministic=False):
    with torch.no_grad():
      state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
      mean, std = self.actor(state_tensor)
      dist = Normal(mean, std)
      if deterministic:
        return mean.squeeze(0).cpu().numpy()
      return dist.sample().squeeze(0).cpu().numpy()

  def compute_kl_divergence(self, states):
    mean, std = self.actor(states)
    dist_new = Normal(mean, std)
    old_mean, old_std = self.old_actor(states)
    dist_old = Normal(old_mean, old_std)
    return kl.kl_divergence(dist_old, dist_new)

  def compute_loss(self, states, actions, advantages):
    mean, std = self.actor(states)
    dist = Normal(mean, std)
    log_prob = dist.log_prob(actions).sum(dim=1)
    return -(log_prob * advantages).mean()

  @staticmethod
  def conjugate_gradient_method(fvp_func, b, n_steps=10, residual_tol=1e-10):
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    rr = torch.dot(r, r)
    for _ in range(n_steps):
      Ap = fvp_func(p)
      alpha = rr / torch.dot(p, Ap)
      x += alpha * p
      r -= alpha * Ap
      rr_new = torch.dot(r, r)
      if rr_new < residual_tol:
        break
      beta = rr_new / rr
      p = r + beta * p
      rr = rr_new
    return x

  def fisher_vector_product(self, states, vector, damping=1e-2):
    kl_div = self.compute_kl_divergence(states).mean()
    grads = torch.autograd.grad(kl_div, self.actor.parameters(), create_graph=True)
    flat_grads = torch.cat([g.view(-1) for g in grads])
    product = (flat_grads * vector).sum()
    grads2 = torch.autograd.grad(product, self.actor.parameters())
    flat_grads2 = torch.cat([g.contiguous().view(-1) for g in grads2])
    return flat_grads2 + damping * vector

  def apply_policy_step(self, step):
    idx = 0
    for param in self.actor.parameters():
      size = param.numel()
      param.data.add_(step[idx: idx + size].view(param.size()))
      idx += size

  def clone_actor(self):
    self.old_actor.load_state_dict(self.actor.state_dict())

  def line_search(self, states, actions, advantages, full_step, max_kl=1e-2, backtrack_coeff=0.5, max_backtracks=10):
    old_loss = self.compute_loss(states, actions, advantages).detach()
    old_actor_state = {k: v.clone() for k, v in self.actor.state_dict().items()}
    for _ in range(max_backtracks):
      self.apply_policy_step(full_step)
      kl_div_val = self.compute_kl_divergence(states).mean().item()
      new_loss = self.compute_loss(states, actions, advantages).detach().item()
      improvement = old_loss - new_loss
      if kl_div_val < max_kl and improvement > 0:
        self.kl_divergence_loss = kl_div_val
        return
      self.actor.load_state_dict(old_actor_state)
      full_step *= backtrack_coeff
    self.kl_divergence_loss = 0.0

  def train_actor(self, states, actions, advantages, max_kl=1e-2, damping=1e-2):
    loss = self.compute_loss(states, actions, advantages)
    self.actor_loss = loss.item()
    grads = torch.autograd.grad(loss, self.actor.parameters())
    flat_grad = torch.cat([g.view(-1) for g in grads]).detach()

    def fvp_func(vec):
      return self.fisher_vector_product(states, vec, damping=damping)

    search_dir = self.conjugate_gradient_method(fvp_func, -flat_grad)
    shs = 0.5 * (search_dir * fvp_func(search_dir)).sum()
    step_size = torch.sqrt(2.0 * max_kl / (shs + 1e-8))
    full_step = search_dir * step_size
    self.line_search(states, actions, advantages, full_step, max_kl=max_kl)

  def train_critic(self, states, targets):
    self.critic_optimizer.zero_grad()
    preds = self.critic(states).squeeze()
    loss = nn.MSELoss()(preds, targets)
    self.critic_loss = loss.item()
    loss.backward()
    self.critic_optimizer.step()

  def train(self, states, actions, rewards, next_states, dones, successes, **kwargs):
    for i in range(len(states)):
      self.rollout_buffer.add(
          state=states[i],
          action=actions[i],
          reward=rewards[i],
          next_state=next_states[i],
          done=dones[i],
          success=successes[i]
      )
    if len(self.rollout_buffer) >= self.n_steps:
      values = self.rollout_buffer.get()
      states = values["states"]
      actions = values["actions"]
      rewards = values["rewards"]
      next_states = values["next_states"]
      dones = values["dones"]
      successes = values["successes"]

      with torch.no_grad():
        state_values = self.critic(states).squeeze()
        next_state_values = self.critic(next_states).squeeze()

      advantages = self.compute_advantages(
          rewards=rewards.cpu().numpy(),
          state_values=state_values.cpu().numpy(),
          next_state_values=next_state_values.cpu().numpy(),
          dones=dones.cpu().numpy(),
          successes=successes.cpu().numpy()
      )

      targets = rewards + self.gamma * next_state_values * (1 - dones.float())
      self.train_critic(states, targets)
      self.train_actor(states, actions, advantages)
      self.clone_actor()
      self.log_loss(self.actor_loss, self.critic_loss, self.kl_divergence_loss)


if __name__ == "__main__":
  import gymnasium as gym
  from itertools import product

  from Train import Train
  from Experiment import Experiment

  # Initialize environment
  from Environments.Pendulum import make
  env = make()
  state_dim = env.observation_space.shape[0]
  action_dim = env.action_space.shape[0]

  # Define parameter grid
  param_grid = {
      "gamma": [0.95, 0.99],
      "lambd": [0.9, 0.95],
      "critic_alpha": [1e-3, 1e-4],
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
  num_batch = 1000

  for i, param_values in enumerate(param_combinations):
    # Create parameter dictionary for this combination
    model_params = dict(zip(param_keys, param_values))

    # Initialize model
    model = TRPO3(**model_params)

    # Define total timesteps for training
    total_timesteps = batch_size * num_batch

    print(f"Starting experiment {i + 1}/{len(param_combinations)} with params: {model_params}")

    # Train the model
    Train.batch(model, env, total_timesteps, batch_size, **model_params)

    # Save the experiment
    Experiment.save(model, env)

    print(f"Experiment {i + 1}/{len(param_combinations)} completed.")
