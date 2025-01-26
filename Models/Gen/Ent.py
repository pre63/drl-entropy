import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque


class ReplayBuffer:
  def __init__(self, capacity):
    self.capacity = capacity
    self.buffer = deque(maxlen=capacity)

  def push(self, s, a, r, s_prime, d):
    self.buffer.append((s, a, r, s_prime, d))

  def sample(self, batch_size):
    batch = random.sample(self.buffer, batch_size)
    s, a, r, s_prime, d = zip(*batch)
    return (torch.FloatTensor(s), torch.FloatTensor(a), torch.FloatTensor(r), torch.FloatTensor(s_prime), torch.FloatTensor(d))

  def __len__(self):
    return len(self.buffer)


class FeatureEncoder(nn.Module):
  def __init__(self, state_dim, latent_dim):
    super().__init__()
    self.fc = nn.Sequential(nn.Linear(state_dim, 64), nn.ReLU(), nn.Linear(64, latent_dim))

  def forward(self, x):
    return self.fc(x)


class ForwardDynamicsModel(nn.Module):
  def __init__(self, latent_dim, action_dim):
    super().__init__()
    self.fc = nn.Sequential(nn.Linear(latent_dim + action_dim, 64), nn.ReLU(), nn.Linear(64, latent_dim))

  def forward(self, latent, action):
    x = torch.cat([latent, action], dim=-1)
    return self.fc(x)


class CuriosityModel:
  def __init__(self, state_dim, action_dim, latent_dim, lr=1e-3):
    self.encoder = FeatureEncoder(state_dim, latent_dim)
    self.fwd = ForwardDynamicsModel(latent_dim, action_dim)
    self.opt = optim.Adam(list(self.encoder.parameters()) + list(self.fwd.parameters()), lr=lr)
    self.latent_dim = latent_dim

  def update(self, states, actions, next_states):
    states_enc = self.encoder(states)
    next_states_enc = self.encoder(next_states)
    pred_next_enc = self.fwd(states_enc, actions)
    loss = F.mse_loss(pred_next_enc, next_states_enc)
    self.opt.zero_grad()
    loss.backward()
    self.opt.step()
    return loss.item()

  def compute_relevance(self, states, actions, next_states):
    with torch.no_grad():
      states_enc = self.encoder(states)
      next_states_enc = self.encoder(next_states)
      pred_next_enc = self.fwd(states_enc, actions)
      errors = 0.5 * ((pred_next_enc - next_states_enc) ** 2).sum(dim=1)
    return errors.cpu().numpy()


class DiffusionModel(nn.Module):
  def __init__(self, state_dim, action_dim, latent_dim=64, timesteps=4):
    super().__init__()
    self.state_dim = state_dim
    self.action_dim = action_dim
    self.timesteps = timesteps
    self.net = nn.Sequential(nn.Linear(state_dim + action_dim + 1, 128), nn.ReLU(), nn.Linear(128, state_dim))

  def forward(self, x, cond):
    return self.net(torch.cat([x, cond], dim=-1))

  def sample(self, cond, steps=10):
    x = torch.randn(cond.size(0), self.state_dim).to(cond.device)
    for _ in range(steps):
      x = x - self.forward(x, cond)
    return x


class GenerativeModel:
  def __init__(self, state_dim, action_dim, lr=1e-3):
    self.model = DiffusionModel(state_dim, action_dim)
    self.opt = optim.Adam(self.model.parameters(), lr=lr)

  def train_model(self, states, actions, next_states, relevance):
    relevance = relevance.unsqueeze(-1)
    cond = torch.cat([actions, relevance], dim=-1)
    x = self.model.forward(states, cond)
    loss = F.mse_loss(x, next_states)
    self.opt.zero_grad()
    loss.backward()
    self.opt.step()
    return loss.item()

  def generate(self, actions, relevance, num_samples):
    cond = torch.cat([actions, relevance.unsqueeze(-1)], dim=-1)
    with torch.no_grad():
      return self.model.sample(cond, steps=10)


class GaussianPolicy(nn.Module):
  def __init__(self, state_dim, action_dim, hidden_dim=64):
    super().__init__()
    self.fc_mean = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, action_dim))
    self.log_std = nn.Parameter(torch.zeros(action_dim))

  def forward(self, state):
    mean = self.fc_mean(state)
    std = self.log_std.exp().expand_as(mean)
    return mean, std

  def act(self, state):
    with torch.no_grad():
      mean, std = self.forward(state)
      noise = torch.randn_like(mean)
      action = mean + std * noise
    return action


class QNetwork(nn.Module):
  def __init__(self, state_dim, action_dim, hidden_dim=64):
    super().__init__()
    self.net = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

  def forward(self, s, a):
    return self.net(torch.cat([s, a], dim=-1))


def train_agent(policy, q_net, optimizer_q, buffer_real, buffer_syn, batch_size, gamma=0.99, entropy_coef=0.01):
  s_real, a_real, r_real, s2_real, d_real = buffer_real.sample(batch_size)
  s_syn, a_syn, r_syn, s2_syn, d_syn = buffer_syn.sample(batch_size)

  q_val_real = q_net(s_real, a_real).squeeze(-1)
  with torch.no_grad():
    mean_next_real, std_next_real = policy(s2_real)
    noise_real = torch.randn_like(mean_next_real)
    a2_real = mean_next_real + std_next_real * noise_real
    q_next_real = q_net(s2_real, a2_real).squeeze(-1)
    log_prob_real = -0.5 * ((noise_real**2).sum(dim=1) + a2_real.size(1) * np.log(2.0 * np.pi) + 2 * std_next_real.log().sum(dim=1))
    target_real = r_real + gamma * (1 - d_real) * (q_next_real + (-entropy_coef) * log_prob_real)

  loss_real = F.mse_loss(q_val_real, target_real)

  q_val_syn = q_net(s_syn, a_syn).squeeze(-1)
  with torch.no_grad():
    mean_next_syn, std_next_syn = policy(s2_syn)
    noise_syn = torch.randn_like(mean_next_syn)
    a2_syn = mean_next_syn + std_next_syn * noise_syn
    q_next_syn = q_net(s2_syn, a2_syn).squeeze(-1)
    log_prob_syn = -0.5 * ((noise_syn**2).sum(dim=1) + a2_syn.size(1) * np.log(2.0 * np.pi) + 2 * std_next_syn.log().sum(dim=1))
    target_syn = r_syn + gamma * (1 - d_syn) * (q_next_syn + (-entropy_coef) * log_prob_syn)

  loss_syn = F.mse_loss(q_val_syn, target_syn)
  loss_q = loss_real + loss_syn
  optimizer_q.zero_grad()
  loss_q.backward()
  optimizer_q.step()

  policy_optimizer = optim.Adam(policy.parameters(), lr=1e-3)
  mean_now, std_now = policy(s_real)
  noise_now = torch.randn_like(mean_now)
  a_now = mean_now + std_now * noise_now
  q_val_now = q_net(s_real, a_now).squeeze(-1)
  log_prob_now = -0.5 * ((noise_now**2).sum(dim=1) + a_now.size(1) * np.log(2.0 * np.pi) + 2 * std_now.log().sum(dim=1))
  loss_policy = -q_val_now.mean() - entropy_coef * log_prob_now.mean()
  policy_optimizer.zero_grad()
  loss_policy.backward()
  policy_optimizer.step()

  return loss_q.item(), loss_policy.item()


def main():
  env = gym.make("LunarLanderContinuous-v3", render_mode=None)
  obs_dim = env.observation_space.shape[0]
  act_dim = env.action_space.shape[0]

  real_buffer = ReplayBuffer(100000)
  syn_buffer = ReplayBuffer(100000)

  policy = GaussianPolicy(obs_dim, act_dim)
  q_net = QNetwork(obs_dim, act_dim)
  optimizer_q = optim.Adam(q_net.parameters(), lr=1e-3)

  curiosity = CuriosityModel(obs_dim, act_dim, latent_dim=16)
  generative = GenerativeModel(obs_dim, act_dim, lr=1e-3)

  episodes = 10000
  max_steps = 1000
  batch_size = 64
  entropy_coef = 0.01

  # Metrics
  reward_history = []
  step_history = []

  for ep in range(episodes):
    print(f"\nEpisode {ep+1}/{episodes}")
    obs, _ = env.reset()
    done = False
    steps = 0
    total_reward = 0  # Tracks total reward for the episode

    while not done and steps < max_steps:
      s_t = torch.FloatTensor(obs).unsqueeze(0)
      act = policy.act(s_t).squeeze(0).numpy()
      next_obs, reward, terminated, truncated, _ = env.step(act)
      d = float(terminated or truncated)
      real_buffer.push(obs, act, reward, next_obs, d)
      obs = next_obs
      steps += 1
      total_reward += reward
      if terminated or truncated:
        break

    # Log episode metrics
    reward_history.append(total_reward)
    step_history.append(steps)
    print(f"  Steps: {steps}, Total Reward: {total_reward:.2f}")

    # Training
    if len(real_buffer) > batch_size:
      s_b, a_b, r_b, s2_b, d_b = real_buffer.sample(batch_size)
      curiosity.update(s_b, a_b, s2_b)
      scores = curiosity.compute_relevance(s_b, a_b, s2_b)
      generative.train_model(s_b, a_b, s2_b, torch.FloatTensor(scores))
      k_thresh = np.percentile(scores, 80)
      idx = scores >= k_thresh
      if idx.any():
        hs, ha, hr, hs2, hd = s_b[idx], a_b[idx], r_b[idx], s2_b[idx], d_b[idx]
        gen_s = generative.generate(ha, torch.FloatTensor(scores[idx]), len(hs))
        gen_s = gen_s.cpu()
        for i in range(gen_s.size(0)):
          syn_buffer.push(hs[i].numpy(), ha[i].numpy(), hr[i].item(), gen_s[i].numpy(), hd[i].item())

    if len(real_buffer) > batch_size and len(syn_buffer) > batch_size:
      train_agent(policy, q_net, optimizer_q, real_buffer, syn_buffer, batch_size, gamma=0.99, entropy_coef=entropy_coef)

  # Print overall statistics
  print("\nTraining Complete")
  print(f"  Average Reward: {np.mean(reward_history):.2f}")
  print(f"  Average Steps per Episode: {np.mean(step_history):.2f}")

  env.close()


if __name__ == "__main__":
  main()
