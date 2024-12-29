import os
import json
import torch
import pandas as pd
from datetime import datetime

import torch


class ModelSpec(torch.nn.Module):
  def __init__(self, **params):
    super(ModelSpec, self).__init__()
    self.params = params

    # Metrics for logging
    self.timesteps = 0
    self.rewards = []  # Full history of rewards
    self.step_times = []  # Full history of step times
    self.episode_rewards = []  # Total reward per episode
    self.episode_step_times = []  # Average step time per episode
    self.episode_successes = []  # Success indicator for each episode
    self.total_obs = []  # Observations per episode


    # Losses
    self.actor_loss, self.critic_loss, self.kl_divergence_loss = 0, 0, 0

    self.actor_loss_history = []
    self.critic_loss_history = []
    self.kl_divergence_loss_history = []

  def log_loss(self, actor_loss, critic_loss, kl_loss):
    self.actor_loss_history.append(actor_loss)
    self.critic_loss_history.append(critic_loss)
    self.kl_divergence_loss_history.append(kl_loss)

  @property
  def name(self):
    return self.__class__.__name__

  def log_step(self, reward, step_time):
    """
    Log data for a single step in the environment.
    """
    self.timesteps += 1
    self.rewards.append(reward)
    self.step_times.append(step_time)

  def log_episode(self, rewards, step_times, total_obs, success):
    """
    Log data for a single episode.
    - rewards: List of rewards for each step in the episode.
    - step_times: List of times taken for each step.
    - total_obs: Total observations (data points) used in this episode.
    - success: Whether the episode was successful (1 for success, 0 for failure).
    """
    self.episode_rewards.append(sum(rewards))
    self.episode_step_times.append(sum(step_times) / len(step_times) if step_times else 0)
    self.total_obs.append(total_obs)
    self.episode_successes.append(success)

  def get_metrics(self):
    """
    Compute and return aggregate metrics for the model.
    """
    avg_reward = sum(self.rewards) / len(self.rewards) if self.rewards else 0
    reward_variance = (
        sum((r - avg_reward) ** 2 for r in self.rewards) / len(self.rewards) if self.rewards else 0
    )
    avg_step_time = sum(self.step_times) / len(self.step_times) if self.step_times else 0
    avg_episode_reward = sum(self.episode_rewards) / len(self.episode_rewards) if self.episode_rewards else 0
    avg_obs_per_episode = sum(self.total_obs) / len(self.total_obs) if self.total_obs else 0
    success_rate = sum(self.episode_successes) / len(self.episode_successes) if self.episode_successes else 0

    # Loss
    actor_loss = sum(self.actor_loss_history) / len(self.actor_loss_history) if self.actor_loss_history else 0
    critic_loss = sum(self.critic_loss_history) / len(self.critic_loss_history) if self.critic_loss_history else 0
    kl_divergence = sum(self.kl_divergence_loss_history) / len(self.kl_divergence_loss_history) if self.kl_divergence_loss_history else 0

    return {
        "Model Name": self.name,
        "Params": self.params,
        "Timesteps": self.timesteps,
        "Average Step Time": avg_step_time,
        "Average Rewards (Step)": avg_reward,
        "Reward Variance (Step)": reward_variance,
        "Average Rewards (Episode)": avg_episode_reward,
        "Average Observations Per Episode": avg_obs_per_episode,
        "Success Rate": success_rate,
        "Actor Loss": actor_loss,
        "Critic Loss": critic_loss,
        "KL Divergence": kl_divergence,
    }

  def get_history(self):
    """
    Return the full history of metrics for analysis and plotting.
    """
    return {
        "Step Rewards": self.rewards,
        "Step Times": self.step_times,
        "Episode Rewards": self.episode_rewards,
        "Episode Step Times": self.episode_step_times,
        "Episode Successes": self.episode_successes,
        "Episode Observations": self.total_obs,
        "KL Divergence": self.kl_divergence_loss_history,
        "Actor Loss": self.actor_loss_history,
        "Critic Loss": self.critic_loss_history,
    }
