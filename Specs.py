import os
import json
import torch
import pandas as pd
from datetime import datetime

import numpy as np
import torch


class ModelSpec(torch.nn.Module):
  """
  Base class for RL Models, providing consistent logging of training and evaluation metrics.
  Subclasses should implement their specific RL algorithms (e.g., TRPO).
  """

  def __init__(self, **params):
    super().__init__()
    self.params = params

    # Running tally of timesteps (step-based counter)
    self.timesteps = 0

    self.step_logs = []
    self.loss_log = []

    self.episode_logs = []
    self.episode_count = 0

    self.eval_logs = ([],[])

    # For storing the most recent losses (actor, critic, KL)
    self.actor_loss = 0.0
    self.critic_loss = 0.0
    self.kl_divergence_loss = 0.0

  @property
  def name(self):
    return self.__class__.__name__

  def log_step(self, reward, step_time):
    """
    Log a single environment step.
    - reward: The immediate reward at this step.
    - step_time: How long this step took to run (for profiling).
    """
    self.timesteps += 1
    # Record step logs
    self.step_logs.append({
        "timestep": self.timesteps,
        "reward": reward,
        "step_time": step_time
    })

  def log_episode(self, episode_rewards, avg_step_time, success, episode_steps):
    """
    Log a single episode.
    - episode_rewards: Sum of rewards in this episode
    - avg_step_time: Average step time across the steps in this episode
    - success: 1 if success, else 0
    - episode_steps: Number of steps in this episode
    """
    self.episode_count += 1
    self.episode_logs.append({
        "episode_index": self.episode_count,
        "episode_rewards": episode_rewards,
        "avg_step_time": avg_step_time,
        "success": success,
        "episode_steps": episode_steps
    })

  def log_loss(self, actor_loss, critic_loss, kl_div):
    self.actor_loss = actor_loss
    self.critic_loss = critic_loss
    self.kl_divergence_loss = kl_div

    self.loss_log.append({
        "actor_loss": actor_loss,
        "critic_loss": critic_loss,
        "kl_divergence": kl_div
    })

  def log_evaluation(self, episode_rewards, episode_successes):
    self.eval_logs = (episode_rewards, episode_successes)

  def get_step_metrics(self):
    """
    Return step-wise logs.
    """
    return self.step_logs

  def get_episode_metrics(self):
    """
    Return episode-wise logs.
    """
    return self.episode_logs

  def get_eval_metrics(self):
    """
    Return all evaluation logs.
    """
    return self.eval_logs

  def get_loss_metrics(self):
    """
    Return all loss logs.
    """
    return self.loss_log

  def get_training_summary(self):
    """
    Compute overall training metrics that are meaningful for an RL setting.
    You might compute an average episode return, success rate, etc.
    """
    return {
        "Model Name": self.name,
        "Params": self.params,
        "Timesteps": self.timesteps,
        "Average Step Time": np.mean([step["step_time"] for step in self.step_logs]),
        "Average Rewards (Step)": np.mean([step["reward"] for step in self.step_logs]),
        "Reward Variance (Step)": np.var([step["reward"] for step in self.step_logs]),
        "Average Rewards (Episode)": np.mean([ep["episode_rewards"] for ep in self.episode_logs]),
        "Average Observations Per Episode": np.mean([ep["episode_steps"] for ep in self.episode_logs]),
        "Eval Success Rate": np.mean(self.eval_logs[1]),
        "Eval Average Rewards": np.mean(self.eval_logs[0]),
        "Episodes": self.episode_count,
        "Actor Loss": self.actor_loss,
        "Critic Loss": self.critic_loss,
        "KL Divergence": self.kl_divergence_loss,
    }

  def __len__(self):
    return self.episode_count
