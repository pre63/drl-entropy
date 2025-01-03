import numpy as np

from stable_baselines3.common.callbacks import BaseCallback


class TrainingMetrics(BaseCallback):
  """
  A callback to track training metrics during steps and rollouts, including dynamic normalization values.
  """

  def __init__(self, max_reward, verbose=0):
    super().__init__(verbose)

    # Track reward history and normalization parameters
    self.reward_history = []
    self.max_reward = max_reward

    # Dynamic normalization values
    self.max_rollout_reward_variance = 1e-6  # Start with a small value to avoid division by zero
    self.max_convergence_metric = 1e-6

    # Step-level metrics
    self.step_success_count = 0
    self.step_total_episodes = 0
    self.step_success_rate = 0

    # Rollout-level metrics
    self.rollout_reward_history = []
    self.rollout_episode_lengths = []
    self.rollout_reward_variance = None
    self.rollout_converged = False

    self.steps = 0

  def _on_step(self):
    """
    Update step-level metrics during training.
    """
    self.steps += 1
    rewards = self.locals.get("rewards", [])
    self.reward_history.extend(rewards)

    infos = self.locals.get("infos", [])
    for info in infos:
      if info.get("success", False):
        self.step_success_count += 1

    return True

  def _on_rollout_end(self):
    """
    Update rollout-level metrics at the end of each training rollout.
    """
    infos = self.locals.get("infos", [])
    rewards = self.locals.get("rewards", [])

    self.step_total_episodes += len(infos)
    self.step_success_rate = self.step_success_count / self.step_total_episodes if self.step_total_episodes > 0 else 0

    self.rollout_reward_history.extend(rewards)
    self.rollout_episode_lengths.extend([info.get("episode", {}).get("l", 0) for info in infos if "episode" in info])

    if len(self.rollout_reward_history) > 1:
      self.rollout_reward_variance = np.var(self.rollout_reward_history)

    # Update dynamic max values
    if self.rollout_reward_variance and self.rollout_reward_variance > self.max_rollout_reward_variance:
      self.max_rollout_reward_variance = self.rollout_reward_variance

    if self.logger:
      self.logger.record("rollout/reward_variance", self.rollout_reward_variance)
      self.logger.record("rollout/success_rate", self.step_success_rate)
      self.logger.record("rollout/total_episodes", self.step_total_episodes)
      self.logger.record("rollout/success_count", self.step_success_count)

  def get_convergence_metric(self):
    """
    Compute a scalar metric representing training convergence, normalized to [0, 1].
    """
    if not self.reward_history:
      return 0

    # AUC (Area Under the Curve)
    auc = np.sum(self.reward_history)

    # Reward stability (1 - normalized variance)
    if len(self.reward_history) > 1:
      variance = np.var(self.reward_history)
      stability = 1 - variance / (self.max_reward - np.min(self.reward_history))
    else:
      stability = 1

    # Final reward (mean of the last 10% of rewards)
    final_rewards = self.reward_history[-max(1, len(self.reward_history) // 10):]
    final_avg_reward = np.mean(final_rewards)

    # Weighted Convergence Metric
    w1, w2, w3 = 0.5, 0.3, 0.2
    convergence_metric = w1 * auc + w2 * stability + w3 * final_avg_reward

    # Normalize convergence metric
    max_possible_convergence = self.steps * self.max_reward  # Maximum possible sum of rewards
    max_possible_convergence = max(max_possible_convergence, 1e-6)  # Avoid division by zero
    convergence_metric_norm = np.clip(convergence_metric / max_possible_convergence, 0, 1)

    return convergence_metric_norm

  def get_normalized_metrics(self):
    """
    Return normalized values for rollout_reward_variance and convergence_metric.
    """
    # Normalize rollout_reward_variance
    if self.rollout_reward_variance is not None:
      rollout_reward_variance_norm = self.rollout_reward_variance / self.max_rollout_reward_variance
    else:
      rollout_reward_variance_norm = 1.0

    # Normalize convergence_metric
    convergence_metric_norm = self.get_convergence_metric()

    return rollout_reward_variance_norm, convergence_metric_norm
