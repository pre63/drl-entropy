from stable_baselines3.common.callbacks import EvalCallback


class EvaluationMetrics(EvalCallback):
  def __init__(self, eval_env, eval_freq=5000, n_eval_episodes=10, verbose=0, **kwargs):
    super().__init__(eval_env, eval_freq=eval_freq, n_eval_episodes=n_eval_episodes, verbose=verbose, **kwargs)
    self.eval_success_count = 0
    self.eval_total_episodes = 0
    self.eval_success_rate = 0.0
    self.eval_mean_reward = 0.0
    self.eval_mean_ep_length = 0.0
    self.eval_mean_success_rate = 0.0
    self.eval_total_rewards = 0.0

  def _on_step(self) -> bool:
    if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
      episode_rewards, episode_lengths = self.evaluate()
      self.log_metrics(episode_rewards, episode_lengths)
    return True

  def evaluate(self):
    print(f"Evaluating model for {self.n_eval_episodes} episodes")
    episode_rewards = []
    episode_lengths = []
    success_count = 0

    for _ in range(self.n_eval_episodes):
      obs = self.eval_env.reset()
      done = False
      episode_reward = 0.0

      while not done:
        action, _ = self.model.predict(obs, deterministic=self.deterministic)
        obs, reward, done, info = self.eval_env.step(action)
        info = info[0] if isinstance(info, (list, tuple)) else info
        episode_reward += reward

        if "success" in info and info["success"]:
          success_count += 1

      episode_rewards.append(episode_reward)
      episode_lengths.append(info.get("episode", {}).get("l", 0))

    self.eval_success_count += success_count
    self.eval_total_episodes += self.n_eval_episodes
    self.eval_success_rate = self.eval_success_count / self.eval_total_episodes if self.eval_total_episodes > 0 else 0.0
    self.eval_mean_reward = sum(episode_rewards) / len(episode_rewards) if episode_rewards else 0.0
    self.eval_mean_ep_length = sum(episode_lengths) / len(episode_lengths) if episode_lengths else 0.0
    self.eval_mean_success_rate = success_count / self.n_eval_episodes if self.n_eval_episodes > 0 else 0.0
    self.eval_total_rewards = sum(episode_rewards)

    return episode_rewards, episode_lengths

  def log_metrics(self, episode_rewards, episode_lengths):
    if self.logger:
      self.logger.record("eval/success_count", int(self.eval_success_count))
      self.logger.record("eval/success_rate", float(self.eval_success_rate))
      self.logger.record("eval/mean_reward", float(self.eval_mean_reward))
      self.logger.record("eval/mean_ep_length", float(self.eval_mean_ep_length))
      self.logger.record("eval/mean_success_rate", float(self.eval_mean_success_rate))
      self.logger.record("eval/episode_rewards", [float(r) for r in episode_rewards])
      self.logger.record("eval/episode_lengths", [int(l) for l in episode_lengths])
      self.logger.record("eval/total_rewards", float(self.eval_total_rewards))
