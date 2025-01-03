from stable_baselines3.common.callbacks import EvalCallback


class EvaluationMetrics(EvalCallback):
  def __init__(self, eval_env, eval_freq=5000, n_eval_episodes=10, verbose=0, **kwargs):
    super().__init__(eval_env, eval_freq=eval_freq, n_eval_episodes=n_eval_episodes, verbose=verbose, **kwargs)
    self.eval_success_count = 0
    self.eval_total_episodes = 0
    self.eval_success_rate = 0.0
    self.eval_mean_reward = 0
    self.eval_mean_ep_length = 0
    self.eval_mean_success_rate = 0
    self.eval_total_rewards = 0

  def _on_step(self) -> bool:
    if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
      episode_rewards, episode_lengths = self.evaluate()
      self.log_metrics(episode_rewards, episode_lengths)
    return True

  def evaluate(self):
    print("Evaluating model for {} episodes".format(self.n_eval_episodes))
    episode_rewards, episode_lengths = [], []
    success_count = 0

    for _ in range(self.n_eval_episodes):
      obs = self.eval_env.reset()
      done = False
      episode_reward = 0.0

      while not done:
        action, _ = self.model.predict(obs, deterministic=self.deterministic)
        obs, reward, done, info = self.eval_env.step(action)
        info = info[0]
        episode_reward += reward

        if "success" in info and info["success"]:
          success_count += 1

      episode_rewards.append(episode_reward)
      episode_lengths.append(info.get("episode", {}).get("l", 0))

    self.eval_success_count += success_count
    self.eval_total_episodes += self.n_eval_episodes
    self.eval_success_rate = self.eval_success_count / self.eval_total_episodes if self.eval_total_episodes > 0 else 0
    self.eval_mean_reward = sum(episode_rewards) / self.n_eval_episodes
    self.eval_mean_ep_length = sum(episode_lengths) / self.n_eval_episodes
    self.eval_mean_success_rate = success_count / self.n_eval_episodes
    self.eval_total_rewards = sum(episode_rewards)

    return episode_rewards, episode_lengths

  def log_metrics(self, episode_rewards, episode_lengths):
    if self.logger:
      self.logger.record("eval/success_count", self.eval_success_count)
      self.logger.record("eval/success_rate", self.eval_success_rate)
      self.logger.record("eval/mean_reward", self.eval_mean_reward)
      self.logger.record("eval/mean_ep_length", self.eval_mean_ep_length)
      self.logger.record("eval/mean_success_rate", self.eval_mean_success_rate)
      self.logger.record("eval/episode_rewards", episode_rewards)
      self.logger.record("eval/episode_lengths", episode_lengths)
      self.logger.record("eval/total_rewards", self.eval_total_rewards)
