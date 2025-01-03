from stable_baselines3.common.callbacks import BaseCallback

class TimestepsProgress(BaseCallback):
  """
  A callback to dynamically adjust total timesteps if exceeded
  and log the overall progress of training.
  """

  def __init__(self, total_timesteps, verbose = 0):
    super().__init__(verbose)
    self.total_timesteps = total_timesteps
    self.interval = 1000 if total_timesteps > 1000000 else 100 if total_timesteps > 100000 else 10
    self.progress_step = total_timesteps // self.interval
    self.next_update = self.progress_step

  def _on_step(self) -> bool:
    current_timesteps = self.model.num_timesteps

    # Adjust total timesteps if exceeded
    if current_timesteps > self.total_timesteps:
      self.total_timesteps = current_timesteps
      self.progress_step = self.total_timesteps// self.interval
      self.next_update = current_timesteps + self.progress_step

    # Log progress if the next update step is reached
    if current_timesteps >= self.next_update:
      progress_percentage = (current_timesteps / self.total_timesteps) * 100
      print(f"[Progress: {progress_percentage:.2f}% ({current_timesteps}/{self.total_timesteps} timesteps)]")
      self.next_update += self.progress_step

    return True

  def _on_training_end(self) -> None:
    print(f"Training completed. Total timesteps: {self.model.num_timesteps}.")
