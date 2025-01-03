from stable_baselines3.common.callbacks import BaseCallback


class TrialPruning(BaseCallback):
  """
  A custom callback that integrates Optuna trial pruning.
  """

  def __init__(self, trial, eval_callback, verbose=0):
    super().__init__(verbose)
    self.trial = trial
    self.eval_callback = eval_callback  # Reference to the EvalCallback to access evaluation results

  def _on_step(self):
    # Access the latest evaluation results
    if len(self.eval_callback.evaluations_results) > 0:
      # Get the latest mean reward from the EvalCallback
      mean_reward = self.eval_callback.evaluations_results[-1][0]  # Mean reward of the last evaluation
      self.trial.report(mean_reward, self.n_calls)

      # Check if the trial should be pruned
      if self.trial.should_prune():
        raise optuna.TrialPruned()
    return True
