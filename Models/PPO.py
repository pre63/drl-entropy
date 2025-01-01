
def optimal(trial):
  # Define the PPO-specific optimization space
  params = {
      "gamma": trial.suggest_float("gamma", 0.98, 0.999, log=True),
      "gae_lambda": trial.suggest_float("gae_lambda", 0.9, 0.99),
      "n_steps": trial.suggest_int("n_steps", 1024, 4096, step=1024),
      "batch_size": trial.suggest_int("batch_size", 64, 256, step=64),
      "ent_coef": trial.suggest_float("ent_coef", 0.0, 0.05),
      "total_timesteps": trial.suggest_int("total_timesteps", 100_000, 1_000_000, step=100_000),
  }
  return params
