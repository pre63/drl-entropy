
def optimal(trial):
  # Define the PPO-specific optimization space
  params = {
      "gamma": trial.suggest_float("gamma", 0.98, 0.999, log=True),
      "gae_lambda": trial.suggest_float("gae_lambda", 0.9, 0.99, step=0.001),
      "n_steps": trial.suggest_int("n_steps", 1024, 4096, step=1024),
      "batch_size": trial.suggest_int("batch_size", 32, 256, step=32),
      "target_kl": trial.suggest_float("target_kl", 0.001, 0.05, step=0.001),
      "ent_coef": trial.suggest_float("ent_coef", 0.0, 0.05),
      "total_timesteps": trial.suggest_int("total_timesteps", 1_000_000, 10_000_000, step=500_000),

  }
  return params
