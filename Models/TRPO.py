

def optimal(trial):
  # Define the TRPO-specific optimization space
  params = {
      "gamma": trial.suggest_float("gamma", 0.98, 0.999, log=True),
      "gae_lambda": trial.suggest_float("gae_lambda", 0.9, 0.99),
      "target_kl": trial.suggest_float("target_kl", 0.005, 0.02),
      "cg_damping": trial.suggest_float("cg_damping", 0.01, 0.1),
      "cg_max_steps": trial.suggest_int("cg_max_steps", 10, 20),
      "line_search_max_iter": trial.suggest_int("line_search_max_iter", 5, 15),
      "n_steps": trial.suggest_categorical("n_steps", [1024, 2048, 4096]),
      "batch_size": trial.suggest_int("batch_size", 64, 256, step=64),
      "total_timesteps": trial.suggest_int("total_timesteps", 100_000, 1_000_000, step=100_000),
  }
  return params
