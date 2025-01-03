

def optimal(trial, total_timesteps):
  # Define the TRPO-specific optimization space
  params = {
      "policy": "MlpPolicy",
      "gamma": trial.suggest_float("gamma", 0.98, 0.999, log=True),
      "gae_lambda": trial.suggest_float("gae_lambda", 0.9, 0.99, step=0.001),
      "target_kl": trial.suggest_float("target_kl", 0.001, 0.05, step=0.001),
      "cg_damping": trial.suggest_float("cg_damping", 0.01, 0.1, step=0.01),
      "cg_max_steps": trial.suggest_int("cg_max_steps", 10, 20, step=1),
      "line_search_max_iter": trial.suggest_int("line_search_max_iter", 5, 15, step=5),
      "n_steps": trial.suggest_categorical("n_steps", [1024, 2048, 4096]),
      "batch_size": trial.suggest_int("batch_size", 32, 256, step=32),
      "total_timesteps": trial.suggest_int("total_timesteps", total_timesteps // 10, total_timesteps, step=total_timesteps // 10),
  }
  return params
