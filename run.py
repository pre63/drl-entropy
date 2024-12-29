import gymnasium as gym
from Experiment import Experiment
from itertools import product
from concurrent.futures import ThreadPoolExecutor
from Models.TRPO import TRPO
from Models.TRPO3 import TRPO3
from Models.EnTRPO import EnTRPO
from Models.EnTRPOTrace import EnTRPOTrace
from Train import Train
from Metrics import Metrics


def run_experiment(model_cls, env, model_params, params, experiment_id):
  batch_size, episodes_per_batch, factor = params["batch_size"], params["episodes_per_batch"], params["factor"]
  model = model_cls(**model_params)
  total_timesteps = batch_size * episodes_per_batch * factor

  print(f"Starting experiment {experiment_id} with params: {model_params}")
  Train.batch(model, env, total_timesteps, batch_size, **model_params)
  Experiment.save(model, env)
  print(f"Experiment {experiment_id} completed.")


def train_all_models_multithreaded(pairs, params, env):
  # Calculate total number of experiments
  total_experiments = sum(
      len(list(product(*param_grid.values()))) for _, param_grid in pairs
  )
  print(f"Total number of experiments: {total_experiments}")

  with ThreadPoolExecutor(max_workers=12) as executor:  # Adjust max_workers based on your system's capabilities
    print("Starting experiments...")
    futures = []
    for model_idx, (model_cls, param_grid) in enumerate(pairs):
      param_combinations = list(product(*param_grid.values()))
      print(f"Training model {model_cls.__name__} with {len(param_combinations)} configurations.")
      param_keys = list(param_grid.keys())

      for param_idx, param_values in enumerate(param_combinations):
        model_params = dict(zip(param_keys, param_values))
        experiment_id = f"Model{model_idx + 1}_Config{param_idx + 1}"
        futures.append(
            executor.submit(run_experiment, model_cls, env, model_params, params, experiment_id)
        )

    for future in futures:
      future.result()


if __name__ == "__main__":
  from Environments.Pendulum import make
  env = make()

  state_dim = env.observation_space.shape[0]
  action_dim = env.action_space.shape[0]

  param_grid = {
      "gamma": [0.95, 0.99],
      "lam": [0.9, 0.95],
      "critic_lr": [1e-1, 1e-2, 1e-3, 1e-4],
      "hidden_sizes": [[64, 64], [128, 128], [256, 256]],
      "kl_threshold": [1e-1, 1e-2, 1e-3, 1e-4],
      "state_dim": [state_dim],
      "action_dim": [action_dim],
  }

  param_grid_entropy = {
      **param_grid,
      "entropy_coeff": [0.01, 0.001],
      "trace_decay": [0.5, 0.8, 0.9],
  }

  models = [TRPO, TRPO3, EnTRPO, EnTRPOTrace]
  configs = [param_grid, param_grid, param_grid_entropy, param_grid_entropy]
  pairs = list(zip(models, configs))

  params = {
      "batch_size": 32,
      "episodes_per_batch": 10,
      "factor": 10000,
  }

  train_all_models_multithreaded(pairs, params, env)
