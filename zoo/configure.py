import difflib
import importlib
import os
import time
import uuid

import gymnasium as gym
import numpy as np
# Register custom environments
import rl_zoo3.import_envs  # noqa: F401
import stable_baselines3 as sb3
import torch as th
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from rl_zoo3.exp_manager import ExperimentManager
from rl_zoo3.utils import ALGOS
from stable_baselines3.common.utils import set_random_seed


def configure(**params):
  # Set default values and override with provided params
  args = {
      "algo": "ppo",
      "env": "CartPole-v1",
      "tensorboard_log": "",
      "trained_agent": "",
      "truncate_last_trajectory": True,
      "n_timesteps": -1,
      "num_threads": -1,
      "log_interval": -1,
      "eval_freq": 25000,
      "optimization_log_path": None,
      "eval_episodes": 5,
      "n_eval_envs": 4,
      "save_freq": -1,
      "save_replay_buffer": False,
      "log_folder": ".logs",
      "seed": -1,
      "vec_env": "dummy",
      "device": "auto",
      "n_trials": 500,
      "max_total_trials": None,
      "optimize_hyperparameters": False,
      "no_optim_plots": False,
      "n_jobs": 2,
      "sampler": "tpe",
      "pruner": "halving",
      "n_startup_trials": 10,
      "n_evaluations": None,
      "storage": None,
      "study_name": None,
      "verbose": 1,
      "gym_packages": [],
      "env_kwargs": {},
      "eval_env_kwargs": {},
      "hyperparams": {},
      "conf_file": None,
      "uuid": False,
      "track": True,
      "wandb_project_name": "sb3",
      "wandb_entity": None,
      "wandb_tags": [],
      "progress": False,
  }

  # Update defaults with provided params
  args.update(params)

  # Register custom environments if specified
  for env_module in args["gym_packages"]:
    importlib.import_module(env_module)

  env_id = args["env"]
  registered_envs = set(gym.envs.registry.keys())

  # Handle non-existing environments
  if env_id not in registered_envs:
    try:
      closest_match = difflib.get_close_matches(env_id, registered_envs, n=1)[0]
    except IndexError:
      closest_match = "'no close match found...'"
    raise ValueError(f"{env_id} not found in gym registry, you maybe meant {closest_match}?")

  # Generate a unique ID if requested
  uuid_str = f"_{uuid.uuid4()}" if args["uuid"] else ""
  if args["seed"] < 0:
    args["seed"] = np.random.randint(2**32 - 1, dtype="int64").item()

  # Set the seed for reproducibility
  set_random_seed(args["seed"])

  # Set PyTorch threads
  if args["num_threads"] > 0:
    th.set_num_threads(args["num_threads"])

  # Validate trained agent path
  if args["trained_agent"]:
    assert args["trained_agent"].endswith(".zip") and os.path.isfile(args["trained_agent"]), \
        "The trained agent must be a valid path to a .zip file"

  print(f"\n{'=' * 10} Training {env_id} {'=' * 10}")
  print(f"Seed: {args['seed']}")

  # Track experiment using TensorBoard
  if args["track"]:
    run_name = f"{args['env']}__{args['algo']}__{args['seed']}__{int(time.time())}"
    args["tensorboard_log"] = f".logs/tensorboard/{run_name}"

  if args['storage'] is None:
    optuna_dir = f".optuna-zoo/{args['algo']}_{args['env']}"
    os.mkdir(optuna_dir) if not os.path.exists(optuna_dir) else None
    storage = JournalStorage(JournalFileBackend(f"{optuna_dir}/storage"))
    args['storage'] = storage

    study_name = f"{args['algo']}_{args['env']}_study"
    args['study_name'] = study_name

  print(f"Timesteps: {args['n_timesteps']}")

  # Initialize the experiment manager
  exp_manager = ExperimentManager(
      args=args,
      algo=args["algo"],
      env_id=env_id,
      log_folder=args["log_folder"],
      tensorboard_log=args["tensorboard_log"],
      n_timesteps=args["n_timesteps"],
      eval_freq=args["eval_freq"],
      n_eval_episodes=args["eval_episodes"],
      save_freq=args["save_freq"],
      hyperparams=args["hyperparams"],
      env_kwargs=args["env_kwargs"],
      eval_env_kwargs=args["eval_env_kwargs"],
      trained_agent=args["trained_agent"],
      optimize_hyperparameters=args["optimize_hyperparameters"],
      storage=storage,
      study_name=args["study_name"],
      n_trials=args["n_trials"],
      max_total_trials=args["max_total_trials"],
      n_jobs=args["n_jobs"],
      sampler=args["sampler"],
      pruner=args["pruner"],
      optimization_log_path=args["optimization_log_path"],
      n_startup_trials=args["n_startup_trials"],
      n_evaluations=args["n_evaluations"],
      truncate_last_trajectory=args["truncate_last_trajectory"],
      uuid_str=uuid_str,
      seed=args["seed"],
      log_interval=args["log_interval"],
      save_replay_buffer=args["save_replay_buffer"],
      verbose=args["verbose"],
      vec_env_type=args["vec_env"],
      n_eval_envs=args["n_eval_envs"],
      no_optim_plots=args["no_optim_plots"],
      device=args["device"],
      config=args["conf_file"],
      show_progress=args["progress"],
  )

  # Prepare and start the experiment
  results = exp_manager.setup_experiment()
  if results is not None:
    model, saved_hyperparams = results
    if model is not None:
      exp_manager.learn(model)
      exp_manager.save_trained_model(model)
  else:
    exp_manager.hyperparameters_optimization()

  return exp_manager


# Example usage
if __name__ == "__main__":
  configure(
      algo="ppo",
      env="CartPole-v1",
      n_timesteps=10000,
      seed=42,
      optimize_hyperparameters=False,
  )
