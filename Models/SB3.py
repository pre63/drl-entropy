from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, TypeVar, Union

import optuna
import torch as th
import torch.nn as nn
from sb3_contrib import TRPO
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, Schedule


class TRPO(TRPO):
  """
    Just a class to ingore extra arguments for compatibility.
    """

  def __init__(
    self,
    policy: Union[str, Type[ActorCriticPolicy]],
    env: Union[GymEnv, str],
    learning_rate: Union[float, Schedule] = 1e-3,
    n_steps: int = 2048,
    batch_size: int = 128,
    gamma: float = 0.99,
    cg_max_steps: int = 15,
    cg_damping: float = 0.1,
    line_search_shrinking_factor: float = 0.8,
    line_search_max_iter: int = 10,
    n_critic_updates: int = 10,
    gae_lambda: float = 0.95,
    use_sde: bool = False,
    sde_sample_freq: int = -1,
    rollout_buffer_class: Optional[Type[RolloutBuffer]] = None,
    rollout_buffer_kwargs: Optional[Dict[str, Any]] = None,
    normalize_advantage: bool = True,
    target_kl: float = 0.01,
    sub_sampling_factor: int = 1,
    stats_window_size: int = 100,
    tensorboard_log: Optional[str] = None,
    policy_kwargs: Optional[Dict[str, Any]] = None,
    verbose: int = 0,
    seed: Optional[int] = None,
    device: Union[th.device, str] = "auto",
    _init_setup_model: bool = True,
    **kwargs,
  ):
    super().__init__(
      policy=policy,
      env=env,
      learning_rate=learning_rate,
      n_steps=n_steps,
      batch_size=batch_size,
      gamma=gamma,
      cg_max_steps=cg_max_steps,
      cg_damping=cg_damping,
      line_search_shrinking_factor=line_search_shrinking_factor,
      line_search_max_iter=line_search_max_iter,
      n_critic_updates=n_critic_updates,
      gae_lambda=gae_lambda,
      use_sde=use_sde,
      sde_sample_freq=sde_sample_freq,
      rollout_buffer_class=rollout_buffer_class,
      rollout_buffer_kwargs=rollout_buffer_kwargs,
      normalize_advantage=normalize_advantage,
      target_kl=target_kl,
      sub_sampling_factor=sub_sampling_factor,
      stats_window_size=stats_window_size,
      tensorboard_log=tensorboard_log,
      policy_kwargs=policy_kwargs,
      verbose=verbose,
      seed=seed,
      device=device,
      _init_setup_model=_init_setup_model,
    )
    # Ignore kwargs for compatibility


def sample_trpo_params(trial: optuna.Trial, n_actions: int, n_envs: int, additional_args: dict) -> Dict[str, Any]:
  """
    Sampler for TRPO hyperparams.

    :param trial:
    :return:
    """
  batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128, 256, 512])
  n_steps = trial.suggest_categorical("n_steps", [8, 16, 32, 64, 128, 256, 512, 1024, 2048])
  gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
  learning_rate = trial.suggest_float("learning_rate", 1e-5, 1, log=True)
  line_search_shrinking_factor = trial.suggest_categorical("line_search_shrinking_factor", [0.6, 0.7, 0.8, 0.9])
  n_critic_updates = trial.suggest_categorical("n_critic_updates", [5, 10, 20, 25, 30])
  cg_max_steps = trial.suggest_categorical("cg_max_steps", [5, 10, 20, 25, 30])
  cg_damping = trial.suggest_categorical("cg_damping", [0.5, 0.2, 0.1, 0.05, 0.01])
  target_kl = trial.suggest_categorical("target_kl", [0.1, 0.05, 0.03, 0.02, 0.01, 0.005, 0.001])
  gae_lambda = trial.suggest_categorical("gae_lambda", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0])
  net_arch_type = trial.suggest_categorical("net_arch", ["small", "medium"])
  # Uncomment for gSDE (continuous actions)
  log_std_init = trial.suggest_float("log_std_init", -4, 1)
  # Uncomment for gSDE (continuous action)
  sde_sample_freq = trial.suggest_categorical("sde_sample_freq", [-1, 8, 16, 32, 64, 128, 256])
  # Orthogonal initialization
  ortho_init = False
  ortho_init = trial.suggest_categorical("ortho_init", [False, True])
  activation_fn = trial.suggest_categorical("activation_fn", ["tanh", "relu", "elu", "leaky_relu"])
  activation_fn_name = trial.suggest_categorical("activation_fn", ["tanh", "relu"])
  # lr_schedule = "constant"
  # Uncomment to enable learning rate schedule
  lr_schedule = trial.suggest_categorical("lr_schedule", ["linear", "constant"])
  if lr_schedule == "linear":
    learning_rate = linear_schedule(learning_rate)

  # TODO: account when using multiple envs
  if batch_size > n_steps:
    batch_size = n_steps

  # Independent networks usually work best
  # when not working with images
  net_arch = {
    "small": dict(pi=[64, 64], vf=[64, 64]),
    "medium": dict(pi=[256, 256], vf=[256, 256]),
  }[net_arch_type]

  activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU, "leaky_relu": nn.LeakyReLU}[activation_fn_name]

  n_timesteps = trial.suggest_int("n_timesteps", 100000, 1000000, step=100000)

  n_envs_choice = [2, 4, 6, 8, 10]
  n_envs = trial.suggest_categorical("n_envs", n_envs_choice)

  return {
    "n_timesteps": n_timesteps,
    "n_steps": n_steps,
    "batch_size": batch_size,
    "gamma": gamma,
    "n_envs": n_envs,
    "cg_damping": cg_damping,
    "cg_max_steps": cg_max_steps,
    "line_search_shrinking_factor": line_search_shrinking_factor,
    "n_critic_updates": n_critic_updates,
    "target_kl": target_kl,
    "learning_rate": learning_rate,
    "gae_lambda": gae_lambda,
    "sde_sample_freq": sde_sample_freq,
    "policy_kwargs": dict(
      log_std_init=log_std_init,
      net_arch=net_arch,
      activation_fn=activation_fn,
      ortho_init=ortho_init,
    ),
  }


class PPO(PPO):
  """
    Just a class to ingore extra arguments for compatibility.
    """

  def __init__(
    self,
    policy: Union[str, Type[ActorCriticPolicy]],
    env: Union[GymEnv, str],
    learning_rate: Union[float, Schedule] = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: Union[float, Schedule] = 0.2,
    clip_range_vf: Union[None, float, Schedule] = None,
    normalize_advantage: bool = True,
    ent_coef: float = 0.0,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    use_sde: bool = False,
    sde_sample_freq: int = -1,
    rollout_buffer_class: Optional[Type[RolloutBuffer]] = None,
    rollout_buffer_kwargs: Optional[Dict[str, Any]] = None,
    target_kl: Optional[float] = None,
    stats_window_size: int = 100,
    tensorboard_log: Optional[str] = None,
    policy_kwargs: Optional[Dict[str, Any]] = None,
    verbose: int = 0,
    seed: Optional[int] = None,
    device: Union[th.device, str] = "auto",
    _init_setup_model: bool = True,
    **kwargs,
  ):
    super().__init__(
      policy=policy,
      env=env,
      learning_rate=learning_rate,
      n_steps=n_steps,
      batch_size=batch_size,
      n_epochs=n_epochs,
      gamma=gamma,
      gae_lambda=gae_lambda,
      clip_range=clip_range,
      clip_range_vf=clip_range_vf,
      normalize_advantage=normalize_advantage,
      ent_coef=ent_coef,
      vf_coef=vf_coef,
      max_grad_norm=max_grad_norm,
      use_sde=use_sde,
      sde_sample_freq=sde_sample_freq,
      rollout_buffer_class=rollout_buffer_class,
      rollout_buffer_kwargs=rollout_buffer_kwargs,
      target_kl=target_kl,
      stats_window_size=stats_window_size,
      tensorboard_log=tensorboard_log,
      policy_kwargs=policy_kwargs,
      verbose=verbose,
      seed=seed,
      device=device,
      _init_setup_model=_init_setup_model,
    )
    # Ignore kwargs for compatibility


def sample_ppo_params(trial: optuna.Trial, n_actions: int, n_envs: int, additional_args: dict) -> Dict[str, Any]:
  """
    Sampler for PPO hyperparams.

    :param trial:
    :return:
    """
  batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128, 256, 512])
  n_steps = trial.suggest_categorical("n_steps", [8, 16, 32, 64, 128, 256, 512, 1024, 2048])
  gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
  learning_rate = trial.suggest_float("learning_rate", 1e-5, 1, log=True)
  ent_coef = trial.suggest_float("ent_coef", 0.00000001, 0.1, log=True)
  clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3, 0.4])
  n_epochs = trial.suggest_categorical("n_epochs", [1, 5, 10, 20])
  gae_lambda = trial.suggest_categorical("gae_lambda", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0])
  max_grad_norm = trial.suggest_categorical("max_grad_norm", [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5])
  vf_coef = trial.suggest_float("vf_coef", 0, 1)
  net_arch_type = trial.suggest_categorical("net_arch", ["tiny", "small", "medium"])
  # Uncomment for gSDE (continuous actions)
  # log_std_init = trial.suggest_float("log_std_init", -4, 1)
  # Uncomment for gSDE (continuous action)
  # sde_sample_freq = trial.suggest_categorical("sde_sample_freq", [-1, 8, 16, 32, 64, 128, 256])
  # Orthogonal initialization
  ortho_init = False
  # ortho_init = trial.suggest_categorical('ortho_init', [False, True])
  # activation_fn = trial.suggest_categorical('activation_fn', ['tanh', 'relu', 'elu', 'leaky_relu'])
  activation_fn_name = trial.suggest_categorical("activation_fn", ["tanh", "relu"])
  # lr_schedule = "constant"
  # Uncomment to enable learning rate schedule
  # lr_schedule = trial.suggest_categorical('lr_schedule', ['linear', 'constant'])
  # if lr_schedule == "linear":
  #     learning_rate = linear_schedule(learning_rate)

  # TODO: account when using multiple envs
  if batch_size > n_steps:
    batch_size = n_steps

  # Independent networks usually work best
  # when not working with images
  net_arch = {
    "tiny": dict(pi=[64], vf=[64]),
    "small": dict(pi=[64, 64], vf=[64, 64]),
    "medium": dict(pi=[256, 256], vf=[256, 256]),
  }[net_arch_type]

  activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU, "leaky_relu": nn.LeakyReLU}[activation_fn_name]

  n_timesteps = trial.suggest_int("n_timesteps", 100000, 1000000, step=100000)
  n_envs_choice = [2, 4, 6, 8, 10]
  n_envs = trial.suggest_categorical("n_envs", n_envs_choice)

  return {
    "n_timesteps": n_timesteps,
    "n_steps": n_steps,
    "batch_size": batch_size,
    "gamma": gamma,
    "n_envs": n_envs,
    "learning_rate": learning_rate,
    "ent_coef": ent_coef,
    "clip_range": clip_range,
    "n_epochs": n_epochs,
    "gae_lambda": gae_lambda,
    "max_grad_norm": max_grad_norm,
    "vf_coef": vf_coef,
    # "sde_sample_freq": sde_sample_freq,
    "policy_kwargs": dict(
      # log_std_init=log_std_init,
      net_arch=net_arch,
      activation_fn=activation_fn,
      ortho_init=ortho_init,
    ),
  }
