import torch as th
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, TypeVar, Union

from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, Schedule


from sb3_contrib import TRPO


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
