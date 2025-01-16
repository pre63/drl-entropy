import argparse
import importlib
import os
import pickle as pkl
import time
import warnings
from collections import OrderedDict
from pathlib import Path
from pprint import pprint
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import optuna
# Register custom envs
import rl_zoo3.import_envs  # noqa: F401
import torch as th
import yaml
from gymnasium import spaces
from huggingface_sb3 import EnvironmentName
from optuna.pruners import BasePruner, MedianPruner, NopPruner, SuccessiveHalvingPruner
from optuna.samplers import BaseSampler, RandomSampler, TPESampler
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState
from optuna.visualization import plot_optimization_history, plot_param_importances
from rl_zoo3.callbacks import SaveVecNormalizeCallback, TrialEvalCallback
from rl_zoo3.exp_manager import ExperimentManager
from rl_zoo3.hyperparams_opt import HYPERPARAMS_SAMPLER
from rl_zoo3.utils import (ALGOS, get_callback_list, get_class_by_name, get_latest_run_id,
                           get_wrapper_class, linear_schedule)
from sb3_contrib.common.vec_env import AsyncEval
# For using HER with GoalEnv
from stable_baselines3 import HerReplayBuffer
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import (BaseCallback, CheckpointCallback, EvalCallback,
                                                ProgressBarCallback)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.preprocessing import is_image_space, is_image_space_channels_first
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike  # noqa: F401
from stable_baselines3.common.utils import constant_fn
from stable_baselines3.common.vec_env import (DummyVecEnv, SubprocVecEnv, VecEnv, VecFrameStack,
                                              VecNormalize, VecTransposeImage, is_vecenv_wrapped)
# For custom activation fn
from torch import nn as nn

class ExperimentManager(ExperimentManager):
  def learn(self, model: BaseAlgorithm) -> None:
    """
    Train the model and capture training rewards and timesteps.
    :param model: an initialized RL model
    """
    kwargs: Dict[str, Any] = {}
    if self.log_interval > -1:
      kwargs = {"log_interval": self.log_interval}

    if len(self.callbacks) > 0:
      kwargs["callback"] = self.callbacks

    # Variables to store training rewards and timesteps
    self.training_rewards = []

    try:
      # Custom callback to log rewards at the end of each episode
      def reward_logger(locals_, globals_):
        infos = locals_['infos']  # Access environment info

        # Extract episode rewards from 'infos'
        episode_rewards = [info['episode']['r'] for info in infos if 'episode' in info]
        self.training_rewards.extend(episode_rewards)
        return True

      # Use the custom reward logger
      model.learn(self.n_timesteps, callback=reward_logger, **kwargs)
    except KeyboardInterrupt:
      # Allow saving the model if interrupted
      pass
    finally:
      # Clean progress bar and release resources
      if len(self.callbacks) > 0:
        self.callbacks[0].on_training_end()
      try:
        assert model.env is not None
        model.env.close()
      except EOFError:
        pass

  def create_callbacks(self):
    if self.show_progress:
      self.callbacks.append(ProgressBarCallback())

    if self.save_freq > 0:
      # Account for the number of parallel environments
      self.save_freq = max(self.save_freq // self.n_envs, 1)
      self.callbacks.append(
          CheckpointCallback(
              save_freq=self.save_freq,
              save_path=self.save_path,
              name_prefix="rl_model",
              verbose=1,
          )
      )

    # Skip evaluation-related callbacks
