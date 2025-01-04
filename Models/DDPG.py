


def optimal(trial, total_timesteps):
  return {
      'policy': 'MlpPolicy',
      'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, step=1e-5),
      'qf_learning_rate': trial.suggest_float('qf_learning_rate', 1e-5, 1e-2, log=True),
      'buffer_size': trial.suggest_int('buffer_size', 10000, 1000000, step=10000),
      'learning_starts': trial.suggest_int('learning_starts', 100, 10000, step=100),
      'batch_size': trial.suggest_categorical('batch_size', [64, 128, 256, 512]),
      'tau': trial.suggest_float('tau', 0.001, 0.1, step=0.001),
      'gamma': trial.suggest_float('gamma', 0.9, 0.99, step=0.01),
      'train_freq': trial.suggest_int('train_freq', 1, 10),
      'gradient_steps': trial.suggest_int('gradient_steps', 1, 10),
      'total_timesteps': total_timesteps,
  }

