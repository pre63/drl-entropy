

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
      'policy_delay': trial.suggest_int('policy_delay', 1, 5),
      'ent_coef': trial.suggest_categorical('ent_coef', ['auto', 0.01, 0.1, 1.0]),
      'target_entropy': trial.suggest_categorical('target_entropy', ['auto', -1.0, -0.5, 0.0]),
      'use_sde': trial.suggest_categorical('use_sde', [True, False]),
      'sde_sample_freq': trial.suggest_int('sde_sample_freq', -1, 99, step=10),
      'use_sde_at_warmup': trial.suggest_categorical('use_sde_at_warmup', [True, False]),
      'total_timesteps': total_timesteps,
  }
