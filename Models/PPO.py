

def optimal(trial, total_timesteps):
  return {
      'policy': 'MlpPolicy',
      'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, step=1e-5),
      'n_steps': trial.suggest_int('n_steps', 128, 2048, step=128),
      'batch_size': trial.suggest_categorical("batch_size", [64, 128, 256, 512]),
      'n_epochs': trial.suggest_int('n_epochs', 1, 20),
      'gamma': trial.suggest_float('gamma', 0.9, 0.99, step=0.01),
      'gae_lambda': trial.suggest_float('gae_lambda', 0.8, 1.0, step=0.01),
      'clip_range_vf': trial.suggest_float('clip_range_vf', 0.1, 0.5, step=0.1),
      'normalize_advantage': trial.suggest_categorical('normalize_advantage', [True, False]),
      'ent_coef': trial.suggest_float('ent_coef', 1e-5, 1e-1, log=True),
      'vf_coef': trial.suggest_float('vf_coef', 0.1, 1.0, step=0.1),
      'max_grad_norm': trial.suggest_float('max_grad_norm', 0.1, 5.0, step=0.1),
      'use_sde': trial.suggest_categorical('use_sde', [True, False]),
      'sde_sample_freq': trial.suggest_int('sde_sample_freq', -1, 99, step=10),
      'target_kl': trial.suggest_float('target_kl', 0.01, 0.2, step=0.01),
      'total_timesteps': total_timesteps
  }
