import torch.nn as nn


class QuantileValueNetwork(nn.Module):
  def __init__(self, state_dim, n_quantiles=25, net_arch=[64, 64], activation_fn=nn.ReLU):
    super().__init__()
    self.n_quantiles = n_quantiles
    layers = []
    input_dim = state_dim
    for units in net_arch:
      layers.append(nn.Linear(input_dim, units))
      layers.append(activation_fn())
      input_dim = units
    layers.append(nn.Linear(input_dim, n_quantiles))
    self.network = nn.Sequential(*layers)

  def forward(self, state):
    return self.network(state)


def optimize_hyperparameters(trial):
  """
    Optuna hyperparameter optimization function
    """

  net_arch_type = trial.suggest_categorical("net_arch", ["small", "medium", "large"])

  # Neural network architecture configuration
  net_arch = {
    "small": [64, 64],
    "medium": [256, 256],
    "large": [400, 300],
  }[net_arch_type]

  activation_fn_name = trial.suggest_categorical("activation_fn", ["tanh", "relu"])
  activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation_fn_name]

  n_quantiles = trial.suggest_categorical("n_quantiles", [10, 25, 50, 100])

  return {"net_arch": net_arch, "activation_fn": activation_fn, "n_quantiles": n_quantiles}
