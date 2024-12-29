import torch
import torch.nn as nn


class GaussianPolicy(nn.Module):
  """
      Gaussian policy network for continuous action spaces.
  """

  def __init__(self, input_dim, hidden_layer_sizes, output_dim):
    super(GaussianPolicy, self).__init__()
    layers = []
    for size in hidden_layer_sizes:
      layers.append(nn.Linear(input_dim, size))
      layers.append(nn.Tanh())
      input_dim = size
    self.hidden_layers = nn.Sequential(*layers)
    self.mean_layer = nn.Linear(input_dim, output_dim)
    self.log_std = nn.Parameter(torch.zeros(1, output_dim))

  def forward(self, state):
    hidden_output = self.hidden_layers(state)
    mean = self.mean_layer(hidden_output)
    std = torch.exp(self.log_std)
    return mean, std
