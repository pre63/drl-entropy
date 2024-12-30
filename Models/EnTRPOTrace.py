import torch
import torch.nn as nn
import torch.optim as optim
from Models.EnTRPO import EnTRPO


class EnTRPOTrace(EnTRPO):
  """
      Experimental TRPO model with eligibility traces for advantage calculation.
  """

  def __init__(self, device=None, **params):
    super(EnTRPOTrace, self).__init__(device=device, **params)
    self.trace_decay = params.get("trace_decay", 0.9)  # Decay rate for eligibility traces

  def compute_advantages_with_traces(self, rewards, values, next_values, dones, successes, gamma, lambd):
    """
    Compute advantages using eligibility traces, factoring in successes.
    """
    advantages = []
    eligibility_trace = 0
    for t in reversed(range(len(rewards))):
      delta = rewards[t] + gamma * next_values[t] * (1 - dones[t]) - values[t]
      # Amplify eligibility trace for successes, dampen for failures
      eligibility_trace = (delta + gamma * lambd * (1 - dones[t]) * eligibility_trace) * (1 + successes[t])
      advantages.insert(0, eligibility_trace)
    return torch.tensor(advantages, dtype=torch.float32).to(self.device)

  def train(self, states, actions, rewards, next_states, dones, successes, **params):
    """
    Override train to include eligibility-trace-based advantage calculation.
    """
    gamma = params.get("gamma", 0.99)
    lambd = params.get("lambd", 0.95)

    # Move tensors to device
    states = states.to(self.device)
    next_states = next_states.to(self.device)
    actions = actions.to(self.device)
    rewards = rewards.to(self.device)
    dones = dones.to(self.device)
    successes = successes.to(self.device)

    # Compute values and advantages
    values = self.critic(states).squeeze()
    next_values = self.critic(next_states).squeeze()

    # Calculate advantages using eligibility traces
    advantages = self.compute_advantages_with_traces(
        rewards, values.detach(), next_values.detach(), dones, successes, gamma, lambd
    )
    targets = rewards + gamma * next_values * (1 - dones)

    # Train critic
    self.train_critic(states, targets)

    # Clone the current actor to old_actor
    self.clone_actor()

    # Train actor
    self.train_actor(states, actions, advantages)

    self.log_loss(self.actor_loss, self.critic_loss, self.kl_divergence_loss)


if __name__ == "__main__":
  import gymnasium as gym
  from Experiment import Experiment
  from Train import Train
  from itertools import product

  # Initialize environment
  from Environments.Pendulum import make
  env = make()
  state_dim = env.observation_space.shape[0]
  action_dim = env.action_space.shape[0]

  # Define parameter grid
  param_grid = {
      "gamma": [0.95, 0.99],
      "lambd": [0.9, 0.95],
      "critic_alpha": [1e-3, 1e-4],
      "hidden_sizes": [[64, 64], [128, 128]],
      "kl_threshold": [1e-2, 5e-3],
      "entropy_coeff": [0.01, 0.001],
      "trace_decay": [0.8, 0.9],
      "state_dim": [state_dim],  # Fixed for the environment
      "action_dim": [action_dim],  # Fixed for the environment
  }

  # Generate all combinations of parameters
  param_combinations = list(product(*param_grid.values()))

  # Map parameter names to combinations
  param_keys = list(param_grid.keys())

  # Training parameters
  batch_size = 128
  num_batch = 1000

  for i, param_values in enumerate(param_combinations):
    # Create parameter dictionary for this combination
    model_params = dict(zip(param_keys, param_values))

    # Initialize model
    model = EnTRPOTrace(**model_params)

    # Define total timesteps for training
    total_timesteps = batch_size * num_batch

    print(f"Starting experiment {i + 1}/{len(param_combinations)} with params: {model_params}")

    # Train the model
    Train.batch(model, env, total_timesteps, batch_size, **model_params)

    # Save the experiment
    Experiment.save(model, env)

    print(f"Experiment {i + 1}/{len(param_combinations)} completed.")
