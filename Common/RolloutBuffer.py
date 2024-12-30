import torch


class RolloutBuffer:
  """
  Rollout buffer for storing trajectories in on-policy algorithms like TRPO,
  without a fixed size constraint. All transitions are appended to lists
  and converted to tensors in get().

  Args:
      device (torch.device): Device to store the buffer data.
  """

  def __init__(self, device):
    self.device = device
    self.reset()

  def reset(self):
    self.states = []
    self.actions = []
    self.rewards = []
    self.next_states = []
    self.dones = []
    self.successes = []

  def add(self, state, action, reward, next_state, done, success):
    if isinstance(state, torch.Tensor):
      state = state.clone().detach().to(self.device)
    else:
      state = torch.tensor(state, dtype=torch.float32, device=self.device)

    if isinstance(action, torch.Tensor):
      action = action.clone().detach().to(self.device)
    else:
      action = torch.tensor(action, dtype=torch.float32, device=self.device)

    if isinstance(next_state, torch.Tensor):
      next_state = next_state.clone().detach().to(self.device)
    else:
      next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device)

    reward = float(reward)
    done = bool(done)
    success = bool(success)

    self.states.append(state)
    self.actions.append(action)
    self.rewards.append(reward)
    self.next_states.append(next_state)
    self.dones.append(done)
    self.successes.append(success)

  def get(self):
    states = torch.stack(self.states).to(self.device)
    actions = torch.stack(self.actions).to(self.device)
    rewards = torch.tensor(self.rewards, dtype=torch.float32, device=self.device)
    next_states = torch.stack(self.next_states).to(self.device)
    dones = torch.tensor(self.dones, dtype=torch.bool, device=self.device)
    successes = torch.tensor(self.successes, dtype=torch.bool, device=self.device)

    self.reset()
    return {
        "states": states,
        "actions": actions,
        "rewards": rewards,
        "next_states": next_states,
        "dones": dones,
        "successes": successes,
    }

  def __len__(self):
    return len(self.states)
