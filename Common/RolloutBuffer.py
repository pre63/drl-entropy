import torch
from collections import namedtuple

class RolloutBuffer:
    """
    Rollout buffer for storing trajectories in on-policy algorithms like TRPO.

    Args:
        buffer_size (int): Maximum number of steps to store in the buffer.
        state_dim (int): Dimensionality of the state space.
        action_dim (int): Dimensionality of the action space.
        device (torch.device): Device to store the buffer data.
    """
    def __init__(self, buffer_size, state_dim, action_dim, device):
        self.buffer_size = buffer_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

        self.reset()

    def reset(self):
        """Reset the buffer to an empty state."""
        self.states = torch.zeros((self.buffer_size, self.state_dim), device=self.device)
        self.actions = torch.zeros((self.buffer_size, self.action_dim), device=self.device)
        self.rewards = torch.zeros(self.buffer_size, device=self.device)
        self.next_states = torch.zeros((self.buffer_size, self.state_dim), device=self.device)
        self.dones = torch.zeros(self.buffer_size, dtype=torch.bool, device=self.device)

        self.size = 0

    def add(self, state, action, reward, next_state, done):
        """
        Add a transition to the buffer.

        Args:
            state (np.array or torch.Tensor): The current state.
            action (np.array or torch.Tensor): The action taken.
            reward (float): The reward received.
            next_state (np.array or torch.Tensor): The next state.
            done (bool): Whether the episode is done.
        """
        if self.size >= self.buffer_size:
            raise ValueError("Buffer overflow: Attempted to add more transitions than buffer size.")

        idx = self.size
        self.states[idx] = torch.tensor(state, dtype=torch.float32, device=self.device)
        self.actions[idx] = torch.tensor(action, dtype=torch.float32, device=self.device)
        self.rewards[idx] = torch.tensor(reward, dtype=torch.float32, device=self.device)
        self.next_states[idx] = torch.tensor(next_state, dtype=torch.float32, device=self.device)
        self.dones[idx] = torch.tensor(done, dtype=torch.bool, device=self.device)

        self.size += 1

    def get(self):
        """
        Retrieve all stored transitions and reset the buffer.

        Returns:
            dict: A dictionary containing the collected transitions.
        """
        data = {
            "states": self.states[:self.size],
            "actions": self.actions[:self.size],
            "rewards": self.rewards[:self.size],
            "next_states": self.next_states[:self.size],
            "dones": self.dones[:self.size],
        }
        self.reset()
        return data

    def __len__(self):
        """Return the current size of the buffer."""
        return self.size
