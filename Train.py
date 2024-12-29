import numpy as np

import torch


class Train:

  @staticmethod
  def batch(model, env, total_timesteps, batch_episodes, **params):
    """
    General training function for reinforcement learning models.

    Args:
        model: The RL model implementing ModelSpec (e.g., TRPO).
        env: The Gymnasium environment to train in.
        total_timesteps: Total number of timesteps for training.
        batch_episodes: Number of episodes per batch for training updates.
        **params: Additional parameters for the training process, passed to model.train.
    """
    timesteps_done = 0
    episode_count = 0
    buffer_states, buffer_actions, buffer_rewards, buffer_next_states, buffer_dones, buffer_successes = [], [], [], [], [], []

    while timesteps_done < total_timesteps:
      state, _ = env.reset()
      done = False
      episode_states, episode_actions, episode_rewards, episode_next_states, episode_dones, episode_successes = [], [], [], [], [], []

      # Collect transitions for one episode
      while not done:
        action = model.select_action(state) * env.action_space.high[0]
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Add to episode transitions
        episode_states.append(state)
        episode_actions.append(action)
        episode_rewards.append(reward)
        episode_next_states.append(next_state)
        episode_dones.append(done)
        episode_successes.append(int(info.get("success", False)))
        success = int(info.get("success", False))

        state = next_state
        timesteps_done += 1

        if timesteps_done >= total_timesteps:
          break

      # Add episode transitions to the buffer
      buffer_states.extend(episode_states)
      buffer_actions.extend(episode_actions)
      buffer_rewards.extend(episode_rewards)
      buffer_next_states.extend(episode_next_states)
      buffer_dones.extend(episode_dones)
      buffer_successes.extend(episode_successes)

      # Log episode metrics
      model.log_episode(
          episode_rewards,
          [0.02] * len(episode_rewards),
          len(episode_rewards),
          success
      )

      episode_count += 1

      # Train the model after completing batch_episodes
      if episode_count >= batch_episodes or timesteps_done >= total_timesteps:
        # Efficient conversion of buffer lists to tensors
        states = torch.tensor(np.array(buffer_states), dtype=torch.float32)
        actions = torch.tensor(np.array(buffer_actions), dtype=torch.float32)
        rewards = torch.tensor(np.array(buffer_rewards), dtype=torch.float32)
        next_states = torch.tensor(np.array(buffer_next_states), dtype=torch.float32)
        dones = torch.tensor(np.array(buffer_dones), dtype=torch.float32)
        successes = torch.tensor(np.array(buffer_successes), dtype=torch.float32)

        # Train the model
        model.train(states, actions, rewards, next_states, dones, successes, **params)

        # Clear the buffer after training
        buffer_states, buffer_actions, buffer_rewards, buffer_next_states, buffer_dones = [], [], [], [], []
        episode_count = 0

        # Print metrics after each batch
        metrics = model.get_metrics()
        print(f"Timesteps: {timesteps_done}/{total_timesteps}, Metrics: {metrics}")
