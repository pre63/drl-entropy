import time
import numpy as np

import torch


class Train:
  """
  Training logic that accumulates a buffer of transitions for a fixed step count
  (batch_size) and then calls the model's train() method. Continues until
  total_timesteps is reached.
  """

  @staticmethod
  def batch(model, env, total_timesteps, batch_size, **params):
    timesteps_done = 0

    # Temporary buffers
    buffer_states = []
    buffer_actions = []
    buffer_rewards = []
    buffer_next_states = []
    buffer_dones = []
    buffer_successes = []

    while timesteps_done < total_timesteps:
      state, _ = env.reset()
      episode_rewards = []
      episode_step_times = []
      episode_success_flags = []
      done = False

      while not done:
        step_start = time.time()

        # Possibly scale your action by the environment's max
        action = model.select_action(state) * env.action_space.high[0]
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Log the environment step to track per-step reward, timing
        step_duration = time.time() - step_start
        model.log_step(reward, step_duration)

        # Collect transition
        buffer_states.append(state)
        buffer_actions.append(action)
        buffer_rewards.append(reward)
        buffer_next_states.append(next_state)
        buffer_dones.append(done)
        success_flag = int(info.get("success", 0))
        buffer_successes.append(success_flag)

        # For episode-level logging
        episode_rewards.append(reward)
        episode_step_times.append(step_duration)
        episode_success_flags.append(success_flag)

        state = next_state
        timesteps_done += 1

        # If we've collected enough transitions, train:
        if len(buffer_states) >= batch_size or timesteps_done >= total_timesteps:
          Train._train_on_batch(
              model,
              buffer_states,
              buffer_actions,
              buffer_rewards,
              buffer_next_states,
              buffer_dones,
              buffer_successes,
              **params
          )

          buffer_states.clear()
          buffer_actions.clear()
          buffer_rewards.clear()
          buffer_next_states.clear()
          buffer_dones.clear()
          buffer_successes.clear()

        if done or timesteps_done >= total_timesteps:
          break

      # Episode is over, record aggregated metrics for the episode
      success = episode_success_flags[-1] if episode_success_flags else 0
      model.log_episode(
          episode_rewards=sum(episode_rewards),
          avg_step_time=float(np.mean(episode_step_times)) if episode_step_times else 0.0,
          success=success,
          episode_steps=len(episode_rewards),
      )

      # Print an occasional summary
      # In a real codebase, you might do this every N episodes or M timesteps
      summary = model.get_training_summary()
      print(f"[Training Summary] {summary}")

  @staticmethod
  def _train_on_batch(model, states, actions, rewards, next_states, dones, successes, **params):
    # Convert transitions to tensors
    st = torch.tensor(np.array(states), dtype=torch.float32)
    ac = torch.tensor(np.array(actions), dtype=torch.float32)
    rw = torch.tensor(np.array(rewards), dtype=torch.float32)
    ns = torch.tensor(np.array(next_states), dtype=torch.float32)
    dn = torch.tensor(np.array(dones), dtype=torch.float32)
    sc = torch.tensor(np.array(successes), dtype=torch.float32)

    # Call model's train with these transitions
    model.train(
        states=st,
        actions=ac,
        rewards=rw,
        next_states=ns,
        dones=dn,
        successes=sc,
        **params
    )

  @staticmethod
  def eval(model, env, episodes=5):
    eval_rewards = []
    eval_successes = []
    for ep_idx in range(episodes):
      state, _ = env.reset()
      total_reward = 0
      done = False
      success_flag = 0
      while not done:
        action = model.select_action(state) * env.action_space.high[0]
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        success_flag = int(info.get("success", 0))

      eval_rewards.append(total_reward)
      eval_successes.append(success_flag)

    # Log and print evaluation
    model.log_evaluation(eval_rewards, eval_successes)

    print(f"Rewards: {sum(eval_rewards)} / {episodes} episodes")
    print(f"Successes: {sum(eval_successes)} / {episodes} episodes")
    return eval_rewards, eval_successes
