"""
    This script is to benchmark the TRPO algorithm from Stable Baselines3 Contrib on the environments.
"""

import gymnasium as gym
from sb3_contrib import TRPO
from Environments.LunarLander import make

if __name__ == "__main__":

  env = make()

  model = TRPO('MlpPolicy', env, verbose=1)

  total_timesteps = 10000000
  model.learn(total_timesteps=total_timesteps)

  eval_timesteps = total_timesteps
  t = 0

  successes = []
  while t < eval_timesteps:
    state, _ = env.reset()
    done = False
    while not done:
      action, _ = model.predict(state)
      state, reward, terminated, truncated, info = env.step(action)

      done = terminated or truncated
      t += 1
    successes.append(info["success"])

  print(f"Environment: {env.name}")
  print(f"Success rate: {sum(successes) / len(successes)}")

  successes = []

  env = make(render_mode='human')
  state, _ = env.reset()
  done = False

  while not done:
    env.render()
    action, _ = model.predict(state)
    state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

  print(f"State: {state}, Reward: {reward}, Done: {done}, Info: {info}")


"""
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 68.2     |
|    ep_rew_mean            | -212     |
| time/                     |          |
|    fps                    | 482      |
|    iterations             | 49       |
|    time_elapsed           | 207      |
|    total_timesteps        | 100352   |
| train/                    |          |
|    explained_variance     | 0.919    |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.00626  |
|    learning_rate          | 0.001    |
|    n_updates              | 48       |
|    policy_objective       | 0.0193   |
|    std                    | 0.743    |
|    value_loss             | 587      |
----------------------------------------
Environment: Pendulum-v1
Success rate: 0.9954248366013072
State: [ 0.99910307 -0.04234432  0.0348728 ], Reward: 0, Done: True, Info: {'success': True}
---
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 17.1     |
|    ep_rew_mean            | 0        |
| time/                     |          |
|    fps                    | 697      |
|    iterations             | 49       |
|    time_elapsed           | 143      |
|    total_timesteps        | 100352   |
| train/                    |          |
|    explained_variance     | 0.104    |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.00962  |
|    learning_rate          | 0.001    |
|    n_updates              | 48       |
|    policy_objective       | 0.0199   |
|    std                    | 0.742    |
|    value_loss             | 3.34e-05 |
----------------------------------------
Environment: RandomWalk
Success rate: 0.0
State: [0.], Reward: 0.0, Done: True, Info: {'success': False}
---

----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 811      |
|    ep_rew_mean            | -50.1    |
| time/                     |          |
|    fps                    | 741      |
|    iterations             | 49       |
|    time_elapsed           | 135      |
|    total_timesteps        | 100352   |
| train/                    |          |
|    explained_variance     | 0.67     |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.00712  |
|    learning_rate          | 0.001    |
|    n_updates              | 48       |
|    policy_objective       | 0.0238   |
|    std                    | 0.774    |
|    value_loss             | 60.9     |
----------------------------------------
Environment: LunarLander-v3
Success rate: 0.019230769230769232
State: [-0.39279193  0.02852891  0.          0.         -0.2508642   0.
  1.          1.        ], Reward: 100, Done: True, Info: {'success': False}
"""
