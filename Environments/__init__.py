from Environments.RocketLander import RocketLander

from gymnasium.envs.registration import register

register(
    id="RocketLander-v0",
    entry_point="Environments.RocketLander:RocketLander",
    max_episode_steps=1000,
    reward_threshold=200,
)
