import argparse

import rl_zoo3
import rl_zoo3.train
from rl_zoo3.train import train
from rl_zoo3.enjoy import enjoy

from sbx import DDPG, DQN, PPO, SAC, TD3, TQC, CrossQ

rl_zoo3.ALGOS["ddpg"] = DDPG
rl_zoo3.ALGOS["dqn"] = DQN
rl_zoo3.ALGOS["sac"] = SAC
rl_zoo3.ALGOS["ppo"] = PPO
rl_zoo3.ALGOS["td3"] = TD3
rl_zoo3.ALGOS["tqc"] = TQC
rl_zoo3.ALGOS["crossq"] = CrossQ
rl_zoo3.train.ALGOS = rl_zoo3.ALGOS
rl_zoo3.exp_manager.ALGOS = rl_zoo3.ALGOS

# python zoo/train.py --algo tqc --env LunarLanderContinuous-v3 -params train_freq:4 gradient_steps:4 -P
if __name__ == "__main__":
  train()
