import os
import tempfile
import collections
import numpy as np
import gymnasium as gym
from stable_baselines3.common import vec_env
from imitation.algorithms import bc, dagger
from imitation.algorithms.dagger import SimpleDAggerTrainer
from imitation.policies.base import NonTrainablePolicy
from imitation.util import util
from stable_baselines3.common.evaluation import evaluate_policy
from imitation.policies import interactive
from datetime import datetime
from datasets import Dataset
from imitation.policies.serialize import load_policy, save_stable_model
from pathlib import Path
from imitation.util.util import make_vec_env


import gymnasium as gym
from stable_baselines3.common import vec_env

env = vec_env.DummyVecEnv([
    lambda: gym.wrappers.TimeLimit(
        gym.make("CartPole-v1", render_mode="human"),  # must include render_mode
        max_episode_steps=200,
    )
])
env.seed(0)

import torch
from imitation.algorithms.dagger import SimpleDAggerTrainer
from torch.serialization import safe_globals

checkpoint_path = "/home/pouyan/phd/imitation_learning/imitation/examples/dagger/logs/2025_06_02_11_40/policy-latest.pt"

with safe_globals([SimpleDAggerTrainer]):
    trainer = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

# policy = trainer.policy

obs = env.reset()
done = False

while True:
    action, _ = trainer.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()  




# use "Checkpoint.pt" with "trainer.policy" OR use "policy-latest.pt" without it and add "trainer.predict" directly.
