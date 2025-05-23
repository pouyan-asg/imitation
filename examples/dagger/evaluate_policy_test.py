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


# rn_seed = np.random.default_rng(0)
# # env = make_vec_env(
# #     "seals:seals/CartPole-v0",
# #     rng=rn_seed,
# #     n_envs=1,
# # )
# env = vec_env.DummyVecEnv([
#     lambda: gym.wrappers.TimeLimit(gym.make("CartPole-v1", 
#                                             render_mode="human"), 
#                                             max_episode_steps=5)
#                                             ])
# env.seed(0)

# # expert_test = load_policy(
# #     policy_type="ppo",        # because you used PPO to train this policy
# #     venv=env,                 # required so it knows the obs/action spaces
# #     path="/home/pouyan/phd/imitation_learning/imitation/examples/dagger/my_saved_policy/model.zip",   # directory that contains model.zip
# # )

# # rews, _ = evaluate_policy(model=expert_test, env=env, n_eval_episodes=10)
# # print(f"\nReward after training: {rews}")


# # from huggingface_sb3 import load_from_hub
# # checkpoint = load_from_hub(
# # 	repo_id="HumanCompatibleAI/ppo-CartPole-v1",
# # 	filename="ppo-CartPole-v1.zip",
# # )
# # print(checkpoint)



# # from imitation.policies.serialize import load_policy
# # policy = load_policy("policy-latest.pt", env)



# dagger.reconstruct_trainer(scratch_dir='/home/pouyan/phd/imitation_learning/imitation/examples/dagger/logs/directpry', venv=env)


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

checkpoint_path = "/home/pouyan/phd/imitation_learning/imitation/examples/dagger/logs/2025_05_22_11_59/policy-latest.pt"

with safe_globals([SimpleDAggerTrainer]):
    trainer = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

# policy = trainer.policy

obs = env.reset()
done = False

while True:
    action, _ = trainer.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()  




# use Checkpoint with trainer.policy OR use policy without it and add trainer.predict directly.
