"""implementation of a simple DAgger for CarRacing-v2 environment using a pre-trained expert policy."""

import os
import numpy as np
import gymnasium as gym
from imitation.policies.serialize import load_policy
from imitation.util.util import make_vec_env
from imitation.algorithms import bc
from imitation.data import rollout
from imitation.algorithms.dagger import SimpleDAggerTrainer
from imitation.data.wrappers import RolloutInfoWrapper
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from datetime import datetime

# more info: https://gymnasium.farama.org/environments/box2d/car_racing/

# -------Initialize----------
root_path = "/home/pouyan/phd/imitation_learning/imitation/examples/dagger/logs"
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
logs_dir = os.path.join(root_path, f"{timestamp}")

rn_seed = np.random.default_rng(0)
train_min_episodes = 2
train_min_timesteps = 10
n_eval_episodes = 2
training_epochs = 1

sample_until = rollout.make_sample_until(
    min_timesteps=train_min_timesteps,
    min_episodes=train_min_episodes,)

# -------Create a vectorized environment----------
"""since the policy's observation space is CHW format, we need to 
transpose the our environemnt's space from HWC to CHW by VecTransposeImage"""

env = VecTransposeImage(DummyVecEnv([lambda: RolloutInfoWrapper(gym.make("CarRacing-v2", 
                                    render_mode="human",
                                    lap_complete_percent=0.95,
                                    domain_randomize=False,
                                    continuous=True))]))

env.seed(0)

# -------- Create the expert policy---------
expert = load_policy(
    policy_type="ppo-huggingface",
    organization="igpaub",
    env_name="CarRacing-v2",
    venv=env,
)

print(f"env.observation_space: {env.observation_space}")
print(f"env.action_space: {env.action_space}")
print(f"expert.observation_space: {expert.observation_space}")
print(f"expert.action_space: {expert.action_space}")

# -------quick evaluation of the expert policy----------
# reward, _ = evaluate_policy(model=expert, env=env, n_eval_episodes=n_eval_episodes, deterministic=True)
# print(f"Expert Reward ==> maximum:1000, my:{reward} \n")

# -------Prepare expert and algorithm----------
bc_trainer = bc.BC(
    observation_space=env.observation_space,
    action_space=env.action_space,
    rng=rn_seed,
)

# dagger_trainer = SimpleDAggerTrainer(
#     venv=env,
#     scratch_dir=logs_dir,
#     expert_policy=expert,
#     bc_trainer=bc_trainer,
#     rng=rn_seed,
# )

# # -------Training loop (code stops here until training will be finished)----------
# dagger_trainer.train(2000)

# # -------Reward evaluation----------
# reward_after_training, _ = evaluate_policy(
#     model=dagger_trainer.policy, 
#     env=env, 
#     n_eval_episodes=n_eval_episodes)
# print(f"\033[91m\nReward after training: {reward_after_training}\033[0m")

# # -------Saving trained policy----------
# print("\033[92m\nSaving the trained policy...\n\033[0m")
# dagger_trainer.save_trainer()
# dagger_trainer.policy.save(os.path.join(logs_dir, "final_policy.pt"))

# dagger_trainer = SimpleDAggerTrainer(
#     venv=env,
#     scratch_dir=logs_dir,
#     expert_policy=expert,
#     bc_trainer=bc_trainer,
#     rng=rn_seed,
# )

# # -------Training loop (code stops here until training will be finished)----------
# dagger_trainer.train(2000)

# # -------Reward evaluation----------
# reward_after_training, _ = evaluate_policy(
#     model=dagger_trainer.policy, 
#     env=env, 
#     n_eval_episodes=n_eval_episodes)
# print(f"\033[91m\nReward after training: {reward_after_training}\033[0m")

# # -------Saving trained policy----------
# print("\033[92m\nSaving the trained policy...\n\033[0m")
# dagger_trainer.save_trainer()
# dagger_trainer.policy.save(os.path.join(logs_dir, "final_policy.pt"))


