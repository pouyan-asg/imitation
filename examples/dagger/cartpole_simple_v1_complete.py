"""modified implementation of DAgger for CartPole-v1"""
"""UNCOMPLETED"""
"""check different gym wrappers for better data collection. I was trying to implement RecordEpisodeStatistics wrapper which was kind of successfull.
check DAgger.py in which there is a "print(infos[0])". after each round of training the result is as follows: 
{'episode': {'r': array([500.], dtype=float32), 'l': array([500], dtype=int32), 't': array([10.717529], dtype=float32)}, 
'TimeLimit.truncated': True, 'terminal_observation': array([-0.6096335 ,  0.05089916,  0.00276603, -0.12746839], dtype=float32)}
"""

import os
import numpy as np
import gymnasium as gym
from imitation.policies.serialize import load_policy
from imitation.util.util import make_vec_env
from imitation.algorithms import bc
from imitation.algorithms.dagger import SimpleDAggerTrainer
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common import vec_env
from datetime import datetime

# -------Initialize----------
root_path = "/home/pouyan/phd/imitation_learning/imitation/examples/dagger/logs"
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
logs_dir = os.path.join(root_path, f"{timestamp}")

rn_seed = np.random.default_rng(0)
env_max_episode_steps = 500
n_eval_episodes = 50

# -------Create a vectorized environment----------
# env = vec_env.DummyVecEnv([
#     lambda: gym.wrappers.TimeLimit(gym.make("CartPole-v1", render_mode="human"), 
#                                    max_episode_steps=env_max_episode_steps)])  # render_mode="human"

env = vec_env.DummyVecEnv([
    lambda: gym.wrappers.TimeLimit(
        gym.wrappers.RecordEpisodeStatistics(
            gym.make("CartPole-v1", render_mode="human")
        ),
        max_episode_steps=env_max_episode_steps
    )
])


env.seed(0)

# -------Create expert trajectories from a pre-trained policy----------
expert = load_policy(
    policy_type="ppo-huggingface",
    organization="HumanCompatibleAI",
    env_name="CartPole-v1",
    venv=env,
)

# -------Prepare expert and algorithm----------
bc_trainer = bc.BC(
    observation_space=env.observation_space,
    action_space=env.action_space,
    rng=rn_seed,
)

dagger_trainer = SimpleDAggerTrainer(
    venv=env,
    scratch_dir=logs_dir,
    expert_policy=expert,
    bc_trainer=bc_trainer,
    rng=rn_seed,
)

# -------Training loop (code stops here until training will be finished)----------
dagger_trainer.train(2000)

# -------Reward evaluation----------
reward_after_training, _ = evaluate_policy(
    model=dagger_trainer.policy, 
    env=env, 
    n_eval_episodes=n_eval_episodes)
print(f"\033[91m\nReward after training: {reward_after_training}\033[0m")

# -------Saving trained policy----------
print("\033[92m\nSaving the trained policy...\n\033[0m")
dagger_trainer.save_trainer()
dagger_trainer.policy.save(os.path.join(logs_dir, "final_policy.pt"))


