"""implementation of Behavior Cloning (BC) on the CarRacing-v2 environment using a pre-trained expert policy."""

import os
import numpy as np
import gymnasium as gym
from stable_baselines3.common.evaluation import evaluate_policy
from imitation.policies.serialize import load_policy
from imitation.util.util import make_vec_env
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.data import rollout
from imitation.algorithms import bc
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from datetime import datetime

"""
Notes:
- Expert Policy: A trained model with PPO from HuggingFace: https://huggingface.co/igpaub/ppo-CarRacing-v2
- CarRacing-v2 environment:
    -- Action Space: Box([left/right, gas, break], max=1.0, shape=(3,), type=float32)
    -- Observation Space: Box(pxmin=0, pxamax=255, (h=96, w=96, c=3), type=uint8)
    -- reward: -0.1 every frame and +1000/N for every track tile visited
    -- more info: https://gymnasium.farama.org/environments/box2d/car_racing/
    
"""

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
reward, _ = evaluate_policy(model=expert, env=env, n_eval_episodes=n_eval_episodes, deterministic=True)
print(f"Expert Reward ==> maximum:1000, my:{reward} \n")

# -------- Create expert trajectories from a pre-trained policy- ---------
rollouts = rollout.rollout(
    policy=expert,
    venv=env,
    sample_until=sample_until,
    rng=rn_seed,)

trajectories = rollout.flatten_trajectories(rollouts)

print(f"""Number of episodes: {train_min_episodes}
The `rollout` function:  size: {len(rollouts)}, type: {type(rollouts[0])}.
After flattening (unpacking): size: {len(trajectories)}, type: {type(trajectories)} transitions.
Transitions object contains arrays for: {', '.join(trajectories.__dict__.keys())}."
""")

# -------- Train & Test ---------
bc_trainer = bc.BC(
    observation_space=env.observation_space,
    action_space=env.action_space,
    demonstrations=trajectories,
    rng=rn_seed,
)

reward_before_training, _ = evaluate_policy(model=bc_trainer.policy, env=env, n_eval_episodes=n_eval_episodes, render=True,)
print(f"\n Reward before training: {reward_before_training}")

print("Training a policy using Behavior Cloning")
bc_trainer.train(n_epochs=training_epochs)

reward_after_training, _ = evaluate_policy(model=bc_trainer.policy, env=env, n_eval_episodes=n_eval_episodes, render=True,)
print(f"\n Reward after training: {reward_after_training}")
