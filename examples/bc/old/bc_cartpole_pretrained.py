"""
Notes:
- Expert Policy: A trained model that can decide what action to take in a given state.
    -- When we train a policy (e.g., PPO), we only save the model weights — 
        not the entire log of everything it did.
- Trajectory Rollout: A recorded sequence of states, actions (and maybe rewards) made by 
    running the expert policy in an environment.
    --  if we want to use the expert demonstrations, 
        we need to run it again to generate the trajectories.
- CartPole-v0 environment:
    -- action space: 0 (left) or 1 (right)
    -- observation space: [cart position (- left, + right), cart velocity, pole angle (rad), pole angular velocity]
    -- reward: 1 for every timestep the pole is upright
    -- more info: https://www.gymlibrary.dev/environments/classic_control/cart_pole/
- expert: obviously it has 2 action space, 4 observation space, and Adam optimizer.
- "HumanCompatibleAI/ppo-seals-CartPole-v0" dataset: 24 episodes, 501 steps (trajectories) in each episode
    
"""

import numpy as np
import gymnasium as gym
from stable_baselines3.common.evaluation import evaluate_policy
from imitation.policies.serialize import load_policy
from imitation.util.util import make_vec_env
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.data import rollout
from imitation.algorithms import bc


rn_seed = rng=np.random.default_rng(0)
env = make_vec_env(
    "seals:seals/CartPole-v0",
    rng=rn_seed,
    post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],  # needed for computing rollouts later
)

# -------- Create the expert policy---------
expert = load_policy(
    "ppo-huggingface",
    organization="HumanCompatibleAI",
    env_name="seals/CartPole-v0",
    venv=env,
)

# quick evaluation of the expert policy
reward, _ = evaluate_policy(model=expert, env=env, n_eval_episodes=10, deterministic=True)
print(f"Expert Reward ==> maximum:500, my:{reward} \n")

# -------- Run the expert in the environment for trajectory generation ---------
min_rollout_episodes = 10

rollouts = rollout.rollout(
    expert,
    env,
    rollout.make_sample_until(min_timesteps=None, min_episodes=min_rollout_episodes),
    rng=rn_seed,
)
trajectories = rollout.flatten_trajectories(rollouts)

# print(f"""The `rollout` function generated a list of {len(rollouts)} {type(rollouts[0])}.
# After flattening (unpacking), this list is turned into a {type(trajectories)} 
# object containing {len(trajectories)} transitions.
# The transitions object contains arrays for: 
# {', '.join(trajectories.__dict__.keys())}."
# """)

print(f"""Number of episodes: {min_rollout_episodes}
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

evaluation_env = make_vec_env(
    "seals:seals/CartPole-v0",
    rng=rng,
    env_make_kwargs={"render_mode": "human"},  # for rendering
)


reward_before_training, _ = evaluate_policy(bc_trainer.policy, evaluation_env, n_eval_episodes=3, render=True,)
print(f"\n Reward before training: {reward_before_training}")

print("Training a policy using Behavior Cloning")
bc_trainer.train(n_epochs=1)

reward_after_training, _ = evaluate_policy(bc_trainer.policy, evaluation_env, n_eval_episodes=3, render=True,)
print(f"\n Reward after training: {reward_after_training}")




# -------- quick check of the original dataset---------

# from datasets import load_dataset

# ds = load_dataset("HumanCompatibleAI/ppo-seals-CartPole-v0")
# print(len(ds["train"]))  # Number of episodes = 24
# print(len(ds["train"][0]["obs"]))  # Number of steps in first episode = 501
