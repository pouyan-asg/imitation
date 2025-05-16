"""This is a simple example demonstrating how to clone the behavior of an expert.

we first train an expert policy using the stable-baselines3 library (RL) and then imitate 
its behavior using Behavioral Cloning (BC). 
"""
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.ppo import MlpPolicy

from imitation.algorithms import bc
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.policies.serialize import load_policy
from imitation.util.util import make_vec_env


rn_seed = rng=np.random.default_rng(0)
env = make_vec_env(
    "seals:seals/CartPole-v0",
    rng=rn_seed,
    post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],  # for computing rollouts
)

# -------- Create the expert policy---------
print("Training a expert with PPO.")
expert = PPO(
    policy=MlpPolicy,
    env=env,
    seed=0,
    batch_size=64,
    ent_coef=0.0,
    learning_rate=0.0003,
    n_epochs=10,
    n_steps=64,
)
expert.learn(100000)  # Note: change this to 100_000 to train a decent expert.

reward, _ = evaluate_policy(model=expert, env=env, n_eval_episodes=10, deterministic=True)
print(f"Expert Reward ==> maximum:500, my:{reward} \n")

# -------- Run the expert in the environment for trajectory generation ---------
print("Sampling expert transitions.")
min_rollout_episodes = 10

rollouts = rollout.rollout(
    expert,
    env,
    rollout.make_sample_until(min_timesteps=None, min_episodes=min_rollout_episodes),
    rng=rn_seed,
)
trajectories = rollout.flatten_trajectories(rollouts)

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

# same as env but for rendering
evaluation_env = make_vec_env(
    "seals:seals/CartPole-v0",
    rng=rng,
    env_make_kwargs={"render_mode": "human"},
)

print("Evaluating the untrained policy.")
reward_before_training, _ = evaluate_policy(
    bc_trainer.policy,
    env=env, # env or evaluation_env
    n_eval_episodes=3,
    render=True, 
)
print(f"\n Reward before training: {reward_before_training}")

print("Training a policy using Behavior Cloning")
bc_trainer.train(n_epochs=1)

print("Evaluating the trained policy.")
reward_after_training, _ = evaluate_policy(
    bc_trainer.policy,
    env=env,
    n_eval_episodes=3,
    render=True,
)
print(f"\n Reward after training: {reward_after_training}")
