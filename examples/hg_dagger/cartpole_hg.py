import os
import numpy as np
import wandb
import gymnasium as gym
import imitation.util.logger as imit_logger
from datetime import datetime
from imitation.algorithms import bc
from imitation.algorithms import hg_dagger, dagger
from imitation.util import util
from imitation.policies import interactive
from imitation.policies.serialize import load_policy
from imitation.data import rollout, serialize, types
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common import vec_env


"""
- max_episode_steps: maximum number of steps in one episode which means total number that 
    expert gives an action. it is like a hard limit on the number of steps in an episode. 
    However, the number of steps in an episode can be less than this limit.
        --  Why it matters: Prevents endless episodes and ensures the agent 
            gets diverse starting conditions.
- total_timesteps: total number of all episodes that the agent will train. 
    Training will loop until it has seen at least 'total_timesteps in each environment 
    (not policy steps, env steps).
- rollout_round_min_episodes (train_min_episodes): This is part of the DAgger training loop. 
    In each round, DAgger will collect at least 'rollout_round_min_episodes' full 
    episode (from expert + some learner actions).
- rollout_round_min_timesteps (train_min_timesteps): Each DAgger round must collect at least 
    'rollout_round_min_timesteps' steps total, no matter how many episodes it takes.
- n_eval_episodes: This controls how many test episodes are run after training 
    is done. Each episode normally contains 'max_episode_steps' steps.
"""

# -------Initialize----------
rng = np.random.default_rng(0)
root_path = "/home/pouyan/phd/imitation_learning/imitation/examples/hg_dagger/logs"
timestamp = datetime.now().strftime("%Y.%m.%d_%H:%M")
logs_dir = os.path.join(root_path, f"{timestamp}")

env_max_episode_steps = 100
train_min_episodes = 10
train_min_timesteps = 50
total_timesteps = 5000
n_eval_episodes = 50

# env_max_episode_steps = 3
# train_min_episodes = 1
# train_min_timesteps = 1
# total_timesteps = 10
# n_eval_episodes = 2

# run = wandb.init(
#     project="CartPole_HG-DAgger_July25",           # your project name
#     entity="electic",         # your WandB username or team name
#     name=f"R_[{timestamp}]",    # optional: name of this specific run
# )

logger = imit_logger.configure(
    folder=logs_dir,
    format_strs=["stdout", "csv", "wandb"],
)  # HumanOutputFormat (aka "stdout" format string)

# -------Create a vectorized environment----------
env = vec_env.DummyVecEnv([
    lambda: gym.wrappers.TimeLimit(gym.make("CartPole-v1", render_mode="human"), 
                                   max_episode_steps=env_max_episode_steps)])
env.seed(0)

# -------Create expert trajectories from a pre-trained policy----------
agent_policy = load_policy(
    policy_type="ppo-huggingface",
    organization="HumanCompatibleAI",
    env_name="CartPole-v1",
    venv=env,
)

# sample_until = rollout.make_sample_until(
#     min_timesteps=train_min_timesteps,
#     min_episodes=train_min_episodes,)

# exprt_trajs = rollout.generate_trajectories(
#     policy=agent_policy,
#     venv=env,
#     sample_until=sample_until,
#     deterministic_policy=True,
#     rng=rng,
# )

# -------Prepare expert and algorithm----------
expert_policy = interactive.CartPoleHG(env)

bc_trainer = bc.BC(
    observation_space=env.observation_space,
    action_space=env.action_space,
    rng=rng,
    # wandb_run=run
)

hgdagger_trainer = hg_dagger.InteractiveHgDAggerTrainer(
    venv=env,
    scratch_dir=logs_dir,
    expert_policy=expert_policy,
    agent_policy=agent_policy,
    bc_trainer=bc_trainer,
    rng=rng,
    # expert_trajs=exprt_trajs,
    # wandb_run=run,
)

# -------Training loop (code stops here until training will be finished)----------
hgdagger_trainer.train(
    total_timesteps=total_timesteps,
    rollout_round_min_episodes=train_min_episodes,
    rollout_round_min_timesteps=train_min_timesteps,
)

# -------Reward evaluation----------
reward_after_training, _ = evaluate_policy(model=hgdagger_trainer.policy, 
                                            env=env, 
                                            n_eval_episodes=n_eval_episodes)
print(f"\033[91m\nReward after training: {reward_after_training}\033[0m")

# -------Saving trained policy----------
print("\033[92m\nSaving the trained policy...\n\033[0m")
hgdagger_trainer.save_trainer()
hgdagger_trainer.policy.save(os.path.join(logs_dir, "final_policy.pt"))