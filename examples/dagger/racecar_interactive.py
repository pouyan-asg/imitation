import os
import numpy as np
import gymnasium as gym
import wandb
<<<<<<< HEAD
import imitation.algorithms.dagger as dagger
import imitation.util.logger as imit_logger
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.algorithms import bc
from imitation.util import util
from imitation.policies import interactive
from imitation.policies.serialize import load_policy
from imitation.data import rollout, serialize, types
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from datetime import datetime
=======
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.algorithms import bc
import imitation.algorithms.dagger as dagger
from imitation.util import util
from stable_baselines3.common.evaluation import evaluate_policy
from imitation.policies import interactive
from datetime import datetime
import imitation.util.logger as imit_logger
from imitation.policies.serialize import load_policy
from imitation.data import rollout, serialize, types
>>>>>>> 9374b46c7b0c6dcf5cfde71d0cb842e567349ee7


"""
- max_episode_steps: maximum number of steps in one episode which means total number that 
    expert gives an action. it is like a hard limit on the number of steps in an episode. 
    However, the number of steps in an episode can be less than this limit.
        --  Why it matters: Prevents endless episodes and ensures the agent 
            gets diverse starting conditions.
- total_timesteps: total number of all episodes that the agent will train. 
    Training will loop until it has seen at least 'total_timesteps in each environment (not policy steps, env steps).
- rollout_round_min_episodes: This is part of the DAgger training loop. 
    In each round, DAgger will collect at least 'rollout_round_min_episodes' full 
    episode (from expert + some learner actions).
- rollout_round_min_timesteps: Each DAgger round must collect at least 
    'rollout_round_min_timesteps' steps total, no matter how many episodes it takes.
- n_eval_episodes: This controls how many test episodes are run after training 
    is done. Each episode normally contains 'max_episode_steps' steps.

- DAgger works in rounds. Each round:
    1. Collects rollouts until both:
        At least 'rollout_round_min_episodes' full episode is completed
        At least 'rollout_round_min_timesteps' steps are collected
    2. Then trains the policy using BC on the data so far
    3. Repeats this until total_timesteps >= XX (value for 'total_timesteps') is reached

- DummyVecEnv: it is used to wrap the environment in a vectorized format so 
that policies and training pipelines (like DAgger and BC) can operate on 
batched inputs and outputs, even when using only one environment. it makes it easy to 
scale up to parallel training later — without changing your core code.

- gym.wrappers.TimeLimit(..., max_episode_steps=XX): This wraps the environment to force 
    episodes to terminate after XX steps, even if the pole has not fallen. It is helpful for 
    controlling training speed and episode length.
    - lambda: This is an anonymous function (closure) that returns a fresh environment 
        instance when called.
"""

# -------Initialize----------
rng = np.random.default_rng(0)
root_path = "/home/pouyan/phd/imitation_learning/imitation/examples/dagger/logs"
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
logs_dir = os.path.join(root_path, f"{timestamp}")

# env_max_episode_steps = 100
# train_min_episodes = 10
# train_min_timesteps = 50
# total_timesteps = 5000
# n_eval_episodes = 50

env_max_episode_steps = 3
train_min_episodes = 1
train_min_timesteps = 1
total_timesteps = 50
n_eval_episodes = 2

run = wandb.init(
    project="CarRacing_DAgger_June25",           # your project name
    entity="electic",         # your WandB username or team name
    name=f"R_[{timestamp}]",    # optional: name of this specific run
)

logger = imit_logger.configure(folder=logs_dir,
                               format_strs=["stdout", "csv", "wandb"],)  
# HumanOutputFormat (aka "stdout" format string)

# -------Create a vectorized environment----------
env = VecTransposeImage(DummyVecEnv([lambda: RolloutInfoWrapper(gym.make(
                                    id="CarRacing-v2", 
                                    render_mode="human",
                                    lap_complete_percent=0.95,
                                    domain_randomize=False,
                                    continuous=True))]))

env.seed(0)

env_backend = VecTransposeImage(DummyVecEnv([lambda: RolloutInfoWrapper(gym.make(
                                    id="CarRacing-v2",
                                    lap_complete_percent=0.5,
                                    domain_randomize=False,
                                    continuous=True))]))

# -------Create expert trajectories from a pre-trained policy----------
initial_policy = load_policy(
    policy_type="ppo-huggingface",
    organization="igpaub",
    env_name="CarRacing-v2",
    venv=env,
)

sample_until = rollout.make_sample_until(min_timesteps=train_min_timesteps,
                                         min_episodes=train_min_episodes,)

exprt_trajs = rollout.generate_trajectories(policy=initial_policy,
                                            venv=env_backend,
                                            sample_until=sample_until,
                                            deterministic_policy=True,
                                            rng=rng,)

print(f"\033[92m\nExpert trajectories collected: {len(exprt_trajs)}\033[0m")

# -------Prepare expert and algorithm----------
expert = interactive.RacingInteractiveExpert(env, run)

bc_trainer = bc.BC(observation_space=env.observation_space,
                   action_space=env.action_space,
                   rng=rng,
                   wandb_run=run)

dagger_trainer = dagger.InteractiveDAggerTrainer(venv=env,
                                                 scratch_dir=logs_dir,
                                                 expert_policy=expert,
                                                 bc_trainer=bc_trainer,
                                                 rng=rng,
                                                 expert_trajs=exprt_trajs,
                                                 wandb_run=run,)

# -------Training loop (code stops here until training will be finished)----------
dagger_trainer.train(
    total_timesteps=total_timesteps,
    rollout_round_min_episodes=train_min_episodes,
    rollout_round_min_timesteps=train_min_timesteps,
)

# -------Reward evaluation----------
reward_after_training, _ = evaluate_policy(model=dagger_trainer.policy, 
                                            env=env, 
                                            n_eval_episodes=n_eval_episodes)
print(f"\033[91m\nReward after training: {reward_after_training}\033[0m")

# -------Saving trained policy----------
print("\033[92m\nSaving the trained policy...\n\033[0m")
dagger_trainer.save_trainer()
dagger_trainer.policy.save(os.path.join(logs_dir, "final_policy.pt"))