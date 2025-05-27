import os
import tempfile
import collections
import numpy as np
import gymnasium as gym
from stable_baselines3.common import vec_env
from imitation.algorithms import bc
from imitation.algorithms.dagger import SimpleDAggerTrainer
from imitation.policies.base import NonTrainablePolicy
from imitation.util import util
from stable_baselines3.common.evaluation import evaluate_policy
from imitation.policies import interactive
from datetime import datetime
from datasets import Dataset
from imitation.policies.serialize import load_policy, save_stable_model
from pathlib import Path
from imitation.util import util


"""
- max_episode_steps: total number of steps in one episode which means total number that 
    expert gives an action.
- rollout_round_min_episodes: This is part of the DAgger training loop. 
    In each round, DAgger will collect at least 'rollout_round_min_episodes' full 
    episode from the expert (with some agent interaction too).
- rollout_round_min_timesteps: Each DAgger round must collect at least 
    'rollout_round_min_timesteps' steps total, no matter how many episodes it takes.
- total_timesteps: total number of all episodes that the agent will train. 
    Training will loop until it has seen at least 'total_timesteps total environment 
    steps (not policy steps, env steps).
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

# -------variables----------
rng = np.random.default_rng(0)
root_path = "/home/pouyan/phd/imitation_learning/imitation/examples/dagger/logs"
timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
logs_dir = os.path.join(root_path, f"{timestamp}")

# -------Create a vectorized environment----------
env = vec_env.DummyVecEnv([
    lambda: gym.wrappers.TimeLimit(gym.make("CartPole-v1", render_mode="human"), 
                                   max_episode_steps=20)
                                            ])
env.seed(0)

# -------Prepare expert and algorithm----------
expert = interactive.CartPoleInteractivePolicy(env)

bc_trainer = bc.BC(
    observation_space=env.observation_space,
    action_space=env.action_space,
    rng=rng,
)

dagger_trainer = SimpleDAggerTrainer(
    venv=env,
    scratch_dir=logs_dir,
    expert_policy=expert,
    bc_trainer=bc_trainer,
    rng=rng,
)

# -------Training loop (code stops here until training will be finished)----------
dagger_trainer.train(
    total_timesteps=100,
    rollout_round_min_episodes=1,
    rollout_round_min_timesteps=1,
)

# -------Reward evaluation----------
reward_after_training, _ = evaluate_policy(model=dagger_trainer.policy, 
                                            env=env, 
                                            n_eval_episodes=10)
print(f"\nReward after training: {reward_after_training}")

# -------Saving trained policy----------
dagger_trainer.save_trainer()









# ---------------OLD CODE ----------------- 
# import tempfile

# import numpy as np
# import gymnasium as gym
# from stable_baselines3.common.evaluation import evaluate_policy

# from imitation.algorithms import bc
# from imitation.algorithms.dagger import SimpleDAggerTrainer
# from imitation.policies.serialize import load_policy
# from imitation.util.util import make_vec_env
# from imitation.policies import interactive
# import collections
# from stable_baselines3.common import vec_env


# rn_seed = np.random.default_rng(0)
# # env = make_vec_env(
# #     "seals:seals/CartPole-v0",
# #     rng=rn_seed,
# #     n_envs=1,
# #     env_make_kwargs={"render_mode": "human"},
# # )
# env = vec_env.DummyVecEnv([lambda: gym.wrappers.TimeLimit(gym.make("CartPole-v1"), 1)])
 
# env.seed(0)

# # -------- Create the expert policy---------
# # expert = load_policy(
# #     "ppo-huggingface",
# #     organization="HumanCompatibleAI",
# #     env_name="seals-CartPole-v0",
# #     venv=env,
# # )

# expert = interactive.CartPoleInteractivePolicy(
#     env=env,
#     action_keys_names=collections.OrderedDict({
#         "a": "left",
#         "d": "right",
#     }),
#     clear_screen_on_query=False,  # clean terminal or not
# )

# # -------- Train & Test ---------
# bc_trainer = bc.BC(
#     observation_space=env.observation_space,
#     action_space=env.action_space,
#     rng=rn_seed,
# )

# with tempfile.TemporaryDirectory(prefix="dagger_example_") as tmpdir:
#     print(tmpdir)  # a temporary directory like /tmp/dagger_example_8cc8n3yu
#     dagger_trainer = SimpleDAggerTrainer(
#         venv=env,
#         scratch_dir=tmpdir,
#         expert_policy=expert,
#         bc_trainer=bc_trainer,
#         rng=rn_seed,
#     )
    
#     dagger_trainer.train(
#             total_timesteps=5,
#             rollout_round_min_episodes=1,
#             rollout_round_min_timesteps=1,
#         )




# #     dagger_trainer.train(2000)

# # reward, _ = evaluate_policy(dagger_trainer.policy, env, 10)
# # print("Reward:", reward)

# -------------------------------------------------


# dagger_trainer.policy.save(f"{os.path.splitext(os.path.basename(__file__))[0]}_policy.zip")

# util.save_policy(dagger_trainer.policy, policy-latest.pt)