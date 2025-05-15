import tempfile

import numpy as np
import gymnasium as gym
from stable_baselines3.common.evaluation import evaluate_policy

from imitation.algorithms import bc
from imitation.algorithms.dagger import SimpleDAggerTrainer
from imitation.policies.serialize import load_policy
from imitation.util.util import make_vec_env
from imitation.policies import interactive
import collections


rn_seed = np.random.default_rng(0)
env = make_vec_env(
    "seals:seals/CartPole-v0",
    rng=rn_seed,
    n_envs=1,
    env_make_kwargs={"render_mode": "human"},
)

print("env done!")

# -------- Create the expert policy---------
# expert = load_policy(
#     "ppo-huggingface",
#     organization="HumanCompatibleAI",
#     env_name="seals-CartPole-v0",
#     venv=env,
# )

# expert = interactive.CartPoleInteractivePolicy(
#     observation_space=env.observation_space,
#     action_space=env.action_space,
#     action_keys_names=collections.OrderedDict({
#         "a": "left",
#         "d": "right",
#     }),
#     clear_screen_on_query=True,
# )

expert = interactive.CartPoleInteractivePolicy(
    env=env,
    action_keys_names=collections.OrderedDict({
        "a": "left",
        "d": "right",
    }),
    clear_screen_on_query=False,  # Keep render window open
)

print("Expert done!")

# -------- Train & Test ---------
bc_trainer = bc.BC(
    observation_space=env.observation_space,
    action_space=env.action_space,
    rng=rn_seed,
)
print("BC done!")

counter = 0
with tempfile.TemporaryDirectory(prefix="dagger_example_") as tmpdir:
    # print(tmpdir)
    dagger_trainer = SimpleDAggerTrainer(
        venv=env,
        scratch_dir=tmpdir,
        expert_policy=expert,
        bc_trainer=bc_trainer,
        rng=rn_seed,
    )
    print("Inja!")
    dagger_trainer.train(
            total_timesteps=5,
            rollout_round_min_episodes=1,
            rollout_round_min_timesteps=1,
        )
    counter += 1
    print(f"[{counter}] Action requested...")




#     dagger_trainer.train(2000)

# reward, _ = evaluate_policy(dagger_trainer.policy, env, 10)
# print("Reward:", reward)