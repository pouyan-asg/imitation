"""Training DAgger with an interactive policy that queries the user for actions.

Note that this is a toy example that does not lead to training a reasonable policy.
"""

import tempfile

import gymnasium as gym
import numpy as np
from stable_baselines3.common import vec_env

from imitation.algorithms import bc
from imitation.algorithms.dagger import SimpleDAggerTrainer
from imitation.policies import interactive
from stable_baselines3.common.evaluation import evaluate_policy


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    env = vec_env.DummyVecEnv([lambda: gym.wrappers.TimeLimit(gym.make("Pong-v4"), 1)])
    # when number of episodes is more than 1, the trajectory will not be added (in rollout 484) and it is always zero.
    # also, active=True, Dones=False.
    # when it is 1, the trajectory will be added and it is always 1.
    # also, active=True, Dones=True.
    # I can not see any update in the render as well.
    env.seed(0)

    expert = interactive.AtariInteractivePolicy(env)

    bc_trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        rng=rng,
    )

    with tempfile.TemporaryDirectory(prefix="dagger_example_") as tmpdir:
        dagger_trainer = SimpleDAggerTrainer(
            venv=env,
            scratch_dir=tmpdir,
            expert_policy=expert,
            bc_trainer=bc_trainer,
            rng=rng,
        )
        dagger_trainer.train(
            total_timesteps=5,
            rollout_round_min_episodes=1,
            rollout_round_min_timesteps=1,
        )

    reward_after_training, _ = evaluate_policy(dagger_trainer.policy, env, 20)
    print(f"\nReward after training: {reward_after_training}")
