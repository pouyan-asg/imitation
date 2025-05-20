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


"""
max_episode_steps: total number of steps in an episode which means total number that expert gives an action.
total_timesteps: total number of all episodes that the agent will train. total steps = total_timesteps * max_episode_steps.
n_eval_episodes: number of episodes that the agent will be evaluated.
"""

if __name__ == "__main__":
    rng = np.random.default_rng(0)

    env = vec_env.DummyVecEnv([
        lambda: gym.wrappers.TimeLimit(gym.make("CartPole-v1", 
                                                render_mode="human"), 
                                                max_episode_steps=10)
                                                ])
    env.seed(0)

    expert = interactive.CartPoleInteractivePolicy(env)

    bc_trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        rng=rng,
    )

my_dir = "/home/pouyan/phd/imitation_learning/imitation/examples/dagger/"

dagger_trainer = SimpleDAggerTrainer(
    venv=env,
    scratch_dir=my_dir,
    expert_policy=expert,
    bc_trainer=bc_trainer,
    rng=rng,
)

dagger_trainer.train(
    total_timesteps=10,
    rollout_round_min_episodes=3,
    rollout_round_min_timesteps=3,
)

reward_after_training, _ = evaluate_policy(model=dagger_trainer.policy, 
                                            env=env, 
                                            n_eval_episodes=10)
print(f"\nReward after training: {reward_after_training}")


