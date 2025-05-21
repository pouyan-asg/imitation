import tempfile
import numpy as np
import gymnasium as gym
from imitation.policies.serialize import load_policy
from imitation.util.util import make_vec_env
from imitation.algorithms import bc
from imitation.algorithms.dagger import SimpleDAggerTrainer
from stable_baselines3.common.evaluation import evaluate_policy
from datasets import Dataset


rn_seed = np.random.default_rng(0)
env = make_vec_env(
    "seals:seals/CartPole-v0",
    rng=rn_seed,
    n_envs=1,
)

# -------- Create the expert policy---------
expert = load_policy(
    "ppo-huggingface",
    organization="HumanCompatibleAI",
    env_name="seals/CartPole-v0",
    venv=env,
)

# -------- Train & Test ---------
bc_trainer = bc.BC(
    observation_space=env.observation_space,
    action_space=env.action_space,
    rng=rn_seed,
)

# with tempfile.TemporaryDirectory(prefix="dagger_example_") as tmpdir:
#     print(f"Temporary directory: {tmpdir}\n")  # a temporary directory like /tmp/dagger_example_8cc8n3yu
#     dagger_trainer = SimpleDAggerTrainer(
#         venv=env,
#         scratch_dir=tmpdir,
#         expert_policy=expert,
#         bc_trainer=bc_trainer,
#         rng=rn_seed,
#     )

#     dagger_trainer.train(2000)

my_dir = "/home/pouyan/phd/imitation_learning/imitation/examples/dagger/logs"
dagger_trainer = SimpleDAggerTrainer(
    venv=env,
    scratch_dir=my_dir,
    expert_policy=expert,
    bc_trainer=bc_trainer,
    rng=rn_seed,
)

dagger_trainer.train(2000)
print(dagger_trainer)

reward_after_training, _ = evaluate_policy(dagger_trainer.policy, env, 20)
print(f"\nReward after training: {reward_after_training}")


# -------- load the saved trained demos ---------
# round0_path = "/home/pouyan/phd/imitation_learning/imitation/examples/dagger/demos/round-000/dagger-demo-0-94a713f17ba248538b53986f9bf96d90.npz"
# round0_dataset = Dataset.from_file(f"{round0_path}/data-00000-of-00001.arrow")

# print("Keys:", round0_dataset.column_names)
# print("Example:", round0_dataset[0])
