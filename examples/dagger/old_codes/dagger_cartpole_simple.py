import os
import tempfile
import numpy as np
import gymnasium as gym
from imitation.policies.serialize import load_policy
from imitation.util.util import make_vec_env
from imitation.algorithms import bc, dagger
from imitation.algorithms.dagger import SimpleDAggerTrainer
from stable_baselines3.common.evaluation import evaluate_policy
from datasets import Dataset
from datetime import datetime
from huggingface_sb3 import load_from_hub
from stable_baselines3 import PPO


rn_seed = np.random.default_rng(0)
env = make_vec_env(
    "CartPole-v1",
    rng=rn_seed,
    n_envs=1,
)

# -------- Create the expert policy---------
# expert = load_policy(
#     policy_type="ppo-huggingface",
#     organization="HumanCompatibleAI",
#     env_name="seals/CartPole-v0",
#     venv=env,
# )

checkpoint = load_from_hub(
	repo_id="HumanCompatibleAI/ppo-CartPole-v1",
	filename="ppo-CartPole-v1.zip",
)

print("Checkpoint loaded successfully from Hugging Face Hub!")

expert = PPO.load(checkpoint).policy

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

root_path = "/home/pouyan/phd/imitation_learning/imitation/examples/dagger/logs"
timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
logs_dir = os.path.join(root_path, f"{timestamp}")

dagger_trainer = SimpleDAggerTrainer(
    venv=env,
    scratch_dir=logs_dir,
    expert_policy=expert,
    bc_trainer=bc_trainer,
    rng=rn_seed,
)

dagger_trainer.train(2000)
dagger_trainer.save_trainer()

reward_after_training, _ = evaluate_policy(dagger_trainer.policy, env, 20)
print(f"\nReward after training: {reward_after_training}")


# -------- load the saved trained demos ---------
# round0_path = "/home/pouyan/phd/imitation_learning/imitation/examples/dagger/demos/round-000/dagger-demo-0-94a713f17ba248538b53986f9bf96d90.npz"
# round0_dataset = Dataset.from_file(f"{round0_path}/data-00000-of-00001.arrow")

# print("Keys:", round0_dataset.column_names)
# print("Example:", round0_dataset[0])