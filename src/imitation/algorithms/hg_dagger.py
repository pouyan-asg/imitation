"""HG-DAgger (https://arxiv.org/pdf/1810.02890).

HG-DAgger is a human-in-the-loop version of DAgger (Dataset Aggregation),
which is an imitation learning algorithm that iteratively collects
demonstrations from a human expert and uses them to improve a policy.
"""

import abc
import logging
logging.basicConfig(level=logging.INFO)  # it shows all logs on the terminal
import os
import pathlib
import uuid
from typing import Any, Callable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch as th
from stable_baselines3.common import policies, utils, vec_env
from stable_baselines3.common.vec_env.base_vec_env import VecEnvStepReturn
from torch.utils import data as th_data

from imitation.algorithms import base, bc
from imitation.data import rollout, serialize, types
from imitation.util import logger as imit_logger
from imitation.util import util
import wandb

def reconstruct_trainer(
    scratch_dir: types.AnyPath,
    venv: vec_env.VecEnv,
    custom_logger: Optional[imit_logger.HierarchicalLogger] = None,
    device: Union[th.device, str] = "auto",
) -> "HgDAggerTrainer":
    """Reconstruct trainer from the latest snapshot in some working directory.

    Requires vectorized environment and (optionally) a logger, as these objects
    cannot be serialized.

    Args:
        scratch_dir: path to the working directory created by a previous run of
            this algorithm. The directory should contain `checkpoint-latest.pt` and
            `policy-latest.pt` files.
        venv: Vectorized training environment.
        custom_logger: Where to log to; if None (default), creates a new logger.
        device: device on which to load the trainer.

    Returns:
        A deserialized `DAggerTrainer`.
    """
    custom_logger = custom_logger or imit_logger.configure()
    scratch_dir = util.parse_path(scratch_dir)
    checkpoint_path = scratch_dir / "checkpoint-latest.pt"
    trainer = th.load(checkpoint_path, map_location=utils.get_device(device))
    trainer.venv = venv
    trainer._logger = custom_logger
    return trainer


def _save_hgdagger_demo(
    trajectory: types.Trajectory,
    trajectory_index: int,
    save_dir: types.AnyPath,
    rng: np.random.Generator,
    prefix: str = "",
) -> None:
    save_dir = util.parse_path(save_dir)
    assert isinstance(trajectory, types.Trajectory)
    actual_prefix = f"{prefix}-" if prefix else ""
    randbits = int.from_bytes(rng.bytes(16), "big")
    random_uuid = uuid.UUID(int=randbits, version=4).hex
    filename = f"{actual_prefix}hgdagger-demo-{trajectory_index}-{random_uuid}.npz"
    npz_path = save_dir / filename
    assert (
        not npz_path.exists()
    ), "The following HG-DAgger demonstration path already exists: {0}".format(npz_path)
    serialize.save(npz_path, [trajectory])
    logging.info(f"Saved demo at '{npz_path}'")


class InteractiveTrajectoryCollector(vec_env.VecEnvWrapper):
    """HG-DAgger VecEnvWrapper for querying and saving expert actions.

    Every call to `.step(actions)` accepts and saves expert actions to `self.save_dir`,
    but only forwards expert actions to the wrapped VecEnv with probability
    `self.beta`. With probability `1 - self.beta`, a "robot" action (i.e
    an action from the imitation policy) is forwarded instead.

    Demonstrations are saved as `TrajectoryWithRew` to `self.save_dir` at the end
    of every episode.
    """

    traj_accum: Optional[rollout.TrajectoryAccumulator]  # accumulates observations/actions into trajectories.
    _last_obs: Optional[np.ndarray]  #  the last observation seen
    _last_user_actions: Optional[np.ndarray]  # expert actions from the user

    def __init__(
        self,
        venv: vec_env.VecEnv,
        get_robot_acts: Callable[[np.ndarray], np.ndarray],
        save_dir: types.AnyPath,
        rng: np.random.Generator,
    ) -> None:

        super().__init__(venv)
        """
        InteractiveTrajectoryCollector inherits from VecEnvWrapper, which is a standard wrapper 
        class for vectorized environments in Stable Baselines3.
        The VecEnvWrapper constructor takes a venv argument (the environment to wrap) and 
        stores it as self.venv.
        """
        self.get_robot_acts = get_robot_acts
        self.traj_accum = None
        self.save_dir = save_dir
        self._last_obs = None
        self._done_before = True
        self._is_reset = False
        self._last_user_actions = None
        self.rng = rng

    def seed(self, seed: Optional[int] = None) -> List[Optional[int]]:
        """Set the seed for the HG-DAgger random number generator and wrapped VecEnv.

        The HG-DAgger RNG is used along with `self.beta` to determine whether the expert
        or robot action is forwarded to the wrapped VecEnv.

        Args:
            seed: The random seed. May be None for completely random seeding.

        Returns:
            A list containing the seeds for each individual env. Note that all list
            elements may be None, if the env does not return anything when seeded.
        """
        self.rng = np.random.default_rng(seed=seed)
        return list(self.venv.seed(seed))

    def reset(self) -> np.ndarray:
        """Resets the environment.

        Returns:
            obs: first observation of a new trajectory.
        """
        self.traj_accum = rollout.TrajectoryAccumulator()
        obs = self.venv.reset()
        assert isinstance(obs, np.ndarray)
        for i, ob in enumerate(obs):
            # Add the first observation to the trajectory accumulator.
            self.traj_accum.add_step({"obs": ob}, key=i)
        self._last_obs = obs
        self._is_reset = True
        self._last_user_actions = None
        return obs

    def step_async(self, actions: np.ndarray) -> None:
        """
        For HG-DAgger: Use the expert action only if human is intervening (takeover).
        Otherwise, use the learner's action.
        """
        assert self._is_reset, "call .reset() before .step()"
        assert self._last_obs is not None

        # You need a way to know, for each env, if the human is intervening.
        # Let's assume you have a boolean array: is_takeover = [True, False, ...]
        # This could be passed in, or you could get it from your controller/environment.
        # For illustration, let's say it's passed as an attribute or computed here:
        # is_takeover = self._get_takeover_flags()  # shape: (num_envs,)
        is_takeover = False

        actual_acts = np.array(actions)
        # For environments where human is NOT intervening, use the learner's action
        # TODO it should be for more than one environment. use:  if np.any(~is_takeover):
        if is_takeover == False:
            actual_acts[~is_takeover] = self.get_robot_acts(self._last_obs[~is_takeover])
            print(f"\033[94mLearner used in envs\033[0m")
        else:
            self._last_user_actions = actions  # Save expert actions for logging/dataset
            print(f"\033[94mExpert used in envs\033[0m")
        
        self.venv.step_async(actual_acts)
    
    def _get_takeover_flags(self) -> np.ndarray:
        # This should return a boolean array of shape (num_envs,)
        # indicating for each env whether the human is intervening.
        # You need to implement this based on your controller/environment.
        # For example, you might store the latest takeover flags in self._last_takeover_flags
        return self._last_takeover_flags

    def step_wait(self) -> VecEnvStepReturn:
        """Returns observation, reward, etc after previous `step_async()` call.

        Stores the transition, and saves trajectory as demo once complete.

        Returns:
            Observation, reward, dones (is terminal?) and info dict.
        """
        next_obs, rews, dones, infos = self.venv.step_wait()
        assert isinstance(next_obs, np.ndarray)
        assert self.traj_accum is not None
        assert self._last_user_actions is not None
        self._last_obs = next_obs

        # Extract takeover flags from infos
        # self._last_takeover_flags = np.array([info.get("takeover", False) for info in infos])

        for i, info in enumerate(infos):
            #for one environment: 
            # i=0, infos=[{'TimeLimit.truncated': False/True}], info={'TimeLimit.truncated': False/True}
            if info.get("TimeLimit.truncated", True) or dones[i]:
                # dones[i]=True for terminal observations
                info["your_custom_key"] = "TEST Pouyan"

        # This is where recording into the dataset happens
        fresh_demos = self.traj_accum.add_steps_and_auto_finish(
            obs=next_obs,
            acts=self._last_user_actions,
            rews=rews,
            infos=infos,
            dones=dones,
        )
        print(f"\033[94mfresh_demos: {len(fresh_demos)} demos collected in this step\033[0m")

        """
        The TrajectoryAccumulator is not saving a list of (obs, act) pairs in one shot. 
        It accumulates one step at a time, and internally stores actions as acting on the 
        previous state.
        The add_steps_and_auto_finish() call is designed to finalize the step that was just 
        executed — meaning it pairs the previous observation obs_t (stored from before) 
        with acts_t, and only uses next_obs to begin the next step.
        Even though you see obs=next_obs, this obs is the next step's start state. 
        The action is saved alongside the previous state, which was stored earlier.
        This is a rolling mechanism where the accumulator:
            - finalizes the previous step with stored obs & action
            - starts a new step with next_obs
        """

        for traj_index, traj in enumerate(fresh_demos):
            _save_hgdagger_demo(traj, traj_index, self.save_dir, self.rng)

        return next_obs, rews, dones, infos


class NeedsDemosException(Exception):
    """Signals demos need to be collected for current round before continuing."""


class HgDAggerTrainer(base.BaseImitationAlgorithm):
    """HG-DAgger training class with low-level API suitable for interactive human feedback.

    In essence, this is just BC with some helpers for incrementally
    resuming training and interpolating between demonstrator/learnt policies.
    Interaction proceeds in "rounds" in which the demonstrator first provides a
    fresh set of demonstrations, and then an underlying `BC` is invoked to
    fine-tune the policy on the entire set of demonstrations collected in all
    rounds so far. Demonstrations and policy/trainer checkpoints are stored in a
    directory with the following structure:
    """
    # holds all the expert demonstrations gathered.
    _all_demos: List[types.Trajectory]

    # The default number of BC training epochs in `extend_and_update`.
    DEFAULT_N_EPOCHS: int = 4
    

    def __init__(
        self,
        *,
        venv: vec_env.VecEnv,
        scratch_dir: types.AnyPath,
        rng: np.random.Generator,
        bc_trainer: bc.BC,
        custom_logger: Optional[imit_logger.HierarchicalLogger] = None,
    ):
        super().__init__(custom_logger=custom_logger)

        self.scratch_dir = util.parse_path(scratch_dir)
        self.venv = venv
        self.round_num = 0  # i in the original paper
        self._last_loaded_round = -1
        self._all_demos = []  # stores all expert demos across rounds.
        self.rng = rng

        # Ensures the observation/action spaces of the BC policy match the environment.
        utils.check_for_correct_spaces(
            self.venv,
            bc_trainer.observation_space,
            bc_trainer.action_space,
        )
        self.bc_trainer = bc_trainer
        self.bc_trainer.logger = self.logger

    def __getstate__(self):
        """Return state excluding non-pickleable objects."""
        d = dict(self.__dict__)
        del d["venv"]
        del d["_logger"]
        return d

    @property
    def logger(self) -> imit_logger.HierarchicalLogger:
        """Returns logger for this object."""
        return super().logger

    @logger.setter
    def logger(self, value: imit_logger.HierarchicalLogger) -> None:
        # DAgger and inner-BC logger should stay in sync
        self._logger = value
        self.bc_trainer.logger = value

    @property
    def policy(self) -> policies.BasePolicy:
        return self.bc_trainer.policy

    @property
    def batch_size(self) -> int:
        return self.bc_trainer.batch_size

    def _load_all_demos(self) -> Tuple[types.Transitions, List[int]]:
        """
        This private method loads all demonstration data from saved files 
        on disk (in .npz format), starting from the last loaded round up to the current round.
        It returns:
            - a single flattened list of transitions for training.
            - a list of number of demos per round (for logging/metrics/debugging).
        """
        num_demos_by_round = []
        for round_num in range(self._last_loaded_round + 1, self.round_num + 1):
            round_dir = self._demo_dir_path_for_round(round_num)
            demo_paths = self._get_demo_paths(round_dir)
            self._all_demos.extend(serialize.load(p)[0] for p in demo_paths)
            num_demos_by_round.append(len(demo_paths))
        logging.info(f"Loaded {len(self._all_demos)} total")
        demo_transitions = rollout.flatten_trajectories(self._all_demos)
        return demo_transitions, num_demos_by_round

    def _get_demo_paths(self, round_dir: pathlib.Path) -> List[pathlib.Path]:
        # listdir returns filenames in an arbitrary order that depends on the
        # file system implementation:
        # https://stackoverflow.com/questions/31534583/is-os-listdir-deterministic
        # To ensure the order is consistent across file systems,
        # we sort by the filename.
        """Returns all .npz files from a demo round directory, 
        sorted alphabetically for consistency."""
        filenames = sorted(os.listdir(round_dir))
        return [round_dir / f for f in filenames if f.endswith(".npz")]

    def _demo_dir_path_for_round(self, round_num: Optional[int] = None) -> pathlib.Path:
        """it creats a demo folder with the round folder(s) inside it (round-000, round-001, etc)"""
        if round_num is None:
            round_num = self.round_num
        return self.scratch_dir / "demos" / f"round-{round_num:03d}"

    def _try_load_demos(self) -> None:
        """Load the dataset for this round and sends them to the self.bc_trainer as a DataLoader.
        
        1. Looks for .npz demo files from the current round.
        2. Loads them and all previous demos into memory.
        3. Converts them into a PyTorch DataLoader.
        4. Passes the DataLoader to the BC trainer for use during training.
        5. Keeps track of what is already been loaded.
        """

        demo_dir = self._demo_dir_path_for_round()
        # If the directory exists, it finds all .npz files inside 
        # (which are saved expert demos). If not, it returns an empty list.
        demo_paths = self._get_demo_paths(demo_dir) if demo_dir.is_dir() else []

        # If no demonstrations are found for the current round, raise an exception.
        if len(demo_paths) == 0:
            raise NeedsDemosException(
                f"No demos found for round {self.round_num} in dir '{demo_dir}'. "
                f"Maybe you need to collect some demos? See "
                f".create_trajectory_collector()",
            )

        if self._last_loaded_round < self.round_num:
            transitions, num_demos = self._load_all_demos()
            logging.info(
                f"Loaded {sum(num_demos)} new demos from {len(num_demos)} rounds",
            )
            # If the total number of transitions (i.e., (obs, act) pairs) 
            # is less than the training batch size, training would fail.
            if len(transitions) < self.batch_size:
                raise ValueError(
                    "Not enough transitions to form a single batch: "
                    f"self.batch_size={self.batch_size} > "
                    f"len(transitions)={len(transitions)}",
                )
            
            #  PyTorch DataLoader
            data_loader = th_data.DataLoader(
                transitions,
                self.batch_size,
                drop_last=True,
                shuffle=True,
                collate_fn=types.transitions_collate_fn,
            )
            # Passes the DataLoader to the BC trainer for use during training.
            self.bc_trainer.set_demonstrations(data_loader)
            self._last_loaded_round = self.round_num

    def extend_and_update(
        self,
        bc_train_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> int:
        """Extend internal batch of data and train BC.

        Specifically, this method will load new transitions (if necessary), train
        the model for a while, and advance the round counter. If there are no fresh
        demonstrations in the demonstration directory for the current round, then
        this will raise a `NeedsDemosException` instead of training or advancing
        the round counter. In that case, the user should call
        `.create_trajectory_collector()` and use the returned
        `InteractiveTrajectoryCollector` to produce a new set of demonstrations for
        the current interaction round.

        Arguments:
            bc_train_kwargs: Keyword arguments for calling `BC.train()`. If
                the `log_rollouts_venv` key is not provided, then it is set to
                `self.venv` by default. If neither of the `n_epochs` and `n_batches`
                keys are provided, then `n_epochs` is set to `self.DEFAULT_N_EPOCHS`.

        Returns:
            New round number after advancing the round counter.
        """
        if bc_train_kwargs is None:
            bc_train_kwargs = {}
        else:
            bc_train_kwargs = dict(bc_train_kwargs)

        user_keys = bc_train_kwargs.keys()
        if "log_rollouts_venv" not in user_keys:
            bc_train_kwargs["log_rollouts_venv"] = self.venv

        if "n_epochs" not in user_keys and "n_batches" not in user_keys:
            bc_train_kwargs["n_epochs"] = self.DEFAULT_N_EPOCHS

        logging.info("\nLoading demonstrations")
        self._try_load_demos()
        logging.info(f"Training at round {self.round_num}")
        self.bc_trainer.train(**bc_train_kwargs)
        self.round_num += 1
        logging.info(f"New round number is {self.round_num}")
        return self.round_num

    def create_trajectory_collector(self) -> InteractiveTrajectoryCollector:
        """Create trajectory collector to extend current round's demonstration set.

        get_robot_acts: Given an input acts, return the first output of 
        self.bc_trainer.policy.predict(acts). predict function has two output 
        'action, _states = policy.predict(obs)', so we need only the first one.
        Thus, Whenever the trajectory collector asks for a robot action based on 
        an observation (called acts here), return the action chosen by the current policy

        Returns:
            A collector configured with the appropriate beta, imitator policy, etc.
            for the current round. Refer to the documentation for
            `InteractiveTrajectoryCollector` to see how to use this.
        """
        save_dir = self._demo_dir_path_for_round()
        collector = InteractiveTrajectoryCollector(
            venv=self.venv,
            get_robot_acts=lambda acts: self.bc_trainer.policy.predict(acts)[0],
            save_dir=save_dir,
            rng=self.rng,
        )
        return collector

    def save_trainer(self) -> Tuple[pathlib.Path, pathlib.Path]:
        """Create a snapshot of trainer in the scratch/working directory.

        The created snapshot can be reloaded with `reconstruct_trainer()`.
        In addition to saving one copy of the policy in the trainer snapshot, this
        method saves a second copy of the policy in its own file. Having a second copy
        of the policy is convenient because it can be loaded on its own and passed to
        evaluation routines for other algorithms.

        Returns:
            checkpoint_path: a path to one of the created `DAggerTrainer` checkpoints.
            policy_path: a path to one of the created `DAggerTrainer` policies.
        """
        self.scratch_dir.mkdir(parents=True, exist_ok=True)

        # save full trainer checkpoints
        checkpoint_paths = [
            self.scratch_dir / f"checkpoint-{self.round_num:03d}.pt",
            self.scratch_dir / "checkpoint-latest.pt",
        ]
        for checkpoint_path in checkpoint_paths:
            th.save(self, checkpoint_path)

        # save policies separately for convenience
        policy_paths = [
            self.scratch_dir / f"policy-{self.round_num:03d}.pt",
            self.scratch_dir / "policy-latest.pt",
        ]
        for policy_path in policy_paths:
            util.save_policy(self.policy, policy_path)

        return checkpoint_paths[0], policy_paths[0]


class InteractiveHgDAggerTrainer(HgDAggerTrainer):
    """
    Note: since we are passing BC trainer to this class, it will be used
    in all functions of DAggerTrainer class. Don't be confused!
    """
    """
    How Training function Works and what is the role of collector?:

    rollout.generate_trajectories
        |
        v
    calls collector.reset()  --> InteractiveTrajectoryCollector.reset()
        |
    calls collector.step(acts) --> InteractiveTrajectoryCollector.step_async(acts)
                                    |
                                    --> β-mixing, data recording
                                InteractiveTrajectoryCollector.step_wait()
                                    |
                                    --> data recording, trajectory saving
    """


    def __init__(
        self,
        *,
        venv: vec_env.VecEnv,
        scratch_dir: types.AnyPath,
        expert_policy: policies.BasePolicy,
        rng: np.random.Generator,
        expert_trajs: Optional[Sequence[types.Trajectory]] = None,
        # wandb_run: Optional[wandb.sdk.wandb_run.Run] = None,
        **dagger_trainer_kwargs,
):
        super().__init__(
            venv=venv,
            scratch_dir=scratch_dir,
            rng=rng,
            **dagger_trainer_kwargs,
        )
        
        # self.wandb_run = wandb_run
        self.log_dir = scratch_dir
        self.expert_policy = expert_policy 

        if expert_policy.observation_space != self.venv.observation_space:
            raise ValueError("Mismatched observation space between expert_policy and venv",)
        if expert_policy.action_space != self.venv.action_space:
            raise ValueError("Mismatched action space between expert_policy and venv")

        # initial expert policy
        if expert_trajs is not None:
            # Save each initial expert trajectory into the "round 0" demonstration
            print(f"\033[93m\nexpert trajs: {len(expert_trajs)}\n\033[0m")
            for traj_index, traj in enumerate(expert_trajs):
                _save_hgdagger_demo(
                    traj,
                    traj_index,
                    self._demo_dir_path_for_round(),
                    self.rng,
                    prefix="initial_data",)

    def train(
        self,
        total_timesteps: int,
        *,
        rollout_round_min_episodes: int = 3,
        rollout_round_min_timesteps: int = 500,
        bc_train_kwargs: Optional[dict] = None,
    ) -> None:

        total_timestep_count = 0
        round_num = 0

        while total_timestep_count < total_timesteps:
            print(f"\033[93m\nStarting round={round_num} with total_timestep_count={total_timestep_count}\033[0m")

            """
            collector is an instance of InteractiveTrajectoryCollector, 
            which wraps your environment and adds DAgger-specific 
            logic (like β-mixing and data recording).
            """
            collector = self.create_trajectory_collector()

            round_episode_count = 0
            round_timestep_count = 0

            sample_until = rollout.make_sample_until(
                min_timesteps=max(rollout_round_min_timesteps, self.batch_size),
                min_episodes=rollout_round_min_episodes,
            )

            """
            below line performs the data collection step. It runs the expert policy 
            in the environment (with β-mixing via the collector) and records 
            the resulting trajectories.

            generate_trajectories interacts with the environment (venv=collector) by 
            repeatedly calling step() and reset(). The collector is an InteractiveTrajectoryCollector, 
            which wraps the environment and handles the β-mixing (deciding whether to use the expert or 
            the learner action at each step). During this process, the collector records all 
            (obs, expert action) pairs and saves them for later training.
            """
            trajectories = rollout.generate_trajectories(
                policy=self.expert_policy,
                venv=collector,
                sample_until=sample_until,
                deterministic_policy=True,
                rng=collector.rng,
            )

            with open(f"{self.log_dir}/logs.txt", "a") as f:
                f.write('-'*20 + "\n")
                f.write(f"Round {round_num} trajectories:\n")
                f.write(f"Total trajectories collected: {len(trajectories)}\n")
                # f.write(f"obs: {trajectories[0].obs}\n")
                # f.write(f"acts: {trajectories[0].acts}\n")
                # f.write(f"rews: {trajectories[0].rews}\n")
                # f.write(f"infos: {trajectories[0].infos}\n")
                f.write(f"Trajectory full: {trajectories}\n")

            for traj in trajectories:
                self._logger.record_mean(
                    "dagger/mean_episode_reward",
                    np.sum(traj.rews),
                )
                round_timestep_count += len(traj)
                total_timestep_count += len(traj)

            round_episode_count += len(trajectories)

            self._logger.record("dagger/total_timesteps", total_timestep_count)
            self._logger.record("dagger/round_num", round_num)
            self._logger.record("dagger/round_episode_count", round_episode_count)
            self._logger.record("dagger/round_timestep_count", round_timestep_count)

            # self.wandb_run.log(
            #     {
            #         "dagger/total_timesteps": total_timestep_count,
            #         "dagger/round_episode_count": round_episode_count,
            #         "dagger/round_timestep_count": round_timestep_count,
            #         "dagger/mean_episode_reward": np.sum(traj.rews),
            #         "dagger/beta": self.beta,
            #     },
            #     step=round_num,
            # )

            # the expert actions are already saved in the demonstration files 
            # during the trajectory collection phase.
            # The BC trainer just needs to know where to find the data 
            # (which is handled by the DAgger trainer’s internal logic).
            self.extend_and_update(bc_train_kwargs)
            round_num += 1