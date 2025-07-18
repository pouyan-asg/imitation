"""DAgger (https://arxiv.org/pdf/1011.0686.pdf).

Interactively trains policy by collecting some demonstrations, doing BC, collecting more
demonstrations, doing BC again, etc. Initially the demonstrations just come from the
expert's policy; over time, they shift to be drawn more and more from the imitator's
policy.
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


class BetaSchedule(abc.ABC):
    """Computes beta (% of time demonstration action used) from training round.
    
    In DAgger, β (beta) is the probability of using the expert action instead of the 
    learned policy action during data collection (rollouts). As training progresses, 
    you usually want to decrease beta, so the agent gradually relies more on its own policy.

    - why Beta formula here differs from paper? 
        The original DAgger paper defines an optional "probabilistic mixing" strategy for beta
        (DAgger with β-mixing), where at round i, the action taken at each state is chosen as:
        π_i = β_i * π* + (1 - β_i) * π̂_i
        The paper proposes either:
            1. Always β = 1 (purely expert like BC), or
            2. A decaying β (e.g., β_i = β^i for exponential decay, or something similar).
        They emphasize that in theoretical guarantees, using β_i = 0 (i.e., only π̂_i) 
        is enough — so the decaying β is just a practical method to stabilize early training.
        This is not from the original paper, but is a practical variant often used to smooth 
        the transition from expert to learned policy.
    """

    @abc.abstractmethod
    def __call__(self, round_num: int) -> float:
        """Computes the value of beta for the current round.

        Args:
            round_num: the current round of training (starting from 0).

        Returns:
            The fraction of the time to sample a demonstrator action. Robot
                actions will be sampled the remainder of the time.
        """  # noqa: DAR202


class LinearBetaSchedule(BetaSchedule):
    """Linearly-decreasing schedule for beta."""

    def __init__(self, rampdown_rounds: int) -> None:
        """Builds LinearBetaSchedule.

        Args:
            rampdown_rounds: number of rounds over which to anneal beta. it controls 
                            how many rounds it takes for beta to decrease from 1 to 0.
        """
        self.rampdown_rounds = rampdown_rounds

    def __call__(self, round_num: int) -> float:
        """Computes beta value.

        Args:
            round_num: the current round number.

        Returns:
            beta linearly decreasing from `1` to `0` between round `0` and
            `self.rampdown_rounds`. After that, it is always 0.
        """
        assert round_num >= 0
        return min(1, max(0, (self.rampdown_rounds - round_num) / self.rampdown_rounds))


class ExponentialBetaSchedule(BetaSchedule):
    """Exponentially decaying schedule for beta."""

    def __init__(self, decay_probability: float):
        """Builds ExponentialBetaSchedule.

        Args:
            decay_probability: the decay factor for beta.

        Raises:
            ValueError: if `decay_probability` not within (0, 1].
        """
        if not (0 < decay_probability <= 1):
            raise ValueError("decay_probability lies outside the range (0, 1].")
        self.decay_probability = decay_probability

    def __call__(self, round_num: int) -> float:
        """Computes beta value.

        Args:
            round_num: the current round number.

        Returns:
            beta as `self.decay_probability ^ round_num`
        """
        assert round_num >= 0
        return self.decay_probability**round_num


def reconstruct_trainer(
    scratch_dir: types.AnyPath,
    venv: vec_env.VecEnv,
    custom_logger: Optional[imit_logger.HierarchicalLogger] = None,
    device: Union[th.device, str] = "auto",
) -> "DAggerTrainer":
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


def _save_dagger_demo(
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
    filename = f"{actual_prefix}dagger-demo-{trajectory_index}-{random_uuid}.npz"
    npz_path = save_dir / filename
    assert (
        not npz_path.exists()
    ), "The following DAgger demonstration path already exists: {0}".format(npz_path)
    serialize.save(npz_path, [trajectory])
    logging.info(f"Saved demo at '{npz_path}'")


class InteractiveTrajectoryCollector(vec_env.VecEnvWrapper):
    """DAgger VecEnvWrapper for querying and saving expert actions.

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
        beta: float,
        save_dir: types.AnyPath,
        rng: np.random.Generator,
    ) -> None:
        """Builds InteractiveTrajectoryCollector.

        Args:
            venv: vectorized environment to sample trajectories from.
            get_robot_acts: get robot actions that can be substituted for
                human actions. Takes a vector of observations as input & returns a
                vector of actions.
            beta: fraction of the time to use action given to .step() instead of
                robot action. The choice of robot or human action is independently
                randomized for each individual `Env` at every timestep.
            save_dir: directory to save collected trajectories in.
            rng: random state for random number generation.
        """

        super().__init__(venv)
        """
        InteractiveTrajectoryCollector inherits from VecEnvWrapper, which is a standard wrapper 
        class for vectorized environments in Stable Baselines3.
        The VecEnvWrapper constructor takes a venv argument (the environment to wrap) and 
        stores it as self.venv.
        """
        self.get_robot_acts = get_robot_acts
        assert 0 <= beta <= 1
        self.beta = beta
        self.traj_accum = None
        self.save_dir = save_dir
        self._last_obs = None
        self._done_before = True
        self._is_reset = False
        self._last_user_actions = None
        self.rng = rng

    def seed(self, seed: Optional[int] = None) -> List[Optional[int]]:
        """Set the seed for the DAgger random number generator and wrapped VecEnv.

        The DAgger RNG is used along with `self.beta` to determine whether the expert
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
        """Steps with a `1 - beta` chance of using `self.get_robot_acts` instead.

        DAgger needs to be able to inject imitation policy actions randomly at some
        subset of time steps. This method has a `self.beta` chance of keeping the
        `actions` passed in as an argument, and a `1 - self.beta` chance of
        forwarding actions generated by `self.get_robot_acts` instead.
        "robot" (i.e. imitation policy) action if necessary.

        IMPORTANT: At the end of every episode, a `TrajectoryWithRew` is saved to `self.save_dir`,
        where every saved action is the expert action, regardless of whether the
        robot action was used during that timestep. 
        🛠️ So, regardless of what action is executed, DAgger always stores the expert action for training.

        Args:
            actions: the _intended_ demonstrator/expert actions for the current
                state. This will be executed with probability `self.beta`.
                Otherwise, a "robot" (typically a BC policy) action will be sampled
                and executed instead via `self.get_robot_act`.
        """
        """"
        Where are the expert actions and learner states recorded?
        They are recorded inside InteractiveTrajectoryCollector via the 
        TrajectoryAccumulator, using self._last_user_actions and self._last_obs,
        regardless of whether the environment actually executed those actions.
        """

        assert self._is_reset, "call .reset() before .step()"
        assert self._last_obs is not None

        # Replace each given action with a robot action 100*(1-beta)% of the time.
        actual_acts = np.array(actions)

        # This line creates a boolean mask that decides, for each environment, 
        # whether to use the robot’s action instead of the expert's. The result 
        # is a boolean mask of shape (num_envs,) indicating which environments 
        # will use the robot policy's action.
        mask = self.rng.uniform(0, 1, size=(self.num_envs,)) > self.beta
        print(f"\033[94mLearner used in envs: {mask}\033[0m")
        # Even though there's no for loop, vectorized NumPy operations like 
        # array[mask] = ... behave just like looping over the True values in 
        # mask — they're just faster and cleaner.
        """"
        For each environment where mask is True, we replace the expert's action 
        (initially in actual_acts) with the learner's action (can be (by chance) the same or difference).
        For environments where mask is False, we keep the expert's action.
        
        Dont forget that actions are 0 or 1 and final result is either [0] or [1] for each step.
        """
        # print(f"actual_acts: {actual_acts}")
        if np.sum(mask) != 0:
            actual_acts[mask] = self.get_robot_acts(self._last_obs[mask])
            # print('-'*20)
            # print(f"actual_acts: {actual_acts}")
            # print(f"mask: {mask}")
            # print(f"_last_obs: {self._last_obs}")
            # print(f"actual_acts[mask]: {actual_acts[mask]}")
            # print(f"self._last_obs[mask]: {self._last_obs[mask]}")
            # print(f"\033[94mRobot actions: {self.get_robot_acts(self._last_obs[mask])}\033[0m")
            # print('-'*20 + "\n")
        
        """
        while π_i = β_i * π* + (1 - β_i) * π̂_i is not explicity shown but when we select
        an action with Beta probablity, it is exactly using either expert or learner policies.
        Generally self.bc_trainer.policy is π^i— the current learner. in order to understand better,
        look at 'get_robot_acts' function. 
        """

        self._last_user_actions = actions  # Expert actions saved for later, regardless of execution
        self.venv.step_async(actual_acts)

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
            _save_dagger_demo(traj, traj_index, self.save_dir, self.rng)

        return next_obs, rews, dones, infos


class NeedsDemosException(Exception):
    """Signals demos need to be collected for current round before continuing."""


class DAggerTrainer(base.BaseImitationAlgorithm):
    """DAgger training class with low-level API suitable for interactive human feedback.

    In essence, this is just BC with some helpers for incrementally
    resuming training and interpolating between demonstrator/learnt policies.
    Interaction proceeds in "rounds" in which the demonstrator first provides a
    fresh set of demonstrations, and then an underlying `BC` is invoked to
    fine-tune the policy on the entire set of demonstrations collected in all
    rounds so far. Demonstrations and policy/trainer checkpoints are stored in a
    directory with the following structure:

       scratch-dir-name/
           checkpoint-001.pt
           checkpoint-002.pt
           …
           checkpoint-XYZ.pt
           checkpoint-latest.pt
           demos/
               round-000/
                   demos_round_000_000.npz
                   demos_round_000_001.npz
                   …
               round-001/
                   demos_round_001_000.npz
                   …
               …
               round-XYZ/
                   …
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
        beta_schedule: Optional[Callable[[int], float]] = None,
        bc_trainer: bc.BC,
        custom_logger: Optional[imit_logger.HierarchicalLogger] = None,
    ):
        """Builds DAggerTrainer.

        Args:
            venv: Vectorized training environment.
            scratch_dir: Directory to use to store intermediate training
                information (e.g. for resuming training).
            rng: random state for random number generation.
            beta_schedule: Provides a value of `beta` (the probability of taking
                expert action in any given state) at each round of training. If
                `None`, then `linear_beta_schedule` will be used instead.
                it is a maximum limits for beta to decrease to 0 (by default is 15).
            bc_trainer: A `BC` instance used to train the underlying policy.
            custom_logger: Where to log to; if None (default), creates a new logger.

        Notes:
            -round:
                -- think of a round as one full cycle of: 
                    collecting new data → aggregating it → training.
            - demo:
                -- One expert-generated episode of (obs, act) pairs (i.e., one trajectory)

        demos/
        ├── round-000/
        │   ├── dagger-demo-XXXX.npz
        │   ├── dagger-demo-XXXX.npz
        ├── round-001/
        │   ├── dagger-demo-XXXX.npz
        │   └── dagger-demo-XXXX.npz
        
        """
        super().__init__(custom_logger=custom_logger)

        if beta_schedule is None:
            beta_schedule = LinearBetaSchedule(15)  # A fixed number of rounds
        self.beta_schedule = beta_schedule
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
        beta = self.beta_schedule(self.round_num)
        self.beta = beta
        collector = InteractiveTrajectoryCollector(
            venv=self.venv,
            get_robot_acts=lambda acts: self.bc_trainer.policy.predict(acts)[0],
            beta=beta,
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


class SimpleDAggerTrainer(DAggerTrainer):
    """Simpler subclass of DAggerTrainer for training with synthetic feedback.
    This is a simplified version of the DAggerTrainer class that is made specifically 
    for cases where the expert feedback is synthetic — meaning it's generated automatically, 
    not from a human.
    SimpleDAggerTrainer = "easy-to-use DAgger implementation designed for when your expert 
    is automated (e.g., a trained model), not a human giving live inputs."
    
    🔧General Structure
    SimpleDAggerTrainer is a subclass of DAggerTrainer and does two main things:
        1. Initializes a DAgger trainer using a synthetic (pre-trained) expert policy.
        2. Runs DAgger training rounds where:
            - It collects trajectories using the expert policy 
            (mixed with the agent own actions depending on a probability beta)
            - It updates the agent via behavioral cloning on all collected data
    """

    def __init__(
        self,
        *,
        venv: vec_env.VecEnv,
        scratch_dir: types.AnyPath,
        expert_policy: policies.BasePolicy,
        rng: np.random.Generator,
        expert_trajs: Optional[Sequence[types.Trajectory]] = None,
        **dagger_trainer_kwargs,
    ):
        """Builds SimpleDAggerTrainer.

        Args:
            venv: Vectorized training environment. Note that when the robot
                action is randomly injected (in accordance with `beta_schedule`
                argument), every individual environment will get a robot action
                simultaneously for that timestep.
            scratch_dir: Directory to use to store intermediate training
                information (e.g. for resuming training).
            expert_policy: The expert policy used to generate synthetic demonstrations.
            rng: Random state to use for the random number generator.
            expert_trajs: Optional starting dataset that is inserted into the round 0
                dataset.
            dagger_trainer_kwargs: Other keyword arguments passed to the
                superclass initializer `DAggerTrainer.__init__`.

        Raises:
            ValueError: The observation or action space does not match between
                `venv` and `expert_policy`.
        """
        super().__init__(
            venv=venv,
            scratch_dir=scratch_dir,
            rng=rng,
            **dagger_trainer_kwargs,
        )
        self.expert_policy = expert_policy  # initial expert policy
        if expert_policy.observation_space != self.venv.observation_space:
            raise ValueError(
                "Mismatched observation space between expert_policy and venv",
            )
        if expert_policy.action_space != self.venv.action_space:
            raise ValueError("Mismatched action space between expert_policy and venv")

        # TODO(shwang):
        #   Might welcome Transitions and DataLoaders as sources of expert data
        #   in the future too, but this will require some refactoring, so for
        #   now we just have `expert_trajs`.
        if expert_trajs is not None:
            # Save each initial expert trajectory into the "round 0" demonstration
            # data directory.
            print("\nexpert trajs:", len(expert_trajs))
            for traj_index, traj in enumerate(expert_trajs):
                _save_dagger_demo(
                    traj,
                    traj_index,
                    self._demo_dir_path_for_round(),
                    self.rng,
                    prefix="initial_data",
                )

    def train(
        self,
        total_timesteps: int,
        *,
        rollout_round_min_episodes: int = 3,
        rollout_round_min_timesteps: int = 500,
        bc_train_kwargs: Optional[dict] = None,
    ) -> None:
        """Train the DAgger agent.

        The agent is trained in "rounds" where each round consists of a dataset
        aggregation step followed by BC update step.

        During a dataset aggregation step, `self.expert_policy` is used to perform
        rollouts in the environment but there is a `1 - beta` chance (beta is
        determined from the round number and `self.beta_schedule`) that the DAgger
        agent's action is used instead. Regardless of whether the DAgger agent's action
        is used during the rollout, the expert action and corresponding observation are
        always appended to the dataset. The number of environment steps in the
        dataset aggregation stage is determined by the `rollout_round_min*` arguments.

        During a BC update step, `BC.train()` is called to update the DAgger agent on
        all data collected so far.

        Args:
            total_timesteps: The number of timesteps to train inside the environment.
                In practice this is a lower bound, because the number of timesteps is
                rounded up to finish the minimum number of episodes or timesteps in the
                last DAgger training round, and the environment timesteps are executed
                in multiples of `self.venv.num_envs`.
            rollout_round_min_episodes: The number of episodes that must be completed
                before a dataset aggregation step ends.
            rollout_round_min_timesteps: The number of environment timesteps that must
                be completed before a dataset aggregation step ends. Also, that any
                round will always train for at least `self.batch_size` timesteps,
                because otherwise BC could fail to receive any batches.
            bc_train_kwargs: Keyword arguments for calling `BC.train()`. If
                the `log_rollouts_venv` key is not provided, then it is set to
                `self.venv` by default. If neither of the `n_epochs` and `n_batches`
                keys are provided, then `n_epochs` is set to `self.DEFAULT_N_EPOCHS`.
        """
        total_timestep_count = 0
        round_num = 0

        while total_timestep_count < total_timesteps:
            print(f"\033[94m\nStarting round {round_num} with total_timestep_count={total_timestep_count}\033[0m")

            collector = self.create_trajectory_collector()

            """
            collector is type of the vectorized environment but with more parameters like Beta.
            collector will be used in 'rollout.generate_trajectories' as the enviornment.

            create_trajectory_collector() is a simple function to use InteractiveTrajectoryCollector class.
            some parameters such as venv, beta, get_robot_acts, etc send to the main class.
            note that get_robot_acts is a function which connects to BC predict (i.e. bc_trainer.policy.predict).
            Also, beta is created in a linear form reagrding a fixed horizon (e.g. LinearBetaSchedule(15))

            InteractiveTrajectoryCollector class has some internal functions which will be run.
            when we want to create trajectory samples, we use 'rollout.generate_trajectories' with
            expert policy and collector which is a more compelet version of the environment.
            in 'rollout.generate_trajectories' we have venv.step() and venv.reset(). The .step()
            function connects to same one in 'base_vec_env.py'. Note that 'InteractiveTrajectoryCollector' class 
            inhrites from VecEnvWrapper class which is a child of abstract class 'VecEnv'.
            so, step() is a function of 'VecEnv' class. in this original function, two other functions
            run include step_async() and step_wait(). However, since this class is an abstract version,
            both latter functions will be run through child class which is 'InteractiveTrajectoryCollector'.

            if you check step_async() and step_wait() functions inside 'InteractiveTrajectoryCollector' class,
            they are both connected to DummyVecEnv class in dummy_vec_env.py. 'DummyVecEnv' class 
            is a child of 'VecEnv' as well.

            we see how all these functions are connected in an abstract format.

            Conclusion: when generate_trajectories() runs, it recevies various parametrs from the environment
            to give us (state, action) pairs. in the internal steps, beta and expert policies are considered too.
            """

            round_episode_count = 0
            round_timestep_count = 0

            sample_until = rollout.make_sample_until(
                min_timesteps=max(rollout_round_min_timesteps, self.batch_size),
                min_episodes=rollout_round_min_episodes,
            )

            # Uses the expert_policy to generate (obs, act) pairs. 
            # It may mix in the learner’s actions depending on beta.
            """process of creating a new dataset of visited states by sampled policy and actions given 
            by expert + saving them is done inside 'rollout.generate_trajectories' which 
            has a back and forth communication with 'InteractiveTrajectoryCollector' class. 
            In the rest of code, we dont work with trajectories since we already saved it.
            """
            trajectories = rollout.generate_trajectories(
                policy=self.expert_policy,
                venv=collector,
                sample_until=sample_until,
                deterministic_policy=True,
                rng=collector.rng,
            )

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

            # `logger.dump` is called inside BC.train within the following fn call:
            # bc_train_kwargs can be None or a dictionary. If None, default training 
            # settings are used. If a dictionary, those settings are passed to the BC trainer.
            self.extend_and_update(bc_train_kwargs)
            round_num += 1


class InteractiveDAggerTrainer(DAggerTrainer):
    """
    a copy of SimpleDAggerTrainer but with some modifications.
    please refer to the SimpleDAggerTrainer class for more information.

    *** Note: since we are passing BC trainer to this class, it will br used
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

        self.log_dir = scratch_dir
        self.expert_policy = expert_policy  # initial expert policy
        if expert_policy.observation_space != self.venv.observation_space:
            raise ValueError(
                "Mismatched observation space between expert_policy and venv",
            )
        if expert_policy.action_space != self.venv.action_space:
            raise ValueError("Mismatched action space between expert_policy and venv")
        # self.wandb_run = wandb_run

        # TODO(shwang):
        #   Might welcome Transitions and DataLoaders as sources of expert data
        #   in the future too, but this will require some refactoring, so for
        #   now we just have `expert_trajs`.

        if expert_trajs is not None:
            # Save each initial expert trajectory into the "round 0" demonstration
            print(f"\033[93m\nexpert trajs: {len(expert_trajs)}\n\033[0m")
            for traj_index, traj in enumerate(expert_trajs):
                _save_dagger_demo(
                    traj,
                    traj_index,
                    self._demo_dir_path_for_round(),
                    self.rng,
                    prefix="initial_data",
                )

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