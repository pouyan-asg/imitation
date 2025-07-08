"""for detailed information, check InteractiveTrajectoryCollector class in dagger.py"""

import numpy as np
from typing import Callable, List, Optional
from stable_baselines3.common import vec_env
from stable_baselines3.common.vec_env.base_vec_env import VecEnvStepReturn
from imitation.data import rollout, types
from imitation.algorithms.dagger import _save_dagger_demo


class InteractiveCollectorHG(vec_env.VecEnvWrapper):
    """HG-DAgger VecEnvWrapper for querying and saving expert actions.

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
        
        super().__init__(venv)
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
        if np.sum(mask) != 0:
            actual_acts[mask] = self.get_robot_acts(self._last_obs[mask])

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

        for traj_index, traj in enumerate(fresh_demos):
            _save_dagger_demo(traj, traj_index, self.save_dir, self.rng)

        return next_obs, rews, dones, infos
    


# ----------------------------------------------------------------------------
"""this ia a sample template for creating a HIL env."""

# import gymnasium as gym


# class HumanTakeoverWrapper(gym.Wrapper):
#     def __init__(self, env):
#         super().__init__(env)
#         self.takeover = False
#         self.takeover_cost = 0

#     def step(self, action):
#         # You need a way to detect if human took over here
#         obs, reward, done, info = self.env.step(action)
#         info["takeover"] = self.takeover
#         info["takeover_cost"] = self.takeover_cost
#         return obs, reward, done, info

#     def reset(self, **kwargs):
#         self.takeover = False
#         self.takeover_cost = 0
#         return self.env.reset(**kwargs)