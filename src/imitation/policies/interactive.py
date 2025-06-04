"""Interactive policies that query the user for actions."""

import abc
import collections
import wandb
from typing import Dict, Optional, Union

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from shimmy import atari_env
from stable_baselines3.common import vec_env

import imitation.policies.base as base_policies
from imitation.util import util
from imitation.policies.base import NonTrainablePolicy


class DiscreteInteractivePolicy(base_policies.NonTrainablePolicy, abc.ABC):
    """Abstract class for interactive policies with discrete actions.

    For each query, the observation is rendered and then the action is provided
    as a keyboard input.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        action_keys_names: collections.OrderedDict,
        clear_screen_on_query: bool = False,
    ):
        """Builds DiscreteInteractivePolicy.

        Args:
            observation_space: Observation space.
            action_space: Action space.
            action_keys_names: `OrderedDict` containing pairs (key, name) for every
                action, where key will be used in the console interface, and name
                is a semantic action name. The index of the pair in the dictionary
                will be used as the discrete, integer action.
            clear_screen_on_query: If `True`, console will be cleared on every query.
        """
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
        )

        assert isinstance(action_space, gym.spaces.Discrete)
        assert (
            len(action_keys_names)
            == len(set(action_keys_names.values()))
            == action_space.n
        )

        self.action_keys_names = action_keys_names
        self.action_key_to_index = {
            k: i for i, k in enumerate(action_keys_names.keys())
        }
        self.clear_screen_on_query = clear_screen_on_query

    # hows the image, prompts the user to press a key, then maps key → action index
    def _choose_action(
        self,
        obs: Union[np.ndarray, Dict[str, np.ndarray]],
    ) -> np.ndarray:
        if self.clear_screen_on_query:
            util.clear_screen()

        if isinstance(obs, dict):
            raise ValueError("Dictionary observations are not supported here")

        context = self._render(obs)
        key = self._get_input_key()
        self._clean_up(context)

        return np.array([self.action_key_to_index[key]])

    def _get_input_key(self) -> str:
        """Obtains input key for action selection."""
        print(
            "Please select an action. Possible choices in [ACTION_NAME:KEY] format:",
            ", ".join([f"{n}:{k}" for k, n in self.action_keys_names.items()]),
        )

        key = input("Your choice (enter key):")
        while key not in self.action_keys_names.keys():
            key = input("Invalid key, please try again! Your choice (enter key):")

        return key

    @abc.abstractmethod
    def _render(self, obs: np.ndarray) -> Optional[object]:
        """Renders an observation, optionally returns a context for later cleanup."""

    def _clean_up(self, context: object) -> None:
        """Cleans up after the input has been captured, e.g. stops showing the image."""


class ImageObsDiscreteInteractivePolicy(DiscreteInteractivePolicy):
    """DiscreteInteractivePolicy that renders image observations."""

    def _render(self, obs: np.ndarray) -> plt.Figure:
        img = self._prepare_obs_image(obs)

        fig, ax = plt.subplots()
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)  # cmap is ignored for RGB images.
        ax.axis("off")
        fig.show()

        return fig

    def _clean_up(self, context: plt.Figure) -> None:
        plt.close(context)

    def _prepare_obs_image(self, obs: np.ndarray) -> np.ndarray:
        """Applies any required observation processing to get an image to show."""
        return obs

# This maps Atari action names (from the environment) to keyboard keys
ATARI_ACTION_NAMES_TO_KEYS = {
    "NOOP": "1",
    "FIRE": "2",
    "UP": "w",
    "RIGHT": "d",
    "LEFT": "a",
    "DOWN": "x",
    "UPRIGHT": "e",
    "UPLEFT": "q",
    "DOWNRIGHT": "c",
    "DOWNLEFT": "z",
    "UPFIRE": "t",
    "RIGHTFIRE": "h",
    "LEFTFIRE": "f",
    "DOWNFIRE": "b",
    "UPRIGHTFIRE": "y",
    "UPLEFTFIRE": "r",
    "DOWNRIGHTFIRE": "n",
    "DOWNLEFTFIRE": "v",
}


class AtariInteractivePolicy(ImageObsDiscreteInteractivePolicy):
    """Interactive policy for Atari environments."""

    def __init__(self, env: Union[atari_env.AtariEnv, vec_env.VecEnv], *args, **kwargs):
        """Builds AtariInteractivePolicy."""
        action_names = (
            env.get_action_meanings()
            if isinstance(env, atari_env.AtariEnv)
            else env.env_method("get_action_meanings", indices=[0])[0]
        )

        # maps the action keys to their semantic names
        action_keys_names = collections.OrderedDict(
            [(ATARI_ACTION_NAMES_TO_KEYS[name], name) for name in action_names],
        )
        super().__init__(
            env.observation_space,
            env.action_space,
            action_keys_names,
            *args,
            **kwargs,
        )


class CartPoleInteractiveExpert(NonTrainablePolicy):
    """Interactive policy for CartPole using keyboard input and live rendering.
    this class depends on NonTrainablePolicy class which is an abstract class. 
    NonTrainablePolicy Inherits from BasePolicy (used by all SB3 policies).
    
    note: _choose_action should be defined in this child class.

    CartPoleInteractivePolicy
    └── inherits from NonTrainablePolicy
            └── inherits from BasePolicy
                    └── inherits from BaseModel (nn.Module)

    - How this class works and what is the objectove?
        -- objective: creating a set of (state , action) which observation comes from gym
        environment and action comes from human expert.
        -- architecture: observation from gym environment and human actions send to NonTrainablePolicy class.
        Actually, actions send through _choose_action to NonTrainablePolicy. In NonTrainablePolicy class,
        _predict depends on _choose_action and it finally sends tensor actions to BasePolicy class.
        BasePolicy.predict() depends on _predict function which returns (state, action) pairs.
        
        policy.predict(obs)
            ⟶ BasePolicy.predict()
                ⟶ NonTrainablePolicy._predict()
                     ⟶ YourPolicy._choose_action(obs) ← Keyboard interaction


    Cartpole observation space is a 4-dimensional vector: cart position, cart velocity, pole angle, and pole angular velocity.
    Cartpole action space is discrete with two actions: 0 (move left) and 1 (move right).
                    
    """

    def __init__(self, env, wand_run, *args, **kwargs):
        """It does not learn, does not predict, and has no weights — 
        its purely a wrapper that lets a human act as the policy."""

        assert isinstance(env, vec_env.VecEnv)
        # self.env_render_func = env.envs[0].render  # real-time rendering
        observation_space = env.observation_space
        action_space = env.action_space
        print('observation_space:', observation_space)
        print('action_space:', action_space)
        # Define two actions: LEFT = 0, RIGHT = 1
        self.key_to_action = {'a': 0, 'd': 1}
        self.wand_run = wand_run  # WandB run for logging if needed
        self.count = 0
        super().__init__(observation_space, action_space)

    def _choose_action(self, obs):
        # self.env_render_func()  # render environment
        """
        It waits for keyboard input ('a' for left, 'd' for right) 
        and outputs discrete actions (0 or 1) based on user choice.
        """

        print("\n🧠 Observation:", obs)
        print("Choose action - Left (a), Right (d):")
        key = input("Your action: ").strip().lower()

        while key not in self.key_to_action:
            key = input("Invalid key! Choose again (a/d): ").strip().lower()
        
        self.count += 1
        # self.wand_run.log({"interaction_count": self.count})
        print(f"\033[96m\nHuman interaction: {self.count}\033[0m")

        return np.array([self.key_to_action[key]])



class CartPoleDiscreteInteractivePolicy(DiscreteInteractivePolicy):
    """Interactive keyboard policy for CartPole using DiscreteInteractivePolicy base."""

    def __init__(self, env: vec_env.VecEnv, *args, **kwargs):
        assert isinstance(env, vec_env.VecEnv)

        # Map keyboard keys to semantic action names (just for display)
        # CartPole has 2 actions: 0 (LEFT), 1 (RIGHT)
        action_keys_names = collections.OrderedDict([
            ("a", "LEFT"),
            ("d", "RIGHT"),
        ])

        self.env_render_func = env.envs[0].render  # Access real-time GUI rendering

        super().__init__(
            observation_space=env.observation_space,
            action_space=env.action_space,
            action_keys_names=action_keys_names,
            *args,
            **kwargs,
        )

    def _render(self, obs: np.ndarray) -> None:
        print("\n🧠 Observation:", obs)
        self.env_render_func()  # This shows the GUI window (if render_mode="human")
        return None

    def _clean_up(self, context: object) -> None:
        pass  # No need to close anything


    