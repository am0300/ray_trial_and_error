from abc import ABCMeta, abstractmethod
from collections.abc import Callable
from typing import Any

import gymnasium as gym
from gymnasium import spaces
from ray.rllib.env import MultiAgentEnv
from ray.rllib.env.env_context import EnvContext
from ray.rllib.evaluation.episode_v2 import EpisodeV2


class EnvWrapper(metaclass=ABCMeta):
    reward_range = (-float("inf"), float("inf"))
    observation_space = None
    action_space = None

    def __init__(self, config: EnvContext):
        """EnvWrapper ctor.

        Args:
            env_fn (Callable[[Dict], Any]): Functor to generate env object
            config_fn (Callable[[], [str, Dict]]): Functor to generate config for env

        """
        super().__init__()

        self.config_fn = config["fn"]

        self.env = None
        self.scenario_name = None
        self.config = None

    @abstractmethod
    def log(self) -> dict:
        """Generate dictionary to log trajectory."""
        raise NotImplementedError

    @abstractmethod
    def initialize_env(self, config: dict) -> Any:
        """Env factory function.

        Args:
            config (Dict): Env config

        """
        raise NotImplementedError

    @abstractmethod
    def get_spaces(
        self,
    ) -> (
        tuple[spaces.Space, spaces.Space] | dict[str, tuple[spaces.Space, spaces.Space]]
    ):
        """Get dictionary with agent name key and  observation and action spaces value."""
        raise NotImplementedError

    @abstractmethod
    def step(
        self,
        action: Any,
    ) -> tuple[Any, float, bool, dict] | dict[str, tuple[Any, float, bool, dict]]:
        """Step.

        Args:
            action (Dict[str, Any]): dictionary with agent name key and action value

        """
        raise NotImplementedError

    @abstractmethod
    def reset_impl(self) -> Any | dict[str, Any]:
        """reset_impl."""
        raise NotImplementedError

    def render(self, mode: str = "human") -> Any:
        """Render.

        Args:
            mode (str, optional): 'rgb_array' or 'human' or 'ansi'

        """
        raise NotImplementedError

    def reset(self, seed, options) -> Any | dict[str, Any]:
        """Reset.

        Returns:
            Union[Any, Dict[str, Any]]: dictionary with agent name key and observation value

        """
        self.scenario_name, self.config = self.config_fn()
        self.env = self.initialize_env(self.config)
        return self.reset_impl()


class MultiEnvWrapper(EnvWrapper, MultiAgentEnv, metaclass=ABCMeta):
    @abstractmethod
    def get_policy_mapping_fn(self) -> Callable[[str, EpisodeV2], str]:
        """Get policy mapping for multiagetn training."""
        raise NotImplementedError


class GymEnv(EnvWrapper, gym.Env):
    def log(self) -> dict:
        return {}

    def initialize_env(self, config: dict) -> Any:
        if self.env is None:
            env = gym.make(**config)
        else:
            env = self.env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        # self.reward_range = env.reward_range
        return env

    def get_spaces(self) -> tuple[spaces.Space, spaces.Space]:
        if self.env is None:
            self.env = self.initialize_env(self.config_fn()[1])
        return self.env.observation_space, self.env.action_space

    def step(self, action: Any) -> tuple[Any, float, bool, dict]:
        return self.env.step(action)

    def reset_impl(self) -> Any:
        return self.env.reset()

    def render(self, mode: str = "rgb_array") -> Any:
        return self.env.render(mode)
