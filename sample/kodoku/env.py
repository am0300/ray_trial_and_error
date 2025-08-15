"""."""

from abc import ABCMeta, abstractmethod
from collections.abc import Callable
from typing import Any

from ray.rllib.env import MultiAgentEnv
from ray.rllib.evaluation.episode_v2 import EpisodeV2


class EnvWrapper(metaclass=ABCMeta):
    """EnvironmentのInterface."""

    reward_range = (-float("inf"), float("inf"))
    observation_space = None
    action_space = None

    @abstractmethod
    def log(self) -> dict:
        """Generate dictionary to log trajectory."""
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

    def render(self, mode: str = "human") -> Any:
        """Render.

        Args:
            mode (str, optional): 'rgb_array' or 'human' or 'ansi'

        """
        raise NotImplementedError


class MultiEnvWrapper(EnvWrapper, MultiAgentEnv, metaclass=ABCMeta):
    """."""

    @abstractmethod
    def get_policy_mapping_fn(self) -> Callable[[str, EpisodeV2], str]:
        """Get policy mapping for multiagent training."""
        raise NotImplementedError
