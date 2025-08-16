"""EnvironmentのWrapper."""

from abc import ABCMeta, abstractmethod
from collections.abc import Callable
from typing import Any

from ray.rllib.env import MultiAgentEnv
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.utils.typing import AgentID


class EnvWrapper(metaclass=ABCMeta):
    """SingleAgent EnvironmentのWrapper."""

    reward_range = (-float("inf"), float("inf"))
    observation_space = None
    action_space = None

    # @abstractmethod
    # def __init__(self, config: dict[str, Any]) -> None:
    #     """Init."""
    #     raise NotImplementedError

    @abstractmethod
    def log(self) -> dict:
        """Generate dictionary to log trajectory."""
        raise NotImplementedError

    @abstractmethod
    def step(
        self,
        action_dict: dict[AgentID, Any],
    ) -> tuple[Any, float, bool, dict] | dict[str, tuple[Any, float, bool, dict]]:
        """Step.

        Args:
            action_dict (Dict[str, Any]): dictionary with agent name key and action value

        """
        raise NotImplementedError

    def render(self, mode: str = "human") -> Any:
        """Render.

        Args:
            mode (str, optional): 'rgb_array' or 'human' or 'ansi'

        """
        raise NotImplementedError


class MultiEnvWrapper(EnvWrapper, MultiAgentEnv, metaclass=ABCMeta):
    """MultiAgent EnvironmentのWrapper."""

    observation_spaces = None
    action_spaces = None

    @abstractmethod
    def get_policy_mapping_fn(self) -> Callable[[str, EpisodeV2], str]:
        """Get policy mapping for multiagent training."""
        raise NotImplementedError
