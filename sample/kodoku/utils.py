"""共通モジュール."""

from typing import Any

from ray.rllib.algorithms import Algorithm
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.env.env_runner import EnvRunner
from ray.rllib.env.multi_agent_episode import MultiAgentEpisode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.policy.tf_policy import TFPolicy
from ray.rllib.policy.torch_policy import TorchPolicy
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger
from ray.rllib.utils.schedules.schedule import Schedule
from ray.rllib.utils.typing import TensorType


class LogCallbacksOldAPI(DefaultCallbacks):
    """OldAPIStack Callback."""

    log_dict: dict = {}
    reward_dict: dict = {}

    def __init__(self):
        super().__init__()
        self.reset()

    def log(self) -> dict:
        return self.log_dict

    def reward(self) -> dict[int, dict]:
        return self.reward_dict

    def reset(self) -> None:
        self.log_dict = {}
        self.reward_dict = {}

    def common_callback(
        self,
        base_env: BaseEnv,
        env_index: int | None = None,
        **kwargs,
    ):
        ei: int = env_index if env_index is not None else 0
        envs = base_env.get_sub_environments()
        scenario_name: str = getattr(envs[ei], "scenario_name", f"scenario_{ei}")

        if scenario_name not in self.log_dict:
            self.log_dict[scenario_name] = {}
        if ei not in self.log_dict[scenario_name]:
            self.log_dict[scenario_name][ei] = []

        return envs[ei], scenario_name, ei

    def on_episode_start(
        self,
        *,
        episode: EpisodeV2,
        env_runner: EnvRunner | None = None,
        metrics_logger: MetricsLogger | None = None,
        base_env: BaseEnv | None = None,
        env_index: int | None = None,
        **kwargs,
    ) -> None:
        if base_env is None:
            return
        env, scenario_name, ei = self.common_callback(base_env, env_index)
        self.log_dict[scenario_name][ei].append([])
        self.log_dict[scenario_name][ei][-1].append(env.log())

    def on_episode_step(
        self,
        *,
        episode: EpisodeV2,
        env_runner: EnvRunner | None = None,
        metrics_logger: MetricsLogger | None = None,
        base_env: BaseEnv | None = None,
        env_index: int | None = None,
        **kwargs,
    ) -> None:
        if base_env is None:
            return
        env, scenario_name, ei = self.common_callback(base_env, env_index)
        if len(self.log_dict[scenario_name][ei]) == 0:
            self.log_dict[scenario_name][ei].append([])
        self.log_dict[scenario_name][ei][-1].append(env.log())

    def on_episode_end(
        self,
        *,
        episode: EpisodeV2,
        env_runner: EnvRunner | None = None,
        metrics_logger: MetricsLogger | None = None,
        base_env: BaseEnv | None = None,
        **kwargs,
    ) -> None:
        self.reward_dict[episode.episode_id] = episode.agent_rewards


class LogCallbacksNewAPI(DefaultCallbacks):
    """NewAPIStack Callback."""

    log_dict: dict = {}
    reward_dict: dict = {}

    def __init__(self):
        super().__init__()
        self.reset()

    def log(self) -> dict:
        return self.log_dict

    def reward(self) -> dict[int, dict]:
        return self.reward_dict

    def reset(self) -> None:
        self.log_dict = {}
        self.reward_dict = {}

    def common_callback(
        self,
        base_env: BaseEnv,
        env_index: int | None = None,
        **kwargs,
    ):
        ei: int = env_index if env_index is not None else 0
        envs = base_env.get_sub_environments()
        scenario_name: str = getattr(envs[ei], "scenario_name", f"scenario_{ei}")

        if scenario_name not in self.log_dict:
            self.log_dict[scenario_name] = {}
        if ei not in self.log_dict[scenario_name]:
            self.log_dict[scenario_name][ei] = []

        return envs[ei], scenario_name, ei

    def on_episode_start(
        self,
        *,
        episode: MultiAgentEpisode,
        env_runner: EnvRunner | None = None,
        metrics_logger: MetricsLogger | None = None,
        base_env: BaseEnv | None = None,
        env_index: int | None = None,
        **kwargs,
    ) -> None:
        if base_env is None:
            return
        env, scenario_name, ei = self.common_callback(base_env, env_index)
        self.log_dict[scenario_name][ei].append([])
        self.log_dict[scenario_name][ei][-1].append(env.log())

    def on_episode_step(
        self,
        *,
        episode: MultiAgentEpisode,
        env_runner: EnvRunner | None = None,
        metrics_logger: MetricsLogger | None = None,
        base_env: BaseEnv | None = None,
        env_index: int | None = None,
        **kwargs,
    ) -> None:
        if base_env is None:
            return
        env, scenario_name, ei = self.common_callback(base_env, env_index)
        if len(self.log_dict[scenario_name][ei]) == 0:
            self.log_dict[scenario_name][ei].append([])
        self.log_dict[scenario_name][ei][-1].append(env.log())

    def on_episode_end(
        self,
        *,
        episode: MultiAgentEpisode,
        env_runner: EnvRunner | None = None,
        metrics_logger: MetricsLogger | None = None,
        base_env: BaseEnv | None = None,
        **kwargs,
    ) -> None:
        self.reward_dict[episode.id_] = episode.get_rewards()


def print_network_architecture(trainer: Algorithm, policies: list[str]) -> None:
    """Print network architectures for policies.

    Args:
            trainer (Algorithm): Trainer object
            policies (List[str]): Policies to print

    """
    for policy_name in policies:
        print(policy_name, "Network Architecture")
        policy = trainer.get_policy(policy_name)
        if policy is not None:
            if isinstance(policy, TorchPolicy):
                print(policy.model)
            elif isinstance(policy, TFPolicy):
                policy.model.base_model.summary()
            else:
                print("Unknown framework:", policy)
        else:
            print("Policy for %s is None" % policy_name)


class ScheduleScaler(Schedule):
    def __init__(self, schedule: Schedule, scale: float = 1.0):
        """Schedule scaler.

        This class wraps existing schedule instance to scale its value

        Args:
            schedule (Schedule): Schedule instance
            scale (float, optional): Scale

        """
        self.schedule = schedule
        self.scale = scale
        self.framework = schedule.framework

    def _value(self, t: int | TensorType) -> Any:
        return self.schedule(t) * self.scale
