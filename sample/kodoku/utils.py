from typing import *

from ray.rllib.algorithms import Algorithm
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.env.env_runner import EnvRunner
from ray.rllib.policy.tf_policy import TFPolicy
from ray.rllib.policy.torch_policy import TorchPolicy
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger
from ray.rllib.utils.schedules.schedule import Schedule
from ray.rllib.utils.typing import TensorType


class LogCallbacks(DefaultCallbacks):
    log_dict: dict = {}
    reward_dict: dict = {}

    def __init__(self):
        super().__init__()
        self.reset()

    def log(self) -> dict:
        return LogCallbacks.log_dict

    def reward(self) -> dict[int, dict]:
        return LogCallbacks.reward_dict

    def reset(self) -> None:
        LogCallbacks.log_dict = {}
        LogCallbacks.reward_dict = {}

    def common_callback(
        self,
        base_env: BaseEnv,
        env_index: int | None = None,
        **kwargs,
    ):
        ei: int = env_index if env_index is not None else 0
        envs = base_env.get_sub_environments()
        scenario_name: str = getattr(envs[ei], "scenario_name", f"scenario_{ei}")

        if scenario_name not in LogCallbacks.log_dict:
            LogCallbacks.log_dict[scenario_name] = {}
        if ei not in LogCallbacks.log_dict[scenario_name]:
            LogCallbacks.log_dict[scenario_name][ei] = []

        return envs[ei], scenario_name, ei

    def on_episode_start(
        self,
        *,
        episode,
        env_runner: EnvRunner | None = None,
        metrics_logger: MetricsLogger | None = None,
        base_env: BaseEnv | None = None,
        env_index: int | None = None,
        **kwargs,
    ) -> None:
        if base_env is None:
            return
        env, scenario_name, ei = self.common_callback(base_env, env_index)
        LogCallbacks.log_dict[scenario_name][ei].append([])
        LogCallbacks.log_dict[scenario_name][ei][-1].append(env.log())

    def on_episode_step(
        self,
        *,
        episode,
        env_runner: EnvRunner | None = None,
        metrics_logger: MetricsLogger | None = None,
        base_env: BaseEnv | None = None,
        env_index: int | None = None,
        **kwargs,
    ) -> None:
        if base_env is None:
            return
        env, scenario_name, ei = self.common_callback(base_env, env_index)
        if len(LogCallbacks.log_dict[scenario_name][ei]) == 0:
            LogCallbacks.log_dict[scenario_name][ei].append([])
        LogCallbacks.log_dict[scenario_name][ei][-1].append(env.log())

    def on_episode_end(
        self,
        *,
        episode,
        env_runner: EnvRunner | None = None,
        metrics_logger: MetricsLogger | None = None,
        base_env: BaseEnv | None = None,
        **kwargs,
    ) -> None:
        LogCallbacks.reward_dict[episode.episode_id] = episode.agent_rewards


def print_network_architecture(trainer: Algorithm, policies: list[str]) -> None:
    """Print network architectures for policies

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
        """Schedule scaler
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
