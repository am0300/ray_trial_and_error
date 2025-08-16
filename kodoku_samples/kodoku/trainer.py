"""."""

import pickle
from collections.abc import Callable
from typing import Any

import numpy as np
import ray
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms.registry import ALGORITHMS
from ray.rllib.models import MODEL_DEFAULTS
from ray.rllib.policy.policy import Policy, PolicySpec
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.utils.typing import ResultDict
from ray.tune.logger import pretty_print
from torch.utils.tensorboard import SummaryWriter

from kodoku_samples.kodoku.env_wrapper import EnvWrapper
from kodoku_samples.kodoku.policy import *


class KODOKUTrainer:
    def __init__(
        self,
        log_dir: str,
        env_class: type[EnvWrapper],
        callbacks: type[DefaultCallbacks],
        train_config: dict[str, Any],
        env_config: dict[str, Any],
        policy_mapping_manager: PolicyMappingManager | None = None,
        ray_config: dict = {},
    ):
        """Trainer constructor.

        Args:
            log_dir (str): Location to store training summary, trajectory, weight, etc..
            env_class (Type[EnvWrapper]): Environment class
            train_config (Dict): Training config
            env_config_fn (Callable[[], Tuple[str, Dict]]): Functor to set config on env
            policy_mapping_manager (Optional[PolicyMappingManager], optional): Policy mapping manager instance
            ray_config (Dict, optional): ray configuration for init (memory, num gpu, etc..)

        """
        # # Training configuration
        # Ray initialization
        ray.init(**ray_config)
        assert ray.is_initialized() == True
        # https://github.com/ray-project/ray/issues/53727 対応
        env_config_ref = ray.put(env_config)
        # rayのenv_configはdictしか扱えないためdictにする
        env_config = {"ray_object_ref": env_config_ref}

        # Tensorboard configuration
        self.summaryWriter = SummaryWriter(log_dir)

        # Geting algorithm config
        _, config = ALGORITHMS[train_config["algorithm"]]()

        # Environment configuration
        self.policy_mapping_manager = policy_mapping_manager
        tmp_env = env_class(config=env_config)

        # Training configuration
        model_defaults = MODEL_DEFAULTS
        if train_config["training"]["model"] != "default":
            train_config["training"]["model"] = (
                model_defaults | train_config["training"]["model"]
            )
        else:
            train_config["training"]["model"] = model_defaults

        # Policies configuration
        policy_mapping_fn_tmp = tmp_env.get_policy_mapping_fn()
        if self.policy_mapping_manager is None:
            policy_mapping_function = policy_mapping_fn_tmp
            policies = {
                policy_name: PolicySpec(None, obs_space, act_space, {})
                for policy_name, (obs_space, act_space) in tmp_env.get_spaces().items()
            }
        else:

            def policy_mapping_fn(
                agent_id: str,
                episode: EpisodeV2,
                **kwargs,
            ) -> str:
                return self.policy_mapping_manager.get_policy_mapping(
                    agent_id,
                    policy_mapping_fn_tmp(agent_id, episode, **kwargs),
                    episode,
                )

            policy_mapping_function = policy_mapping_fn
            policies = {
                subpolicy_name: PolicySpec(None, obs_space, act_space, {})
                for policy_name, (
                    obs_space,
                    act_space,
                ) in tmp_env.get_spaces().items()
                for subpolicy_name in self.policy_mapping_manager.get_policy_mapping_list(
                    policy_name,
                )
            }

        # Initialize trainer
        algorithm_config = (
            config.api_stack(
                enable_env_runner_and_connector_v2=False,
                enable_rl_module_and_learner=False,
            )
            .multi_agent(
                policies=policies,
                policy_mapping_fn=policy_mapping_function,
                **train_config["multi_agent"],
            )
            .environment(
                env=env_class,
                env_config=env_config,
                **train_config["environment"],
            )
            .env_runners(**train_config["env_runners"])
            .callbacks(callbacks_class=callbacks)
            .training(**train_config["training"])
        )
        self.trainer = algorithm_config.build_algo()

        # print_network_architecture(
        #     self.trainer,
        #     policies.keys(),
        # )

    def train(
        self,
        num_epochs: int,
        start_epoch: int = 0,
        epoch_callback: Callable[["KODOKUTrainer", int, ResultDict], None]
        | None = None,
    ) -> None:
        """Run train.

        Args:
            num_epochs (int): Number of epochs to train
            start_epoch (int, optional): Start epoch, if you want to start from intermediate
            epoch_callback (Optional[Callable[["KODOKUTrainer", int, ResultDict], None]], optional): Callback function called on each epoch

        """
        for epoch in range(start_epoch, num_epochs):
            print(f"~~~~~~~ epoch: {epoch} ~~~~~~~")
            # self.trainer.reset()
            # self.trainer.callbacks.reset()
            result = self.trainer.train()

            # Update policy mapping
            # if self.policy_mapping_manager is not None:
            #     self.policy_mapping_manager.update_policy_configuration(
            #         self.trainer,
            #         result,
            #         self.reward(),
            #     )

            # Invoke callback
            # if epoch_callback is not None:
            #     epoch_callback(self, epoch, result)

            print(pretty_print(result))

            # self.log_to_tensorboard(epoch, result)

    def log_to_tensorboard(self, epoch, result):
        """."""
        # write log to tensorboard
        if len(result["policy_reward_mean"]) > 0:
            for policy in result["policy_reward_mean"]:
                if policy in result["info"]["learner"]:
                    for k, v in result["info"]["learner"][policy][
                        "learner_stats"
                    ].items():
                        if np.isscalar(v):
                            self.summaryWriter.add_scalar(k + "/" + policy, v, epoch)
                for k in ["mean", "min", "max"]:
                    self.summaryWriter.add_scalar(
                        "EpRet_" + k + "/" + policy,
                        result["policy_reward_" + k][policy],
                        epoch,
                    )

        # else:
        elif DEFAULT_POLICY_ID in result["info"]["learner"]:
            for k, v in result["info"]["learner"][DEFAULT_POLICY_ID][
                "learner_stats"
            ].items():
                if np.isscalar(v):
                    self.summaryWriter.add_scalar(k, v, epoch)
            for k in ["mean", "min", "max"]:
                self.summaryWriter.add_scalar(
                    "EpRet_" + k,
                    result["episode_reward_" + k],
                    epoch,
                )

        if "evaluation" in result:
            if len(result["evaluate"]["policy_reward_mean"]) > 0:
                for policy in result["evaluate"]["policy_reward_mean"]:
                    if policy in result["info"]["learner"]:
                        for k in ["mean", "min", "max"]:
                            self.summaryWriter.add_scalar(
                                "EpRet_" + k + "/" + policy,
                                result["policy_reward_" + k][policy],
                                epoch,
                            )
            else:
                for k in ["mean", "min", "max"]:
                    self.summaryWriter.add_scalar(
                        "EpRet_" + k + "/" + policy,
                        result["policy_reward_" + k][policy],
                        epoch,
                    )

    def add_scalar(self, name, value, epoch):
        self.summaryWriter.add_scalar(name, value, epoch)

    def add_figure(self, name, figure, epoch):
        self.summaryWriter.add_figure(name, figure, epoch)

    def evaluate_kai(self, num_epochs, start_epoch, callback):
        for epoch in range(start_epoch, num_epochs):
            self.trainer.callbacks._callback_list[0].reset()
            result = self.trainer.evaluate()
            print(pretty_print(result))

            # Invoke callback
            if callback is not None:
                callback(self, epoch, result)

        return result

    def evaluate(self) -> ResultDict:
        """Run evaluation.

        Returns:
            ResultDict: Evaluation result

        """
        # self.trainer.callbacks.reset()
        result = self.trainer.evaluate()
        print(pretty_print(result))
        return result

    def shutdown(self) -> None:
        """rayをshutdownする."""
        ray.shutdown()

    def log(self) -> dict:
        """Get training log.

        Returns:
            Dict: log

        """
        return self.trainer.callbacks.log()

    def reward(self) -> dict[int, dict]:
        """Get training rewards.

        Returns:
                List: reward

        """
        return self.trainer.callbacks.reward()

    def get_policy(self, policy_id: str) -> Policy:
        """Get policy instance by id.

        Args:
            policy_id (str): Policy id

        """
        policy = self.trainer.get_policy(policy_id)
        return policy

    def save_policy(self, path: str, policy_id: str | None = None) -> None:
        """Save indivisual or whole policy into file.

        Args:
            path (str): Save file path
            policy_id (Optional[str], optional): Policy id, if you want to select specific policy.

        """
        with open(path, "wb") as f:
            if policy_id is not None:
                pickle.dump({policy_id: self.get_policy(policy_id).get_weights()}, f)
            else:
                pickle.dump(
                    {
                        policy_id: self.get_policy(policy_id).get_weights()
                        for policy_id in self.train_config["multi_agent"][
                            "policies"
                        ].keys()
                    },
                    f,
                )

    def load_policy(self, path: str, policy_id: str | None = None) -> None:
        """Load indivisual or whole policy from file.

        Args:
            path (str): Load file path
            policy_id (Optional[str], optional): Policy id, if you want to select specific policy

        """
        with open(path, "rb") as f:
            weights = pickle.load(f)
            if policy_id is not None:
                if policy_id in weights:
                    self.get_policy(policy_id).set_weights(weights[policy_id])
                else:
                    print(
                        "Policy %s not found in %s. If you want to explicitly load policy with different name, use load_alternative_policy instead."
                        % (policy_id, path),
                    )
            else:
                for policy_id, weight in weights.items():
                    self.get_policy(policy_id).set_weights(weight)

    def load_alternative_policy(
        self,
        path: str,
        from_policy_id: str,
        to_policy_id: str,
    ) -> None:
        """Explicitly load policy with different id.

        Note that policy's model must have exactly same parameters.

        Args:
            path (str): Load file path
            from_policy_id (str): Previous policy id
            to_policy_id (str): New policy id

        """
        with open(path, "rb") as f:
            weights = pickle.load(f)
            self.get_policy(to_policy_id).set_weights(weights[from_policy_id])

    def save_checkpoint(self, path: str) -> None:
        """Save training checkpoint.

        Args:
            path (str): Save dir path

        """
        pickle.dump(self.trainer.__getstate__(), open(path, "wb"))

    def load_checkpoint(self, path: str) -> None:
        """Load training checkpoint.

        Args:
            path (str): Load dir path

        """
        self.trainer.__setstate__(pickle.load(open(path, "rb")))


class SingleAgentTrainer(KODOKUTrainer):
    def __init__(
        self,
        log_dir: str,
        env_class: type[EnvWrapper],
        train_config: dict,
        env_config_fn: Callable[[], tuple[str, dict]],
        ray_config: dict = {},
    ):
        """Trainer ctor.

        Args:
            log_dir (str): Location to store training summary, trajectory, weight, etc..
            env_class (Type[EnvWrapper]): Environment class
            train_config (Dict): Training config
            env_config_fn (Callable[[], Tuple[str, Dict]]): Functor to set config on env
            ray_config (Dict, optional): ray configuration for init (memory, num gpu, etc..)

        """
        super().__init__(
            log_dir,
            env_class,
            train_config,
            env_config_fn,
            None,
            ray_config,
        )

    def get_policy(self, policy_id: str = DEFAULT_POLICY_ID) -> Policy:
        """Get policy instance by id.

        Args:
            policy_id (str): Policy id

        """
        return super().get_policy(policy_id)

    def save_policy(self, path: str) -> None:
        """Save indivisual or whole policy into file.

        Args:
            path (str): Save file path

        """
        super().save_policy(path, DEFAULT_POLICY_ID)

    def load_policy(self, path: str) -> None:
        """Load indivisual or whole policy from file.

        Args:
            path (str): Load file path

        """
        super().load_policy(path, DEFAULT_POLICY_ID)
