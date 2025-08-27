"""."""

import os
from typing import Any

os.environ["RAY_AIR_NEW_OUTPUT"] = "0"
import ray
from kodoku.env_wrapper import EnvWrapper
from kodoku.policy import PolicyMappingManager
from ray import tune
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms.registry import ALGORITHMS
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.models import MODEL_DEFAULTS
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.utils.metrics import (
    ENV_RUNNER_RESULTS,
    EPISODE_RETURN_MEAN,
    NUM_ENV_STEPS_SAMPLED_LIFETIME,
)
from ray.tune.result import TRAINING_ITERATION

# 設定できる項目は同ディレクトリのcli_report_template.yamlを参照
my_multi_agent_progress_reporter = tune.CLIReporter(
    # In the following dict, the keys are the (possibly nested) keys that can be found
    # in RLlib's (PPO's) result dict, produced at every training iteration, and the
    # values are the column names you would like to see in your console reports.
    # Note that for nested result dict keys, you need to use slashes "/" to define the
    # exact path.
    metric_columns={
        TRAINING_ITERATION: "epoch",
        "time_total_s": "total time (s)",
        NUM_ENV_STEPS_SAMPLED_LIFETIME: "ts",
        # RLlib always sums up all agents' rewards and reports it under:
        # result_dict[ENV_RUNNER_RESULTS][EPISODE_RETURN_MEAN].
        f"{ENV_RUNNER_RESULTS}/{EPISODE_RETURN_MEAN}": "episode_return_mean",
        # Because RLlib sums up all returns of all agents, we would like to also
        # see the individual agents' returns. We can find these under the result dict's
        # 'env_runners/module_episode_returns_mean/' key (then the policy ID):
        f"{ENV_RUNNER_RESULTS}/episode_reward_mean": "episode_reward_mean",
        f"{ENV_RUNNER_RESULTS}/policy_reward_mean/atk": "atk_policy_reward_mean",
        f"{ENV_RUNNER_RESULTS}/policy_reward_mean/def": "def_policy_reward_mean",
        # **{
        #     f"{ENV_RUNNER_RESULTS}/module_episode_returns_mean/{pid}": f"return {pid}"
        #     for pid in ["policy1", "policy2", "policy3"]
        # },
    },
)


def get_algorithm_config(
    train_config: dict[str, Any],
    env_class: type[EnvWrapper],
    env_config: dict[str, Any],
    callbacks: type[DefaultCallbacks],
    policy_mapping_manager: PolicyMappingManager | None = None,
) -> AlgorithmConfig:
    """Make AlgorithmConfig."""
    # Geting algorithm config
    _, config = ALGORITHMS[train_config["algorithm"]]()

    # Environment configuration
    policy_mapping_manager = policy_mapping_manager
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
    if policy_mapping_manager is None:
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
            return policy_mapping_manager.get_policy_mapping(
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
            for subpolicy_name in policy_mapping_manager.get_policy_mapping_list(
                policy_name,
            )
        }

    # Initialize trainer
    algorithm_config = (
        config.api_stack(
            enable_env_runner_and_connector_v2=True,
            enable_rl_module_and_learner=True,
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
        .resources(**train_config["resources"])
    )
    return algorithm_config


def tune_function(
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

    # envの登録
    tune.register_env("env", lambda _: env_class(env_config))

    algorithm_config = get_algorithm_config(
        train_config,
        env_class,
        env_config,
        callbacks,
        policy_mapping_manager,
    )

    stop = {f"{ENV_RUNNER_RESULTS}/{EPISODE_RETURN_MEAN}": 200.0}
    stop2 = {"training_iteration": 10}  # epoch数
    results = tune.Tuner(
        algorithm_config.algo_class,
        param_space=algorithm_config,
        run_config=tune.RunConfig(
            stop=stop2,
            verbose=2,
            progress_reporter=my_multi_agent_progress_reporter,
            checkpoint_config=tune.CheckpointConfig(
                checkpoint_frequency=2,
                checkpoint_at_end=True,
            ),
        ),
        tune_config=tune.TuneConfig(
            num_samples=1,  # hyper parameter探索で増やす
        ),
    ).fit()
    print(results)

    trainer = algorithm_config.build_algo()

    trainer.save_to_path("~/new_save_result")

    # print(pretty_print(results))

    # print_network_architecture(
    #     trainer,
    #     policies.keys(),
    # )


def my_experiment(
    config: dict[str, Any],
):
    """."""
    train_config = config["train_config"]
    env_class = config["env_class"]
    env_config = config["env_config"]
    callbacks = config["callbacks"]
    policy_mapping_manager = config["policy_mapping_manager"]

    algorithm_config = get_algorithm_config(
        train_config=train_config,
        env_class=env_class,
        env_config=env_config,
        callbacks=callbacks,
        policy_mapping_manager=policy_mapping_manager,
    )

    # build
    trainer = algorithm_config.build_algo()

    # train for n iterations
    for _ in range(10):
        train_results = trainer.train()
        tune.report(train_results)

    # print(pretty_print(results))

    # print_network_architecture(
    #     trainer,
    #     policies.keys(),
    # )


def custom_tune_function(
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
    # env_config_ref = ray.put(env_config)
    # rayのenv_configはdictしか扱えないためdictにする
    # env_config = {"ray_object_ref": env_config_ref}

    # envの登録
    # tune.register_env("env", lambda _: SimpleBattlefieldEnv(env_config))

    # large data objects
    training_function = tune.with_parameters(
        trainable=my_experiment,
        config={
            "train_config": train_config,
            "env_class": env_class,
            "env_config": env_config,
            "callbacks": callbacks,
            "policy_mapping_manager": policy_mapping_manager,
        },
    )

    """
    # specify resource requests
    training_function = tune.with_resources(
        trainable=training_parameters,
        resources=tune.PlacementGroupFactory(
            [train_config["resources"]],
        ),
        strategy="PACK",
    )
    """

    stop = {"training_iteration": 10}  # epoch数
    tuner = tune.Tuner(
        training_function,
        run_config=tune.RunConfig(
            stop=stop,
            verbose=2,
            progress_reporter=my_multi_agent_progress_reporter,
        ),
        tune_config=tune.TuneConfig(
            num_samples=1,
        ),
    )
    results = tuner.fit()
    best_results = results.get_best_result()

    print(f"evaluation episode returns={best_results}")
