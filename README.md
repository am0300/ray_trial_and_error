# ray trial and error
-----

## About
- 以下のソースコードを改造
- kodoku_samples
  - [KODOKU](https://github.com/Kajune/KODOKU)
- ray_examples
  - [Examples](https://docs.ray.io/en/latest/rllib/rllib-examples.html#examples)

## Installation
- build image and up container
  - `./docker_build.sh`
- up container
  - `./docker_up.sh`
- stop container
  - `./docker_stop.sh`
- stop and remove container
  - `./docker_down.sh`

## Execution
- kodoku_samples
  - OldAPIStackで安定して実行可能
  - manual
    - `./run_kodoku_main.sh`
  - ray_tune
    - `./run_kodoku_main_tune.sh`
  - ray_tune custom experiment
    - `./run_kodoku_main_tune_custom.sh`
  - NewAPIStackにする方法
    - kodoku_samples/main.pyのimportをLogCallbacksNewAPIに変更
    - kodoku_samples/kodoku/trainer.pyの以下をTrueに変更
      ```
      enable_env_runner_and_connector_v2=True,
      enable_rl_module_and_learner=True,
      ```
    - NewAPIStackは厳格になっておりterminatedされたagentがobsに含まれるバグにより落ちる
- ray_examples
  - 実行したいファイルの実行コマンドのコメントアウトを外して下さい
  - `./run_ray_samples.sh`

## Bug fix
- see issues