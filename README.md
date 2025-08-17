# ray trial and error
-----
## About
- 以下のソースコードを改造
- kodoku_samples
  - [KODOKU](https://github.com/Kajune/KODOKU)
- ray_examples
  - [Examples](https://docs.ray.io/en/latest/rllib/rllib-examples.html#examples)

## Instration
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
  - :::note OldAPIStackで安定して実行可能:::
  - manual
    - `./run_kodoku_main.sh`
  - ray_tune
    - `./run_kodoku_main_tune.sh`
  - ray_tune custom experiment
    - `./run_kodoku_main_tune_custom.sh`
- ray_examples
  - 実行したいファイルの実行コマンドのコメントアウトを外して下さい
  - `./run_ray_samples.sh`

## Bug fix
- see issues