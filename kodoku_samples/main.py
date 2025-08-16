"""非対称 Multi Agent 強化学習の実行."""

import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from kodoku_samples.kodoku.env import SimpleBattlefieldEnv
from kodoku_samples.kodoku.trainer import KODOKUTrainer
from kodoku_samples.kodoku.utils import LogCallbacksOldAPI as LogCallbacks

# from kodoku_sample.kodoku.utils import LogCallbacksNewAPI as LogCallbacks

simulator_config = {
    "depth": 2.0,
    "width": 1.0,
    "atk_spawn_line": 1.5,
    "def_spawn_line": 0.5,
    "def_line": 0.5,
    "atk_num": 4,
    "def_num": 3,
    "atk_unit_hp": 1.0,
    "atk_unit_power": 0.1,
    "atk_unit_range": 0.2,
    "atk_unit_speed": 0.05,
    "def_unit_hp": 1.0,
    "def_unit_power": 0.1,
    "def_unit_range": 0.2,
    "def_unit_speed": 0.02,
    "timelimit": 500,
}


def callback(trainer: KODOKUTrainer, epoch: int, result: dict):
    log = trainer.log()
    json.dump(log, open("./log_dir/latest_log.json", "w"), indent=2)


if __name__ == "__main__":
    trainer = KODOKUTrainer(
        log_dir="./log_dir",
        env_class=SimpleBattlefieldEnv,
        callbacks=LogCallbacks,
        train_config=json.load(open("./kodoku_samples/train_config.json")),
        env_config=simulator_config,
    )

    trainer.train(10, epoch_callback=callback)
    trainer.evaluate()
    trainer.shutdown()
