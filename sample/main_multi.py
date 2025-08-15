import json
import os
import sys
from typing import *

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from sample.battle_field_env import SimpleBattlefieldEnv
from sample.kodoku.trainer import KODOKUTrainer


def config_fn():
    return "default", {
        "depth": 2.0,
        "width": 1.0,
        "atk_spawn_line": 1.5,
        "def_spawn_line": 0.5,
        "def_line": 0.5,
        "atk_num": 4,
        "def_num": 3,
        "atk_unit_hp": 1.0,
        "atk_unit_power": 0.1,
        "atk_unit_range": 0.1,
        "atk_unit_speed": 0.05,
        "def_unit_hp": 1.0,
        "def_unit_power": 0.1,
        "def_unit_range": 0.13,
        "def_unit_speed": 0.02,
        "timelimit": 500,
    }


def callback(trainer: KODOKUTrainer, epoch: int, result: Dict):
    log = trainer.log()
    json.dump(log, open("./log_dir/latest_log.json", "w"), indent=2)


if __name__ == "__main__":
    trainer = KODOKUTrainer(
        log_dir="./log_dir",
        env_class=SimpleBattlefieldEnv,
        train_config=json.load(open("./sample/train_config.json")),
        env_config_fn=config_fn,
    )

    trainer.train(10, epoch_callback=callback)
    trainer.evaluate()
