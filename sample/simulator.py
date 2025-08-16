"""シミュレータの定義."""

from typing import Any

import numpy as np

from sample.schemas import Config, Unit
from sample.simulation_object import SimpleBattlefieldUnit


class SimpleBattlefieldSimulator:
    """戦場全体の時間進行・戦闘処理."""

    def __init__(
        self,
        config: Config,
    ) -> None:
        """シミュレーションの初期化."""
        self.depth = config.depth  # 戦場の奥行
        self.width = config.width  # 戦場の幅
        self.timelimit = config.timelimit  # 戦闘時間
        self.def_line = config.def_line

        rng = np.random.default_rng()
        self.atk_units = [
            SimpleBattlefieldUnit(
                Unit(
                    name="atk" + str(i),
                    max_hp=config.atk_unit_hp,
                    power=config.atk_unit_power,
                    range=config.atk_unit_range,
                    speed=config.atk_unit_speed,
                ),
                pos=[
                    rng.uniform(config.atk_spawn_line, config.depth),
                    rng.uniform(0, config.width),
                ],
                accel=[0, 0],
            )
            for i in range(config.atk_num)
        ]

        self.def_units = [
            SimpleBattlefieldUnit(
                Unit(
                    name="def" + str(i),
                    max_hp=config.def_unit_hp,
                    power=config.def_unit_power,
                    range=config.def_unit_range,
                    speed=config.def_unit_speed,
                ),
                pos=[
                    rng.uniform(0, config.def_spawn_line),
                    rng.uniform(0, config.width),
                ],
                accel=[0, 0],
            )
            for i in range(config.def_num)
        ]

        self.elapsed = 0

    def step(self) -> dict[str, Any]:
        """シミュレーションを1step進める."""
        self.elapsed += 1

        # calc zoc
        atk_pos = np.array([unit.pos for unit in self.atk_units])
        def_pos = np.array([unit.pos for unit in self.def_units])
        atk_power = np.array([unit.power for unit in self.atk_units])
        def_power = np.array([unit.power for unit in self.def_units])
        atk_range = np.array([unit.range for unit in self.atk_units])
        def_range = np.array([unit.range for unit in self.def_units])
        atk_integrity = np.maximum(
            np.array([unit.hp / unit.max_hp for unit in self.atk_units]),
            0,
        )
        def_integrity = np.maximum(
            np.array([unit.hp / unit.max_hp for unit in self.def_units]),
            0,
        )

        dist = np.linalg.norm(
            atk_pos[np.newaxis, :, :] - def_pos[:, np.newaxis, :],
            axis=2,
        )
        zoc_atk = np.sqrt(np.maximum(1 - 3 * dist / (4 * atk_range), 0)) * atk_integrity
        zoc_def = (
            np.sqrt(np.maximum(1 - 3 * dist.T / (4 * def_range), 0)) * def_integrity
        )

        # calc damage
        damage_atk = np.sum(zoc_atk * atk_power, axis=1)
        damage_def = np.sum(zoc_def * def_power, axis=1)
        atk_killed = 0
        def_killed = 0
        for ui, unit in enumerate(self.atk_units):
            hp_old = unit.hp
            unit.do_damage(damage_def[ui])
            if hp_old > 0 and unit.hp <= 0:
                atk_killed += 1

        for ui, unit in enumerate(self.def_units):
            hp_old = unit.hp
            unit.do_damage(damage_atk[ui])
            if hp_old > 0 and unit.hp <= 0:
                def_killed += 1

        # move units
        for ui, unit in enumerate(self.atk_units):
            unit.step(
                np.sum(zoc_def[ui]),
                np.float32([0, 0]),
                np.float32([self.depth, self.width]),
            )
        for ui, unit in enumerate(self.def_units):
            unit.step(
                np.sum(zoc_atk[ui]),
                np.float32([0, 0]),
                np.float32([self.depth, self.width]),
            )

        # event
        events = {}
        events["ATK_KILLED"] = atk_killed
        events["DEF_KILLED"] = def_killed

        atk_integrity = np.maximum(
            np.array([unit.hp / unit.max_hp for unit in self.atk_units]),
            0,
        )
        def_integrity = np.maximum(
            np.array([unit.hp / unit.max_hp for unit in self.def_units]),
            0,
        )

        if np.sum(atk_integrity > 0) <= len(self.atk_units) / 2:
            events["ATK_EXTINCT"] = 1
        if np.sum(def_integrity > 0) <= len(self.def_units) / 2:
            events["DEF_EXTINCT"] = 1

        if self.elapsed >= self.timelimit:
            events["TIMELIMIT"] = 1
        events["TIME_ELAPSE"] = 1 / self.timelimit

        atk_pos = np.array([unit.pos for unit in self.atk_units])
        atk_integrity = np.maximum(
            np.array([unit.hp / unit.max_hp for unit in self.atk_units]),
            0,
        )

        atk_mean_pos_old = np.mean(atk_pos[:, 0])
        atk_pos = np.array([unit.pos for unit in self.atk_units])
        events["ATK_APPROACH"] = (np.mean(atk_pos[:, 0]) - atk_mean_pos_old) / (
            self.depth - self.def_line
        )

        if (
            np.sum((atk_pos[:, 0] <= self.def_line) & (atk_integrity > 0))
            >= len(self.atk_units) / 2
        ):
            events["ATK_BREACH"] = 1

        return events
