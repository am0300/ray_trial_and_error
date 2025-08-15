import os
import sys
from collections.abc import Callable
from typing import Any

import cv2
import numpy as np
from gymnasium import spaces
from ray.rllib.env.env_context import EnvContext
from ray.rllib.evaluation.episode_v2 import EpisodeV2

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from sample.kodoku.env import MultiEnvWrapper

# renderで使う色
BLACK = (0, 0, 0)
BLUE = (255, 0, 0)
RED = (0, 0, 255)
RENDER_FPS = 1


class SimpleBattlefieldUnit:
    """Unit単体の物理・HP管理."""

    def __init__(
        self,
        pos: np.ndarray,  # 座標
        hp: float,  # 体力
        power: float,  # 攻撃力
        range: float,  # 射程
        speed: float,  # 移動速度
        name: str,  # 名前
    ) -> None:
        """Unitの初期化処理."""
        self.pos = pos
        self.hp = hp
        self.max_hp = hp
        self.power = power
        self.range = range
        self.speed = speed
        self.accel = np.float32([0, 0])
        self._name = name

    def step(self, zoc: float, pos_min: np.ndarray, pos_max: np.ndarray) -> None:
        """Unitの行動を1step進める."""
        if self.hp <= 0:
            print(f"+++++ unit name: {self.name} is destroyed. +++++")
            return

        self.pos += self.speed * self.accel * (1 - zoc)
        self.pos = np.clip(self.pos, a_min=pos_min, a_max=pos_max)

    def do_damage(self, damage: float) -> None:
        """Unitが受けたダメージ処理."""
        self.hp = max(0, self.hp - damage)

    @property
    def name(self) -> str:
        """Unitの名前."""
        return self._name


class SimpleBattlefieldSimulator:
    """戦場全体の時間進行・戦闘処理."""

    def __init__(
        self,
        depth: float,
        width: float,
        atk_spawn_line: float,
        def_spawn_line: float,
        atk_num: int,
        def_num: int,
        atk_unit_hp: float,
        atk_unit_power: float,
        atk_unit_range: float,
        atk_unit_speed: float,
        def_unit_hp: float,
        def_unit_power: float,
        def_unit_range: float,
        def_unit_speed: float,
        timelimit: int,
        **kwargs,
    ) -> None:
        """シミュレーションの初期化."""
        self.depth = depth  # 戦場の奥行
        self.width = width  # 戦場の幅
        self.timelimit = timelimit  # 戦闘時間

        rng = np.random.default_rng()
        self.atk_units = [
            SimpleBattlefieldUnit(
                np.array(
                    [
                        rng.uniform(atk_spawn_line, depth),
                        rng.uniform(0, width),
                    ],
                ),
                atk_unit_hp,
                atk_unit_power,
                atk_unit_range,
                atk_unit_speed,
                "atk" + str(i),
            )
            for i in range(atk_num)
        ]

        self.def_units = [
            SimpleBattlefieldUnit(
                np.array(
                    [rng.uniform(0, def_spawn_line), rng.uniform(0, width)],
                ),
                def_unit_hp,
                def_unit_power,
                def_unit_range,
                def_unit_speed,
                "def" + str(i),
            )
            for i in range(def_num)
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

        return events


class ExtendedSimulator(SimpleBattlefieldSimulator):
    """防衛シナリオ用拡張シミュレータ."""

    def __init__(self, def_line: float, **kwargs) -> None:
        """シミュレーションの初期化."""
        super().__init__(**kwargs)
        self.def_line = def_line

    def step(self) -> dict[str, Any]:
        """シミュレーションを1step進める."""
        events = super().step()

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


class SimpleBattlefieldEnv(MultiEnvWrapper):
    """非対称型RL環境."""

    def __init__(self, config: EnvContext) -> None:
        """環境の初期化."""
        super().__init__()
        self.config_fn = config["fn"]

        self.sim: ExtendedSimulator | None = None
        self.scenario_name = None
        self.config = None

        self.viewer = None
        self.possible_agents = ["atk0", "atk1", "atk2", "atk3", "def0", "def1", "def2"]
        self.agents = self.possible_agents
        space_dict = self.get_spaces()
        self.observation_spaces = {}
        self.action_spaces = {}
        for agent_id in self.possible_agents:
            policy_id = agent_id[:-1]
            self.observation_spaces[agent_id] = space_dict[policy_id][0]
            self.action_spaces[agent_id] = space_dict[policy_id][1]

    def initialize_simulator(self, config: dict) -> ExtendedSimulator:
        """シミュレータを初期化する."""
        self.events = {}  # シミュレーションのstepで発生したeventの辞書
        return ExtendedSimulator(**config)

    def reset(self, *, seed, options) -> dict[str, Any]:
        """Reset."""
        self.scenario_name, self.config = self.config_fn()
        self.sim = self.initialize_simulator(self.config)
        obs = self.get_obs()
        self.agents = list(obs.keys())
        return obs, {}

    def step(
        self,
        action_dict: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, float], dict[str, bool], dict]:
        """Update the environment's state based on the action dict."""
        for unit in self.sim.atk_units + self.sim.def_units:
            if unit.name in action_dict:
                unit.accel = action_dict[unit.name]

        self.events = self.sim.step()

        rewards = {unit.name: 0 for unit in self.sim.atk_units + self.sim.def_units}
        terminated = {
            unit.name: False for unit in self.sim.atk_units + self.sim.def_units
        }
        truncated = {
            unit.name: False for unit in self.sim.atk_units + self.sim.def_units
        }
        terminated["__all__"] = False
        truncated["__all__"] = False

        rewards, terminated, truncated = self.calc_reward(
            rewards,
            terminated,
            truncated,
            self.sim.atk_units,
            1.0,
        )
        rewards, terminated, truncated = self.calc_reward(
            rewards,
            terminated,
            truncated,
            self.sim.def_units,
            -1.0,
        )
        obs = self.get_obs()
        self.agents = list(obs.keys())

        self.render()
        return obs, rewards, terminated, truncated, {}

    def log(self) -> dict:
        """Log."""
        return {"events": self.events}

    def get_spaces(self) -> dict[str, tuple[spaces.Space, spaces.Space]]:
        """環境の空間情報を取得する."""
        if self.sim is None:
            self.sim = self.initialize_simulator(self.config_fn()[1])
            print("Initialized dummy env to compute spaces.")

        obs_space = spaces.Box(
            low=0,
            high=1,
            shape=(3 * (len(self.sim.atk_units) + len(self.sim.def_units)),),
            dtype=np.float32,
        )
        act_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        space_dict = {
            "atk": (obs_space, act_space),
            "def": (obs_space, act_space),
        }

        return space_dict

    def get_policy_mapping_fn(self) -> Callable[[str, EpisodeV2], str]:
        if self.sim is None:
            self.sim = self.initialize_simulator(self.config_fn()[1])
            print("Initialized dummy env to compute spaces.")

        def policy_mapping_fn(
            agent_id,
            episode,
            atk_units=[unit.name for unit in self.sim.atk_units],
            def_units=[unit.name for unit in self.sim.def_units],
            **kwargs,
        ):
            if agent_id in atk_units:
                return "atk"
            if agent_id in def_units:
                return "def"
            print("Unknown agent %s specified!" % agent_id)
            raise NotImplementedError

        return policy_mapping_fn

    def get_obs(self) -> dict[str, Any]:
        def append_obs(obs, unit):
            obs.append(unit.pos[0] / self.sim.depth)
            obs.append(unit.pos[1] / self.sim.width)
            obs.append(unit.hp / unit.max_hp)
            return obs

        def make_obs(blue_agents, red_agents):
            obs_dict = {}
            for ai_idx, agent in enumerate(blue_agents):
                if agent.hp <= 0:
                    continue

                obs = []
                for i in range(len(blue_agents)):
                    unit = blue_agents[(ai_idx + i) % len(blue_agents)]
                    obs = append_obs(obs, unit)

                for i in range(len(red_agents)):
                    unit = red_agents[i]
                    obs = append_obs(obs, unit)

                obs_dict[agent.name] = np.array(obs)
            return obs_dict

        atk_obs = make_obs(self.sim.atk_units, self.sim.def_units)
        def_obs = make_obs(self.sim.def_units, self.sim.atk_units)
        return atk_obs | def_obs

    def calc_reward(
        self,
        rewards,
        terminated,
        truncated,
        unit_list,
        scale,
        kill_reward_scale=0.1,
        approach_reward_scale=1.0,
        breach_reward=1,
        extinct_reward=1,
        timelimit_reward=0.0,
        timeelapse_reward=0.5,
    ):
        """報酬を計算する."""
        for unit in unit_list:
            if self.events.get("ATK_KILLED"):
                rewards[unit.name] -= (
                    kill_reward_scale * self.events["ATK_KILLED"] * scale
                )
            if self.events.get("DEF_KILLED"):
                rewards[unit.name] += (
                    kill_reward_scale * self.events["DEF_KILLED"] * scale
                )
            if self.events.get("ATK_APPROACH"):
                rewards[unit.name] += (
                    approach_reward_scale * self.events["ATK_APPROACH"] * scale
                )

            rewards[unit.name] -= timeelapse_reward * self.events["TIME_ELAPSE"] * scale

            if "ATK_BREACH" in self.events:
                rewards[unit.name] += breach_reward * scale
                terminated["__all__"] = True
            if "ATK_EXTINCT" in self.events:
                rewards[unit.name] -= extinct_reward * scale
                terminated["__all__"] = True
            if "TIMELIMIT" in self.events:
                rewards[unit.name] -= timelimit_reward * scale
                truncated["__all__"] = True

            if unit.hp <= 0:
                terminated[unit.name] = True
        return rewards, terminated, truncated

    def render(self, mode: str = "human") -> None:
        """opencvを使ってシミュレーションを可視化する."""
        if mode == "human":
            img_scale = 256
            field_delta = 10  # 余白のサイズ(delta + img_scale + delta)
            frame_width = int(self.sim.depth * img_scale + field_delta * 2)
            frame_height = int(self.sim.width * img_scale + field_delta * 2)
            field_height = int(self.sim.width * img_scale)
            field_width = int(self.sim.depth * img_scale)

            img = np.ones((frame_height, frame_width, 3), np.uint8) * 255
            img = cv2.rectangle(
                img,
                (field_delta, field_delta),
                (
                    field_width,
                    field_height,
                ),
                BLACK,
                2,
            )
            img = cv2.line(
                img,
                (int((self.sim.def_line) * img_scale + field_delta), field_delta),
                (int((self.sim.def_line) * img_scale + field_delta), field_height),
                BLUE,
                2,
            )
            for unit in self.sim.atk_units:
                img = cv2.circle(
                    img,
                    (int(unit.pos[0] * img_scale), int(unit.pos[1] * img_scale)),
                    int(unit.range * img_scale),
                    (0, 0, int(255 * unit.hp / unit.max_hp)),
                    1,
                )
                cv2.drawMarker(
                    img,
                    (int(unit.pos[0] * img_scale), int(unit.pos[1] * img_scale)),
                    (0, 0, int(255 * unit.hp / unit.max_hp)),
                    markerSize=15,
                )
            for unit in self.sim.def_units:
                img = cv2.circle(
                    img,
                    (int(unit.pos[0] * img_scale), int(unit.pos[1] * img_scale)),
                    int(unit.range * img_scale),
                    (int(255 * unit.hp / unit.max_hp), 0, 0),
                    1,
                )
                cv2.drawMarker(
                    img,
                    (int(unit.pos[0] * img_scale), int(unit.pos[1] * img_scale)),
                    (int(255 * unit.hp / unit.max_hp), 0, 0),
                    markerSize=15,
                )

            # OpenCV ウィンドウに表示
            cv2.imshow("SimpleBattleFieldViewer", img)
            # cv2.waitKey(int(1000 / fps))  # 描画間隔
            cv2.waitKey(RENDER_FPS)
        else:
            raise NotImplementedError


if __name__ == "__main__":

    def config_fn():
        return "default", {
            "depth": 2.0,  # 横幅
            "width": 1.0,  # 縦幅
            "atk_spawn_line": 1.5,
            "def_spawn_line": 0.5,
            "def_line": 1.0,  # depthの範囲でどのくらいの割合か
            "atk_num": 4,
            "def_num": 3,
            "atk_unit_hp": 1.0,
            "atk_unit_power": 0.1,
            "atk_unit_range": 0.1,
            "atk_unit_speed": 0.05,
            "def_unit_hp": 1.0,
            "def_unit_power": 0.1,
            "def_unit_range": 0.1,
            "def_unit_speed": 0.05,
            "timelimit": 500,
        }

    env_config = {"fn": config_fn}
    env = SimpleBattlefieldEnv(env_config)

    for i in range(1):
        obs = env.reset(12345, {})
        while True:
            # print(obs.keys())
            action = {}
            for agent, (obs_space, act_space) in env.get_spaces().items():
                action[agent] = act_space.sample()
            obs, rewards, terminated, truncated, info = env.step(action)
            # print(f"action: {action}")
            # print(rewards, dones, info, env.log())

            env.render()

            if terminated["__all__"] or truncated["__all__"]:
                break
