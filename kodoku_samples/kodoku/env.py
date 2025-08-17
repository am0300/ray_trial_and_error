"""環境の定義."""

import os
import sys
from collections.abc import Callable
from typing import Any

import cv2
import numpy as np
import ray
from gymnasium import spaces
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.utils.typing import MultiAgentDict

from kodoku.schemas import Config
from simulator.engine import SimpleBattlefieldSimulator
from simulator.objects import SimpleBattlefieldUnit
from kodoku.env_wrapper import MultiEnvWrapper

# renderで使う色
BLACK = (0, 0, 0)
BLUE = (255, 0, 0)
RED = (0, 0, 255)
RENDER_FPS = 1


class SimpleBattlefieldEnv(MultiEnvWrapper):
    """非対称型RL環境."""

    def __init__(self, config: dict[str, Any]) -> None:
        """環境の初期化."""
        super().__init__()
        if config.get("ray_object_ref"):
            config_ref = config.get("ray_object_ref")
            config = ray.get(config_ref)
        self.config = Config.from_dict(config)

        self.sim: SimpleBattlefieldSimulator | None = None
        self.scenario_name = None

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

    def initialize_simulator(self, config: Config) -> SimpleBattlefieldSimulator:
        """シミュレータを初期化する."""
        self.events = {}  # シミュレーションのstepで発生したeventの辞書
        return SimpleBattlefieldSimulator(config)

    def initialize_agent_episode_flag(self) -> None:
        """episodeの終了判定フラグを初期化する."""
        self.terminated = {
            unit.name: False for unit in self.sim.atk_units + self.sim.def_units
        }
        self.truncated = {
            unit.name: False for unit in self.sim.atk_units + self.sim.def_units
        }
        self.terminated["__all__"] = False
        self.truncated["__all__"] = False

    def reset(
        self,
        *,
        seed: int | None,
        options: dict[Any, Any] | None,
    ) -> tuple[MultiAgentDict, MultiAgentDict]:
        """Reset."""
        # self.scenario_name, self.config = self.config_fn()
        cv2.destroyAllWindows()
        self.sim = self.initialize_simulator(self.config)
        self.initialize_agent_episode_flag()
        obs = self.get_obs(self.terminated, self.truncated)
        self.agents = list(obs.keys())
        return obs, {}

    def step(
        self,
        action_dict: MultiAgentDict,
    ) -> tuple[
        MultiAgentDict,
        MultiAgentDict,
        MultiAgentDict,
        MultiAgentDict,
        MultiAgentDict,
    ]:
        """Update the environment's state based on the action dict."""
        for unit in self.sim.atk_units + self.sim.def_units:
            if unit.name in action_dict:
                unit.accel = action_dict[unit.name]

        self.events = self.sim.step()

        # 報酬の計算と終了判定
        rewards = {unit.name: 0 for unit in self.sim.atk_units + self.sim.def_units}
        rewards, self.terminated, self.truncated = self.calc_reward(
            rewards,
            self.terminated,
            self.truncated,
            self.sim.atk_units,
            1.0,
        )
        rewards, self.terminated, self.truncated = self.calc_reward(
            rewards,
            self.terminated,
            self.truncated,
            self.sim.def_units,
            -1.0,
        )

        # 観測を生成
        obs = self.get_obs(self.terminated, self.truncated)
        self.agents = list(obs.keys())

        self.render()
        return obs, rewards, self.terminated, self.truncated, {}

    def log(self) -> dict:
        """Log."""
        return {"events": self.events}

    def get_spaces(self) -> dict[str, tuple[spaces.Space, spaces.Space]]:
        """環境の空間情報を取得する."""
        if self.sim is None:
            self.sim = self.initialize_simulator(self.config)
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
            self.sim = self.initialize_simulator(self.config)
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

    def get_obs(
        self,
        terminated: dict[str, Any],
        truncated: dict[str, Any],
    ) -> dict[str, Any]:
        """観測を取得する."""

        def append_obs(
            obs: list[float],
            unit: SimpleBattlefieldUnit,
        ) -> list[float]:
            obs.append(unit.pos[0] / self.sim.depth)
            obs.append(unit.pos[1] / self.sim.width)
            obs.append(unit.hp / unit.max_hp)
            return obs

        def make_obs(
            blue_agents: list[SimpleBattlefieldUnit],
            red_agents: list[SimpleBattlefieldUnit],
        ) -> dict[str, list[float]]:
            """観測を生成する."""
            obs_dict = {}
            for idx, agent in enumerate(blue_agents):
                if terminated[agent.name] or truncated[agent.name]:
                    print(f"++++++ skip make obs: {agent.name} ++++++")
                    print(f"terminated: {terminated}")
                    print(f"truncated: {truncated}")
                    continue

                obs = []
                for i in range(len(blue_agents)):
                    # 自分のindex番号を先頭に順番に観測を生成していく
                    unit = blue_agents[(idx + i) % len(blue_agents)]
                    obs = append_obs(obs, unit)

                for i in range(len(red_agents)):
                    # 相手はindex順に観測を生成
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
            rewards[unit.name] -= kill_reward_scale * self.events["ATK_KILLED"] * scale
            rewards[unit.name] += kill_reward_scale * self.events["DEF_KILLED"] * scale
            rewards[unit.name] += (
                approach_reward_scale * self.events["ATK_APPROACH"] * scale
            )

            rewards[unit.name] -= timeelapse_reward * self.events["TIME_ELAPSE"] * scale

            if self.events.get("ATK_BREACH"):
                rewards[unit.name] += breach_reward * scale
                terminated["__all__"] = True
            if self.events.get("ATK_EXTINCT"):
                rewards[unit.name] -= extinct_reward * scale
                terminated["__all__"] = True
            if self.events.get("TIMELIMIT"):
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
                    2,
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
                    2,
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
