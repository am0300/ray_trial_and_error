"""スキーマ定義."""

from typing import Any

from pydantic import BaseModel


class Unit(BaseModel):
    """Unit情報."""

    name: str  # 名前
    max_hp: float  # 体力の最大値
    power: float  # 攻撃力
    range: float  # 射程
    speed: float  # 移動速度


class Simulation(BaseModel):
    """シミュレーション設定."""


class Config(BaseModel):
    """シミュレーション情報."""

    timelimit: int = 500
    depth: float = 2.0
    width: float = 1.0
    atk_spawn_line: float = 1.5
    def_spawn_line: float = 0.5
    def_line: float = 0.5
    atk_num: int = 4
    def_num: int = 3
    atk_unit_hp: float = 1.0
    atk_unit_power: float = 0.1
    atk_unit_range: float = 0.1
    atk_unit_speed: float = 0.05
    def_unit_hp: float = 1.0
    def_unit_power: float = 0.1
    def_unit_range: float = 0.13
    def_unit_speed: float = 0.02

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "Config":
        """辞書からインスタンスを生成する."""
        return cls.model_validate(config)
