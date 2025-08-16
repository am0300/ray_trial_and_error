"""シミュレーション上で活動するオブジェクト情報."""

import numpy as np

from sample.schemas import Unit


class SimpleBattlefieldUnit:
    """Unit単体の物理・HP管理."""

    def __init__(
        self,
        unit: Unit,
        pos: list[float, float],
        accel: list[float, float],
    ) -> None:
        """Unitの初期化処理."""
        self.unit = unit
        self._hp = unit.max_hp
        self._pos = pos
        self._accel = accel

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
        """名前."""
        return self.unit.name

    @property
    def max_hp(self) -> str:
        """体力の最大値."""
        return self.unit.max_hp

    @property
    def power(self) -> str:
        """攻撃力."""
        return self.unit.power

    @property
    def range(self) -> str:
        """射程."""
        return self.unit.range

    @property
    def speed(self) -> str:
        """移動速度."""
        return self.unit.speed

    @property
    def pos(self) -> str:
        """現在の座標."""
        return self._pos

    @pos.setter
    def pos(self, value: list[float, float]) -> None:
        self._pos = value

    @property
    def hp(self) -> str:
        """現在の体力."""
        return self._hp

    @hp.setter
    def hp(self, value: float) -> None:
        self._hp = value

    @property
    def accel(self) -> list[float, float]:
        """加速強度."""
        return self._accel

    @accel.setter
    def accel(self, value: float) -> None:
        self._accel = value
