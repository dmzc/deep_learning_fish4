from abc import ABC
import numpy as np
from src.common.enums import Action, Coord


class IGrid(ABC):
    @property
    def height(self) -> int: ...

    @property
    def width(self) -> int: ...

    def reward(self, state: Coord, action: Action, next_state: Coord) -> float: ...

    def next_state(self, state: Coord, action: Action) -> Coord: ...

    def get_v_table() -> np.ndarray: ...
