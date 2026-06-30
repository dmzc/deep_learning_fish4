from enum import Enum
from dataclasses import dataclass, field


class Action(Enum):
    up = "up"
    down = "down"
    left = "left"
    right = "right"


@dataclass(frozen=True)
class Coord:
    x: int = None
    y: int = None


@dataclass
class GridArgs:
    # x最大值（即横向最大格子数）
    x_max: int = 2

    # y最大值（即纵向最大格子数）
    y_max: int = 1

    # 目标位置,传递这个参数，代表到达这个位置就结束游戏
    goal_cell: Coord = None

    # 目标位置奖励得分，虽然说到达goal就结束，但是算上一个状态需要达到下一步的得
    # 分
    goal_score: int = 1  # 目标位置奖励得分

    # 墙壁格子
    wall_cells: list[Coord] = None

    # 撞墙惩罚值，包括墙壁格子和边界墙，默认撞墙没有惩罚
    wall_collision_score: int = -1

    # 炸弹格子
    mine_cells: list[Coord] = None

    # 炸弹格子分数
    mine_score: int = -1

    # 奖励格子
    reward_cells: list[Coord] = None

    # 奖励格子分数
    reward_score: int = 1

    # 初始动作概率分布
    init_action_distributions: dict[Action, float] = field(
        default_factory=lambda: {
            Action.up: 0,
            Action.down: 0,
            Action.left: 0.5,
            Action.right: 0.5,
        }
    )

    # 长期价值折扣率
    discount_factor: float = 0.9
