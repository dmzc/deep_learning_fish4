from src.common.enums import GridArgs
from src.common.grid import Grid, Coord
from src.common.enums import Action

# 迭代策略评估
#
# 场景: L1、L2两个格子，L1 - 得0分，L2 - 得1分，撞墙 - 扣1分。有向左、向右两个动
# 作，概率分别是0.5，状态变更是确定的。
args = GridArgs(
    reward_cells=[Coord(x=1, y=0)],
    wall_collision_score=-1,
    wall_cells=None,
    mine_cells=None,
    y_max=1,
    x_max=2,
    init_action_distributions={
        Action.up: 0.0,
        Action.down: 0.0,
        Action.left: 0.5,
        Action.right: 0.5,
    },
)

grid = Grid(args)

print(grid.get_v_table())
