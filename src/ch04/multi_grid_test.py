from src.common.enums import GridArgs
from src.common.grid import Grid, Coord
from src.common.enums import Action

args = GridArgs(
    goal_cell=Coord(x=3, y=0),
    wall_collision_score=-1,
    wall_cells=[Coord(x=1, y=1)],
    mine_cells=[Coord(x=3, y=1)],
    y_max=3,
    x_max=4,
    init_action_distributions={
        Action.up: 0.25,
        Action.down: 0.25,
        Action.left: 0.25,
        Action.right: 0.25,
    },
)

grid = Grid(args)

print(grid.get_v_table())  # 迭代策略评估

grid.get_optimal_policy()  # 策略迭代法求最优策略

grid.get_optimal_policy(use_value_iteration=True, use_cache=False)  # 价值迭代法求最优策略
