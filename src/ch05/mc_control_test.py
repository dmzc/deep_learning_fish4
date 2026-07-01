from src.common.types import GridArgs, Coord, Action
from src.common.grid import Grid
from src.ch05.agents import RandomQAgent

discount_factor = 0.9
action_distributions = {
    Action.up: 0.25,
    Action.down: 0.25,
    Action.left: 0.25,
    Action.right: 0.25,
}

args = GridArgs(
    goal_cell=Coord(x=3, y=0),
    wall_collision_score=-1,
    wall_cells=[Coord(x=1, y=1)],
    mine_cells=[Coord(x=3, y=1)],
    y_max=3,
    x_max=4,
    discount_factor=discount_factor,
    init_action_distributions=action_distributions,
    agent_cell=Coord(y=2, x=0),
)

grid = Grid(args)

agent = RandomQAgent(
    log=True, action_distribution=action_distributions, discount_factor=discount_factor
)

# 轮回次数
episode_count = 1000

for episode_index in range(episode_count):
    # agent的状态是设计成可以跨回合的，所以不用重置
    grid.reset()

    state = grid.get_agent_state()
    while True:
        action = agent.choose_action(state=state)
        next_state, reward, done = grid.step(action=action)
        agent.update((action, next_state, reward), is_episode_done=done)
        state = next_state
        if done:
            break
print(agent.get_optimal_policy())
