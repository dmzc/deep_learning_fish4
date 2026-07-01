from src.common.types import IGrid, Coord, Action, GridArgs
import numpy as np


# 网格世界
class Grid(IGrid):

    # 即时奖励，网格世界比较简单，即时奖励跟状态绑定。
    __immediate_rewards: np.ndarray

    # 网格参数
    __args: GridArgs

    # 价值表
    __v_table: np.ndarray

    # 最优策略，二维数组(y,x)，值为最优行动索引，分别是上下左右
    __optimal_policy: np.ndarray

    __actions = [Action.up, Action.down, Action.left, Action.right]

    __agent_state: Coord

    def __init__(self, args: GridArgs):
        self.__args = args
        self.do_init()

    def do_init(self) -> None:
        args = self.__args
        self.__v_table = None
        self.__optimal_policy = None
        self.__agent_state = args.agent_cell
        __immediate_rewards = self.__immediate_rewards = np.zeros(
            (args.y_max, args.x_max)
        )
        goal_cell = args.goal_cell
        wall_cells = args.wall_cells
        mine_cells = args.mine_cells
        mine_score = args.mine_score
        reward_cells = args.reward_cells
        reward_score = args.reward_score
        if goal_cell is not None:
            __immediate_rewards[goal_cell.y, goal_cell.x] = args.goal_score
        if wall_cells is not None:
            for wall_cell in wall_cells:
                __immediate_rewards[wall_cell.y, wall_cell.x] = None
        if mine_cells is not None:
            for mine_cell in mine_cells:
                __immediate_rewards[mine_cell.y, mine_cell.x] = mine_score
        if reward_cells is not None:
            for reward_cell in reward_cells:
                __immediate_rewards[reward_cell.y, reward_cell.x] = reward_score
        pass

    @property
    def height(self) -> int:
        return self.__args.y_max

    @property
    def width(self) -> int:
        return self.__args.x_max

    @property
    def actions(self) -> list[Action]:
        return self.__actions

    def get_action_id(self, action: Action) -> int:
        return self.__actions.index(action)

    def get_action_by_id(self, index: int) -> Action:
        return self.actions[index]

    def reward(self, state: Coord, action: Action, next_state: Coord):
        if state == next_state:  # 撞墙
            return self.__args.wall_collision_score
        else:
            return self.__immediate_rewards[next_state.y, next_state.x]

    def next_state(self, state: Coord, action: Action) -> Coord:
        x_delta: int = 0
        y_delta: int = 0
        match action:
            case Action.up:
                if state.y > 0:
                    y_delta = -1
            case Action.down:
                if state.y < self.height - 1:
                    y_delta = 1
            case Action.left:
                if state.x > 0:
                    x_delta = -1
            case Action.right:
                if state.x < self.width - 1:
                    x_delta = 1
        next_state = Coord(state.x + x_delta, state.y + y_delta)
        wall_cells = self.__args.wall_cells
        if wall_cells is not None and (next_state in wall_cells):  # 撞墙，状态维持不变
            next_state = state
        return next_state

    # 获取给定策略下的价值表
    def get_v_table(self, delta=0.00001, log=True):
        if self.__v_table is not None:
            return self.__v_table
        values = self.__immediate_rewards.copy()
        values[~np.isnan(values)] = 0
        __args = self.__args
        discount_factor = __args.discount_factor
        goal_cell = __args.goal_cell
        cnt = 0
        while True:
            if log:
                print(f"迭代次数：{cnt}")
            cnt += 1
            max_delta = 0
            for idx, value in np.ndenumerate(values):
                if np.isnan(value):  # 墙壁格子
                    continue
                y, x = idx
                state = Coord(x=x, y=y)
                if state == goal_cell:  # 重点格子不需要计算，因为它没有下一个状态
                    continue
                score = 0.0
                for action in self.actions:
                    p = self.__get_probability(state, action)
                    if p != 0.0:  # 概率为0的动作不用管
                        next_state = self.next_state(action=action, state=state)
                        reward = self.reward(
                            state=state, action=action, next_state=next_state
                        )
                        score += p * (
                            reward
                            + discount_factor * values[next_state.y, next_state.x]
                        )
                max_delta = max(abs(value - score), max_delta)
                values[state.y, state.x] = score
            if max_delta < delta:
                break
        self.__v_table = values
        return values

    def get_optimal_policy(
        self, use_value_iteration=False, delta: float = 0.00001, use_cache: bool = True
    ):
        if self.__optimal_policy is None or not use_cache:
            (
                self.__value_iteration(delta=delta)
                if use_value_iteration
                else self.__policy_iteration(delta=delta)
            )
        return self.__transform()

    def step(self, action: Action) -> tuple[Coord, float, bool]:
        current_state = self.__agent_state
        next_state = self.next_state(state=current_state, action=action)
        reward = self.reward(state=current_state, action=action, next_state=next_state)
        self.__agent_state = next_state
        return (next_state, reward, self.__agent_state == self.__args.goal_cell)

    def reset(self):
        self.do_init()

    def get_agent_state(self):
        return self.__agent_state

    """
    策略迭代法：
    
    策略评估 - 策略提升 - 策略评估 - 策略提升 ......
    """

    def __policy_iteration(self, delta: float = 0.00001):
        cnt = 0
        while True:
            cnt += 1
            old_policy = self.__optimal_policy
            v_table = self.get_v_table(log=False, delta=delta)
            print(f"第{cnt}轮价值表：{v_table}")
            optimal_policy = self.__optimal_policy = np.zeros(
                (self.height, self.width), dtype=np.uint8
            )
            discount_factor = self.__args.discount_factor
            for idx, value in np.ndenumerate(v_table):
                if np.isnan(value):  # 墙节点
                    continue
                state = Coord(y=idx[0], x=idx[1])
                max_action: Action = None
                score: float = None
                for action in self.actions:
                    next_state = self.next_state(state=state, action=action)
                    r = self.reward(state=state, action=action, next_state=next_state)
                    value = r + discount_factor * v_table[next_state.y, next_state.x]
                    if score is None or value > score:
                        score = value
                        max_action = action
                optimal_policy[state.y, state.x] = self.get_action_id(max_action)
            print(f"第{cnt}轮策略表：{self.__transform()}")
            if old_policy is not None and (old_policy == self.__optimal_policy).all():
                break
            self.__v_table = None

    """
    价值迭代法:
    
    不保留策略，每步选择行动价值最大的动作，以此进行状态价值迭代，知道所有状态价
    值不在有明显的迭代后，既是最佳策略
    """

    def __value_iteration(self, delta: float = 0.00001):
        # 随机价值表，价值迭代法不对价值有什么特殊的要求
        r_v_table = self.__immediate_rewards.copy()
        r_v_table[~np.isnan(r_v_table)] = 0
        optimal_policy = self.__optimal_policy = np.zeros(
            (self.height, self.width), dtype=np.uint8
        )
        print("\n\n价值迭代：")
        cnt = 0
        while True:
            cnt += 1
            discount_factor = self.__args.discount_factor
            max_delta: float = None
            print(f"第{cnt}轮价值表：{r_v_table}")
            for idx, value in np.ndenumerate(r_v_table):
                if np.isnan(value):  # 墙节点
                    continue
                state = Coord(y=idx[0], x=idx[1])
                max_action: Action = None
                max_action_score: float = None

                for action in self.actions:
                    next_state = self.next_state(state=state, action=action)
                    r = self.reward(state=state, action=action, next_state=next_state)
                    value = r + discount_factor * r_v_table[next_state.y, next_state.x]
                    if max_action_score is None or value > max_action_score:
                        max_action = action
                        max_action_score = value
                optimal_policy[state.y, state.x] = self.get_action_id(max_action)
                # 更新价值表
                n_delta = abs(max_action_score - r_v_table[state.y, state.x])
                if max_delta is None or max_delta < n_delta:
                    max_delta = n_delta
                r_v_table[state.y, state.x] = max_action_score
            print(f"第{cnt}轮策略表：{self.__transform()}")
            if max_delta < delta:
                break
        self.__v_table = r_v_table

    def __transform(self) -> list[list[Action]]:

        policies = self.__optimal_policy.tolist()
        wall_cells = self.__args.wall_cells
        for y, policy in enumerate(policies):
            for x, value in enumerate(policy):
                coord = Coord(y=y, x=x)
                if wall_cells is None or coord not in wall_cells:
                    policies[y][x] = self.get_action_by_id(value).name
        return policies

    def __get_probability(self, coord: Coord, action: Action) -> float:
        optimal_policy = self.__optimal_policy
        if optimal_policy is not None:
            t_action = self.get_action_by_id(optimal_policy[coord.y, coord.x])
            return 1.0 if t_action == action else 0.0
        return self.__args.init_action_distributions[action]
