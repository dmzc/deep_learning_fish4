from src.common.types import Action, Coord
from src.common.util import greedy_rellocate_probalities
from collections import defaultdict
import numpy as np


class RandomVAgent:
    _discount_factor: float
    _action_distribution: dict[Action, float]

    _action_probabilities: dict[Coord, tuple[float, float, float, float]]

    # 记录一个回合内的状态，动作 - 转移到的状态 - 奖励
    _memories: list[tuple[Action, Coord, float]]

    # 记录状态发生的次数，这是是跨回合，因为要算多回合平均收益
    _counts: dict[Coord, int]

    # 状态价值表
    _v_table: dict[Coord, float]

    # 是否记录日志
    _log: bool

    # 当前回合
    _episode: int

    def __init__(
        self,
        log=False,
        discount_factor: float = 0.9,
        action_distribution: dict[Action, float] = {
            Action.up: 0.25,
            Action.down: 0.25,
            Action.left: 0.25,
            Action.right: 0.25,
        },
    ):
        self._episode = 0
        self._log = log
        self._discount_factor = discount_factor
        self._action_distribution = action_distribution
        self._action_probabilities = defaultdict(
            lambda: list(action_distribution.values())
        )
        self._memories = []
        self._counts = defaultdict(lambda: 0)
        self._v_table = defaultdict(lambda: 0)

    def choose_action(self, state: Coord) -> Action:
        action_probabilies = self._action_probabilities
        return np.random.choice(
            list(self._action_distribution.keys()), p=action_probabilies.get(state)
        )

    def update(
        self, memory: tuple[Action, Coord, float] = None, is_episode_done=False
    ) -> None:
        """
        更新状态

        @param memory：动作-转移到的状态-即时奖励

        @param is_episode_done 当前回合是否结束，回合结束会更新一次价值表
        """
        self._memories.append(memory)
        if not is_episode_done:
            return
        self._episode += 1
        """
        一轮结束时，根据期间的memories更新价值表
        """
        if self._log:
            print(f"第{self._episode}回合打印开始")
            for idx, (action, state, reward) in enumerate(self._memories):
                print(" " * idx + f"位置{state},动作：{action.name},奖励：{reward}")

            print(f"第{self._episode}回合结束")
        counts = self._counts
        discount_factor = self._discount_factor
        v_table = self._v_table
        last_G = 0
        for action, state, reward in reversed(self._memories):
            counts[state] += 1
            last_G = (
                reward + discount_factor * last_G
            )  # 计算当前state收益，记录为last_G供下一个状态用

            last_v_G = v_table[state]  # 上一个当前状态的G，用于计算均值
            v_table[state] += (last_G - last_v_G) / counts[state]
        # 计算后将当前回合记忆清空
        self._memories.clear()

    def get_v_table(self) -> np.ndarray:
        return self._v_table


class RandomQAgent(RandomVAgent):

    # 探索 - 利用基础概率
    __epislon: float

    __alpha: float

    # 行动价值表
    __q_table: dict[tuple[Coord, Action], float]

    __action_size: int

    __unvisited_action_score: int = (
        0  # 没有走过的动作的分数值，如果给的比较高，那么就比较鼓励探索未知项
    )

    def __init__(
        self,
        log=False,
        discount_factor=0.9,  # 折扣率
        epislon=0.1,  # 探索 - 利用比率
        alpha=0.1,  # 指数平均指数
        action_distribution={
            Action.up: 0.25,
            Action.down: 0.25,
            Action.left: 0.25,
            Action.right: 0.25,
        },
    ):
        super().__init__(log, discount_factor, action_distribution)
        self.__epislon = epislon
        self.__alpha = alpha
        self.__q_table = {}
        self.__action_size = len(self._action_distribution.keys())

    def get_action_by_id(self, id: int) -> Action:
        match id:
            case 0:
                return Action.up
            case 1:
                return Action.down
            case 2:
                return Action.left
            case 3:
                return Action.right

    def update(
        self, memory: tuple[Action, Coord, float] = None, is_episode_done=False
    ) -> None:
        """
        更新状态

        @param memory：动作-转移到的状态-即时奖励

        @param is_episode_done 当前回合是否结束，回合结束会更新一次价值表
        """
        self._memories.append(memory)
        if not is_episode_done:
            return
        self._episode += 1
        """
        一轮结束时，根据期间的memories更新价值表
        """
        if self._log:
            print(f"第{self._episode}回合打印开始")
            for idx, (action, state, reward) in enumerate(self._memories):
                print(f"位置{state},动作：{action.name},奖励：{reward}")

            print(f"第{self._episode}回合结束")
        discount_factor = self._discount_factor
        alpha = self.__alpha
        q_table = self.__q_table
        last_G = 0
        for action, state, reward in reversed(self._memories):
            last_G = (
                reward + discount_factor * last_G
            )  # 计算当前state收益，记录为last_G供下一个状态用

            q_key = (action, state)

            if q_key not in q_table:
                last_v_G = q_table[q_key] = 0
            else:
                last_v_G = q_table[q_key]  # 上一个当前状态的G，用于计算均值
            q_table[q_key] += (
                last_G - last_v_G
            ) * alpha  # 因为动作概率分布会变化，是非稳态问题，所以用指数移动平均，最近的动作最有效

            # 更新动作概率分布
            max_action = None
            max_q = None
            for index in range(self.__action_size):
                q = q_table.get(
                    (state, self.get_action_by_id(index)), self.__unvisited_action_score
                )
                if max_q is None or max_q < q:
                    max_q = q
                    max_action = index
            self._action_probabilities[state] = greedy_rellocate_probalities(
                epislon=self.__epislon,
                max_action=max_action,
                action_size=self.__action_size,
            )

        # 计算后将当前回合记忆清空
        self._memories.clear()

    def get_q_table(self):
        return self.__q_table

    def get_optimal_policy(self):
        policy = {}
        action_probabilities = self._action_probabilities
        for state in action_probabilities.keys():
            probalities = action_probabilities.get(state)
            policy[state] = self.get_action_by_id(
                probalities.index(max(probalities))
            ).name
        return policy
