import numpy as np


class Agent:

    _epsilon: float  # 探索-利用模式，探索行为的比例
    _Qs: list[float]  # 行为分数
    _Ns: list[float]  # 行为次数
    _action_size: int  # 行为种类

    """
    epsilon - 探索行为的比例
    action_size - 行为的种类
    """

    def __init__(self, epsilon=0.1, action_size=10):
        self._epsilon = epsilon
        self._action_size = action_size
        self._Qs = np.zeros(action_size)
        self._Ns = np.zeros(action_size)

    def update(self, action: int, reward: float):
        self._Ns[action] += 1
        self._Qs += (reward - self._Qs[action]) / self._Ns[action]

    def get_action(self) -> int:
        if np.random.rand() < self._epsilon:
            return np.random.randint(0, self._action_size)
        else:
            return np.argmax(self._Qs)

    def reset(self):
        self._Ns = np.zeros(self._action_size)
        self._Qs = np.zeros(self._action_size)


class AlphaAgent(Agent):
    _alpha: float

    def __init__(self, epsilon=0.1, action_size=10, alpha: float = 0.8):
        super().__init__(epsilon, action_size)
        self._alpha = alpha

    def update(self, action, reward):
        self._Qs[action] += (reward - self._Qs[action]) * self._alpha
