import numpy as np


class Bandit:
    """
    多臂老虎机
    """

    __arm_rates: np.ndarray  # 老虎臂得分概率设置

    __is_stationary: bool  # 老虎臂得分概率是不是稳定的

    def __init__(self, arm_num: int, is_stationary=True):
        self.__arm_rates = np.random.rand(arm_num)
        self.__is_stationary = is_stationary

    def play(self, arm_type: int) -> int:
        if not self.__is_stationary:
            self.__arm_rates += 0.1 * np.random.rand(len(self.__arm_rates))
        if self.__arm_rates[arm_type] > np.random.rand():
            return 1
        else:
            return 0

    def reset(self):
        self.__arm_rates = np.random.rand(len(self.__arm_rates))
