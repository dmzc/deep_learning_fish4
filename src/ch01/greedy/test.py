from bandit import Bandit
from src.ch01.greedy.agents import Agent, AlphaAgent
import numpy as np
import src.common.ui as ui

arm_num = 10

runs = 200
steps = 1000
epsilon = 0.1

banit = Bandit(arm_num=arm_num)
agent = Agent(action_size=arm_num, epsilon=epsilon)

n_banit = Bandit(arm_num=arm_num, is_stationary=False)  # 非稳态老虎机
alpha_agent = AlphaAgent(
    action_size=arm_num, epsilon=epsilon
)  # 按指数移动平均方式计算奖励


def get_stat_data() -> tuple[str, np.ndarray]:
    agent.reset()
    banit.reset()
    cumulative_reward = 0
    cumulative_rewards = []
    rates = []

    for step in range(steps):
        action = agent.get_action()  # 选择行动
        reward = banit.play(action)  # 实际行动
        agent.update(action=action, reward=reward)  # 更新分数

        cumulative_reward += reward
        cumulative_rewards.append(cumulative_reward)
        rate = cumulative_reward / (step + 1)
        rates.append(rate)
    return ("稳态-单轮-简单平均", rates)


def get_stat_avg_data() -> tuple[str, np.ndarray]:
    all_run_rates = np.zeros((runs, steps))
    for run in range(runs):
        all_run_rates[run] = get_stat_data()[1]
    avg_run_rates = np.average(all_run_rates, axis=0)
    return ("稳态-多轮-简单平均", avg_run_rates)


def get_n_stat_data(use_ema=True) -> tuple[str, np.ndarray]:
    used_agent = None
    if use_ema:
        used_agent = alpha_agent
    else:
        used_agent = agent
    used_agent.reset()
    n_banit.reset()
    cumulative_reward = 0
    cumulative_rewards = []
    rates = []

    for step in range(steps):
        action = used_agent.get_action()
        reward = n_banit.play(action)  # 实际行动
        used_agent.update(action=action, reward=reward)  # 更新分数

        cumulative_reward += reward
        cumulative_rewards.append(cumulative_reward)
        rate = cumulative_reward / (step + 1)
        rates.append(rate)
    label = ""
    if use_ema:
        label = "非稳态-单轮-指数移动平均"
    else:
        label = "非稳态-单轮-简单平均"
    return (label, rates)


"""
use_ema - 是否使用指数移动平均的方式更新奖励
"""


def get_n_stat_avg_data(use_ema=True):
    all_run_rates = np.zeros((runs, steps))
    for run in range(runs):
        all_run_rates[run] = get_n_stat_data(use_ema=use_ema)[1]
    avg_run_rates = np.average(all_run_rates, axis=0)
    if use_ema:
        label = "非稳态-多轮-移动指数平均"
    else:
        label = "非稳态-多轮-简单平均"
    return (label, avg_run_rates)


ui.plot([get_stat_data(), get_stat_avg_data()], "稳态老虎机单轮、多轮平均对比")

ui.plot(
    [get_n_stat_data(use_ema=False), get_n_stat_avg_data(use_ema=False)],
    "非稳态老虎机单轮、多轮平均对比",
)

ui.plot(
    [get_n_stat_data(), get_n_stat_avg_data()], "非稳态老虎机单轮、多轮平均对比（ema）"
)
ui.plot(
    [get_n_stat_avg_data(use_ema=False), get_n_stat_avg_data()],
    "非稳态老虎机简单平均、ema对比",
)
