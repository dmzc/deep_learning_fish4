def greedy_rellocate_probalities(
    action_size: int,  # 动作种类
    max_action: int,  # 最大动作索引
    epislon: float = 0.1,  # 探索-利用概率
):
    probabilities = [0] * action_size
    avg_epislon = epislon / action_size
    for index in range(action_size):
        if index == max_action:
            probabilities[index] = avg_epislon + 1 - epislon
        else:
            probabilities[index] = avg_epislon
    return probabilities
