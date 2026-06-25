import numpy as np
import matplotlib.pyplot as plt

# 解决中文显示
plt.rcParams["font.family"] = ["SimHei"]
# 解决负号 '-' 显示方块
plt.rcParams["axes.unicode_minus"] = False


def plot(data: list[tuple[str, np.ndarray]], title: str):

    plt.title(title)
    plt.ylabel("Rates")
    plt.xlabel("Steps")
    for item in data:
        plt.plot(item[1], label=item[0])
    plt.legend()
    plt.show()
