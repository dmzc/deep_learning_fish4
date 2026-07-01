import numpy as np

# 重要性采样
x = np.array([1, 2, 3])
pi = np.array([0.1, 0.2, 0.7])

print(f"依据概率求均值：{np.sum(x * pi)}")

n = 100
samples: list[int] = []
for _ in range(n):
    samples.append(np.random.choice(x, p=pi))
print(f"依据抽样（蒙特卡洛），均值：{np.mean(samples)}；方差：{np.var(samples)}")

qi = np.array([0.2, 0.2, 0.6])

samples = []
idxs = np.arange(len(qi))
for _ in range(n):
    idx = np.random.choice(idxs, p=qi)
    sample = x[idx]
    rho = pi[idx] / qi[idx]
    samples.append(rho * sample)

print(f"依据抽样（重要性采样），均值：{np.mean(samples)}；方差：{np.var(samples)}")
