# 程序文件 ex9_3.py
from scipy.optimize import fsolve
from scipy.stats import norm

c1 = norm.ppf(0.25, 3, 2)  # 求 0.25 分位数


def fc(c):
    return 1 - norm.cdf(c, 3, 2) - 3 * norm.cdf(c, 3, 2)


c2 = fsolve(fc, 1)  # 求初始值为 1 的方程零点
print("c1=", c1)
print("c2=", c2)
