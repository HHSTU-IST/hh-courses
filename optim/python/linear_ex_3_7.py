# 程序文件 ex3_7.py
import numpy as np
import pylab as plt  # 加载 Matplotlib 的 Pylab 接口
from PIL import Image
from numpy import linalg as LA

plt.rc("font", size=13)
plt.rc("font", family="SimHei")
a = Image.open("Lena.bmp")  # 返回一个 PIL 图像对象
if a.mode != "L":
    a = a.convert("L")  # 转换为灰度图像
b = np.array(a).astype(float)  # 把图像对象转换为数组
[p, d, q] = LA.svd(b)
m, n = b.shape
R = LA.matrix_rank(b)  # 图像矩阵的秩
plt.figure(0)
plt.plot(np.arange(1, len(d) + 1), d, "k.")
plt.ylabel("奇异值")
plt.title("图像矩阵的奇异值")
CR = []
for K in range(1, int(R / 4), 10):
    plt.figure(K)
    plt.subplot(121)
    plt.title("原图")
    plt.imshow(b, cmap="gray")
    I = p[:, : K + 1] @ (np.diag(d[: K + 1])) @ (q[: K + 1, :])
    plt.subplot(122)
    plt.title("图像矩阵的秩=" + str(K))
    plt.imshow(I, cmap="gray")
    src = m * n
    compress = K * (m + n + 1)
    ratio = (1 - compress / src) * 100  # 计算压缩比率
    CR.append(ratio)
    print(f"Rank={R}:K={K}个：ratio={ratio:.2f}%")
plt.figure()
plt.plot(range(1, int(R / 4), 10), CR, "ob-")
plt.title("奇异值个数与压缩比率的关系")
plt.xlabel("奇异值个数")
plt.ylabel("压缩比率")
plt.show()
