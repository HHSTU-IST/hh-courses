# 程序文件 ex11_6.py
import numpy as np
import pylab as plt
from sklearn.cluster import KMeans

a = np.loadtxt("data11_2.txt")
b = (a - a.min(axis=0)) / (a.max(axis=0) - a.min(axis=0))
SSE = []
K = range(1, len(a) + 1)
for i in K:
    md = KMeans(i).fit(b)
    SSE.append(md.inertia_)
plt.plot(K, SSE, "*-")
plt.show()
