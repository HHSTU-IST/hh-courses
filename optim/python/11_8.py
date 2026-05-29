# 程序文件 ex11_8.py
import numpy as np
import pandas as pd
import pylab as plt
import scipy.cluster.hierarchy as sch

a = pd.read_excel("data11_8.xlsx", header=None)
b = a.values.T
b = np.triu(b, k=1)  # 取对角线上方元素
r = b[np.nonzero(b)]
d = 1 - abs(r)
z = sch.linkage(d, "complete")
sch.dendrogram(z, labels=range(1, 15))
plt.show()
