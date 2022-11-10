import matplotlib.pyplot as plt
import numpy as np

import dataset

xs,ys = dataset.get_beans(100)

# plt.xlabel("Size")
# plt.xlabel("Toxicity")
# plt.scatter(xs,ys)
# plt.show()

w = 0.5
pred = w * xs
# plt.plot(xs,pred)
# plt.show()

es = (ys - pred)**2

# 平均误差
sum_e = es.sum()/100

# w 对 es 的函数图像
ws = np.arange(0,3,0.1)
sum_es = []
for w in ws:
    pred = w * xs
    es = (ys - pred) ** 2
    sum_e = es.sum() / 100
    sum_es.append(sum_e)


plt.xlabel("ws")
plt.xlabel("es")
plt.plot(ws,sum_es)
plt.show()

# 求该图像的最低点对应的w
# 根据公式推导
# sum_es = ( ys- w* xs)**2/100
# sum_es = ys**2 - 2*xs*ys* w + xs*xs* w**2
# -b/2a
w_min = np.sum(xs*ys)/np.sum(xs*xs)
pred = w_min*xs
plt.scatter(xs,ys)
plt.plot(xs,pred)
plt.show()