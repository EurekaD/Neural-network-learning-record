import matplotlib.pyplot as plt

import dataset

xs,ys = dataset.get_beans(100)

plt.xlabel("Size")
plt.xlabel("Toxicity")
plt.scatter(xs,ys)
# plt.show()

w = 0.5
# b = 0
for _ in range(100):
    for i in range(100):
        x = xs[i]
        y = ys[i]

        pred = w * x
        e = y - pred
        alpha = 0.05
        w = w + alpha * e * x

print(w)
pred = w * xs

plt.plot(xs,pred)
plt.show()