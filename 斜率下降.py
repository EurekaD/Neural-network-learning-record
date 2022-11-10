import matplotlib.pyplot as plt

import dataset

xs,ys = dataset.get_beans(100)

# plt.xlabel("Size")
# plt.xlabel("Toxicity")
# plt.scatter(xs,ys)
# plt.show()


w = 1
y_pred = w * xs
# plt.plot(xs,y_pred)


for _ in range(100):
    for i in range(100):
        x = xs[i]
        y = ys[i]
        k = 2 * (x**2) * w - 2 * x * y
        alpha = 0.1
        w = w - alpha * k
        plt.clf()
        plt.xlim(0,1)
        plt.ylim(0,1.2)
        plt.scatter(xs,ys)
        y_pred = w * xs
        plt.plot(xs,y_pred)
        plt.pause(0.01)

# plt.scatter(xs,ys)
# y_pred  = w * xs
# plt.plot(xs,y_pred)
# plt.show()