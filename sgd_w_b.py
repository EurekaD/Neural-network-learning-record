import matplotlib.pyplot as plt

import dataset

xs,ys = dataset.get_beans(100)
plt.scatter(xs,ys)

w = 1
b = 1


for _ in range(100):
    for i in range(100):
        x = xs[i]
        y = ys[i]
        dw = 2*w * (x**2) + 2*b*x - 2*x*y
        db = 2*b + 2*w*x - 2*y
        alpha = 0.01
        # 对 w 和 b 分别进行梯度下降，总的来说完成了一次梯度下降
        w = w - alpha*dw
        b = b - alpha*db
    y_pred = w*xs + b
    plt.clf()
    plt.scatter(xs,ys)
    plt.xlim(0,1)
    plt.ylim(0,1.2)
    plt.plot(xs,y_pred)
    plt.pause(0.05)



# y_pred = w * xs + b
# plt.plot(xs,y_pred)
# plt.show()

