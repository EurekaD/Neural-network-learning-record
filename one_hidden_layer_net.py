import matplotlib.pyplot as plt
import numpy as np

def get_beans(counts):
	xs = np.random.rand(counts)*2
	xs = np.sort(xs)
	ys = np.zeros(counts)
	for i in range(counts):
		x = xs[i]
		yi = 0.7*x+(0.5-np.random.rand())/50+0.5
		if yi > 0.8 and yi < 1.4:
			ys[i] = 1


	return xs,ys


def sigmoid(x):
	return 1/(1+np.exp(-x))


m = 100
xs,ys = get_beans(m)
plt.title("Size-Toxicity Funtion",fontsize=12)
plt.xlabel('Bean Size')
plt.ylabel('Toxicity')
plt.scatter(xs,ys)

 
w11_1 = np.random.rand()
b1_1 = np.random.rand()
w12_1 = np.random.rand()
b2_1 = np.random.rand()

w11_2 = np.random.rand()
w21_2 = np.random.rand()
b1_2 = np.random.rand()


# 前向传播
def forward_propgation(xs):
	z1_1 = w11_1*xs + b1_1
	a1_1 = sigmoid(z1_1)

	z2_1 = w12_1*xs + b2_1
	a2_1 = sigmoid(z2_1)

	z1_2 = w11_2*a1_1 + w21_2*a2_1 + b1_2
	a1_2 = sigmoid(z1_2)
	return z1_1,a1_1,z2_1,a2_1,z1_2,a1_2

z1_1,a1_1,z2_1,a2_1,z1_2,a1_2=forward_propgation(xs)

plt.plot(xs,a1_2)
plt.show()

for _ in range(5000):
	for i in range(100):
		x = xs[i]
		y = ys[i]

		z1_1,a1_1,z2_1,a2_1,z1_2,a1_2=forward_propgation(x)

		e = (y-a1_2)**2

		deda1_2=2*(a1_2-y)
		da1_2dz1_2=a1_2*(1-a1_2)

		dz1_2dw11_2=a1_1
		dz1_2dw21_2=a2_1
		dz1_2db1_2=1

		dedw11_2=deda1_2*da1_2dz1_2*dz1_2dw11_2
		dedw21_2=deda1_2*da1_2dz1_2*dz1_2dw21_2
		dedb1_2=deda1_2*da1_2dz1_2*dz1_2db1_2

		# 省略一点
		dedw11_1=2*(a1_2-y)*a1_2*(1-a1_2)*w11_2*x*a1_1*(1-a1_1)
		dedb1_1=2*(a1_2-y)*a1_2*(1-a1_2)*w11_2*a1_1*(1-a1_1)

		dedw12_1=2*(a1_2-y)*a1_2*(1-a1_2)*w21_2*x*a2_1*(1-a2_1)
		dedb2_1=2*(a1_2-y)*a1_2*(1-a1_2)*w21_2*a2_1*(1-a2_1)

		# 梯度下降
		alpha=0.03
		w11_2 = w11_2-alpha*dedw11_2
		w21_2 = w21_2-alpha*dedw21_2
		b1_2 = b1_2-alpha*dedb1_2

		w11_1 = w11_1-alpha*dedw11_1
		w12_1 = w12_1-alpha*dedw12_1
		b1_1 = b1_1-alpha*dedb1_1
		b2_1 = b2_1-alpha*dedb2_1


	if _%100==0:

		plt.clf()
		plt.scatter(xs,ys)
		z1_1,a1_1,z2_1,a2_1,z1_2,a1_2=forward_propgation(xs)
		plt.plot(xs,a1_2)
		plt.pause(0.01)

plt.show()