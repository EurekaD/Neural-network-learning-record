
import matplotlib.pyplot as plt
import numpy as np

def get_beans(counts):
	xs = np.random.rand(counts)
	xs = np.sort(xs)
	ys = np.zeros(counts)
	for i in range(counts):
		x = xs[i]
		yi = 0.7*x+(0.5-np.random.rand())/50+0.5
		if yi > 0.8:
			ys[i] = 1
	return xs,ys


m = 100
xs,ys = get_beans(m)
plt.title("Size-Toxicity Funtion",fontsize=12)
plt.xlabel('Bean Size')
plt.ylabel('Toxicity')
plt.scatter(xs,ys)

 
w = 0.1
b = 0.1
# z = w*xs + b
# a = 1/(1+np.exp(-z))


# plt.plot(xs,a)
# plt.show()

for _ in range(5000):
	for i in range(100):
		x = xs[i]
		y = ys[i]
		z = w*x + b
		a = 1/(1+np.exp(-z))
		e = (y-a)**2

		dzdw = x
		dzdb = 1
		dadz = a*(1-a)
		deda = 2*(a-y)

		dedw = deda*dadz*dzdw
		dedb = deda*dadz*dzdb

		alpha = 0.05
		w = w-alpha*dedw
		b = b-alpha*deda


	if _%100==0:
		z = w*xs + b
		a = 1/(1+np.exp(-z))

		plt.clf()
		plt.scatter(xs,ys)

		plt.xlim(0,1)
		plt.ylim(0,1.2)
		plt.plot(xs,a)
		plt.pause(0.01)

plt.show()