import numpy as np
import dataset
import plot_utils

m=100
xs,ys = dataset.get_beans(m)

plot_utils.show_scatter(xs,ys)

w1=0.1
w2=0.2
b=0.1

x1s = xs[:,0]
x2s = xs[:,1]

def fowrward_propgation(x1s,x2s):
	z = w1*x1s+w2*x2s+b
	a = 1/(1+np.exp(-z))
	return a

plot_utils.show_scatter_surface(xs,ys,fowrward_propgation)
# e = (y-a)**2

for _ in range(500):
	for i in range(m):
		x=xs[i]
		y=ys[i]
		x1=x[0]
		x2=x[1]

		a = fowrward_propgation(x1,x2)

		e =(y-a)**2

		dedw1=2*a*(1-a)*(a-y)*x1
		dedw2=2*a*(1-a)*(a-y)*x2
		dedb=2*a*(1-a)*(a-y)

		alpha=0.01
		w1=w1-alpha*dedw1
		w2=w2-alpha*dedw2
		b=b-alpha*dedb

plot_utils.show_scatter_surface(xs,ys,fowrward_propgation)	