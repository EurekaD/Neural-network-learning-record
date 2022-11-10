from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
# from keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam,Nadam, SGD
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from keras.layers import Conv2D
from keras.layers import AveragePooling2D
from keras.layers import Flatten

# 导入数据集
(X_train,Y_train),(X_test,Y_test) = mnist.load_data()

# print("X_train.shape"+str(x_train.shape))

plt.imshow(X_train[0],cmap='gray')
plt.show()

# 将数组reshape为6万个28乘28的图片矩阵（gray图，只有一个通道，如果是rgb，则是3个通道），并把它们全部除以255来归一化数据
X_train = X_train.reshape(60000,28,28,1)/255.0
X_test = X_test.reshape(10000,28,28,1)/255.0

# 将 y 原类别数据（0，1，2，...）转换为二进制的矩阵表示，长度为10.即为将原有的类别向量转换为独热编码的形式
Y_train = to_categorical(Y_train,10)
Y_test = to_categorical(Y_test,10)

model = Sequential()
# conv2D 二维卷积层,（卷积核数量6，卷积核尺寸5*5，卷积核右移下移的步长都为1，输入的形状28*28*1通道，不加填充模式（越卷越小），激活函数）
# 这一层的结果是将28*28的数据卷成 24*24*6
model.add(Conv2D(filters=6,kernel_size=(5,5),strides=(1,1),input_shape=(28,28,1),padding='valid',activation='relu'))

# 池化层，结果变为12*12*6
model.add(AveragePooling2D(pool_size=(2,2)))

# 第二层卷积层，结果变为8*8*16
model.add(Conv2D(filters=16,kernel_size=(5,5),strides=(1,1),padding='valid',activation='relu'))

# 池化层，结果变为4*4*16
model.add(AveragePooling2D(pool_size=(2,2)))

# 平铺开
model.add(Flatten())

model.add(Dense(units=120,activation='relu'))
model.add(Dense(units=84,activation='relu'))

# 输出层
model.add(Dense(units=10,activation='softmax'))

model.compile(loss='mean_squared_error',optimizer=SGD(lr=0.05),metrics=['accuracy'])
model.fit(X_train,Y_train,epochs=5000,batch_size=2048)

loss,accuracy = model.evaluate(X_test,Y_test)
print("Loss"+str(loss))
print("accuracy"+str(accuracy))