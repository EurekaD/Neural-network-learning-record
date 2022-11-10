from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
# from keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam,Nadam, SGD
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
(X_train,Y_train),(X_test,Y_test) = mnist.load_data()

# print("X_train.shape"+str(x_train.shape))

# 打印第一张图片
plt.imshow(X_train[0],cmap='gray')
plt.show()

# 把mnist矩阵数据reshape平铺开  28*28-->784
# 除以255归一化
X_train = X_train.reshape(60000,784)/255.0
X_test = X_test.reshape(10000,784)/255.0

# 十个分类，改为独热编码形式
Y_train = to_categorical(Y_train,10)
Y_test = to_categorical(Y_test,10)


model = Sequential()
# 输入数据784个
model.add(Dense(units=256,activation='relu',input_dim=784))

model.add(Dense(units=256,activation='relu'))
model.add(Dense(units=256,activation='relu'))

# 输出层10个神经元，softmax处理输出的概率
model.add(Dense(units=10,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer=SGD(lr=0.05),metrics=['accuracy'])

# 送入训练
model.fit(X_train,Y_train,epochs=5000,batch_size=2048)

loss,accuracy = model.evaluate(X_test,Y_test)
print("Loss"+str(loss))
print("accuracy"+str(accuracy))