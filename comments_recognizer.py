import shopping_data
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense,Embedding,Flatten

x_train,y_train,x_test,y_test = shopping_data.load_data()


vocalen, word_index = shopping_data.createWordIndex(x_train,x_test)
# print(word_index)
# print('词典总词数：',vocalen)

x_train_index = shopping_data.word2Index(x_train,word_index)
x_test_index = shopping_data.word2Index(x_test,word_index)

maxlen = 25
x_train_index = sequence.pad_sequences(x_train_index, maxlen=maxlen)
x_test_index = sequence.pad_sequences(x_test_index, maxlen=maxlen)

model = Sequential()
model.add(Embedding(trainable=False,input_dim=vocalen,output_dim=300, input_length=maxlen))
model.add(Flatten())

model.add(Dense(256,activation='relu'))
model.add(Dense(256,activation='relu'))
model.add(Dense(256,activation='relu'))

model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy']
              )

model.fit(x_train_index,y_train,
          batch_size=512,
          epochs=200)
score, acc = model.evaluate(x_test_index,y_test)
print('Test score :',score)
print('Test accuracy: ',acc)