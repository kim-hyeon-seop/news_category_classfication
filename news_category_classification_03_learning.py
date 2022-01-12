# -*- coding: utf-8 -*-
"""news_category_classification_03_learning_me.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1AyW3NT_QYygs8yi6z_MrMproUtno4ElT
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *

X_train, X_test, Y_train, Y_test = np.load(
    './models/news_data_max_23_size_15826.npy' , allow_pickle = True
)
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

model = Sequential()
model.add(Embedding(15826, 300, input_length= 23))

# 차원의 저주 때문에 1002차원에서 992단어만 있으면 희소하게 되서 학습이 안됨 , 너무 줄여도 안되고 너무 커서도 안됨(300은 그냥 임의의 값..(보통 300~400사이로 준다.))
# input_length는 단어의 길이
model.add(Conv1D(32, kernel_size= 5, padding = 'same',
                 activation = 'relu'))
model.add(MaxPool1D(pool_size =1))

model.add(LSTM(128, activation = 'tanh',
               return_sequences = True))     ## LSTM 은 무조건 tanh 만쓴다. / 그리고 무조건 return_sequences = True를 줘야함
model.add(Dropout(0.3))                      ## return_sequences 를 True를 안주면 맨 마지막껏만(결과14) 들어감 1~13까지 다시 LSTM 에 넣어주려면 True를 줘야함

model.add(LSTM(64, activation = 'tanh',
               return_sequences = True))
model.add(Dropout(0.3))

model.add(LSTM(64, activation = 'tanh'))     ## 그래서 마지막 LSTM 레이어에는 True를 주지 않아도 된다.
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dense(6, activation= 'softmax'))
print(model.summary())

model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

fit_hist = model.fit(X_train, Y_train, batch_size= 100,
                     epochs = 10, validation_data = (X_test, Y_test))

model.save('./models/news_category_classfication_model_{}.h5'.format(
    fit_hist.history['val_accuracy'][-1]))

plt.plot(fit_hist.history['accuracy'], label='accuracy')
plt.plot(fit_hist.history['val_accuracy'], label='val accuracy')
plt.legend()
plt.show()
