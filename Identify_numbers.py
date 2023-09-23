# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 15:20:52 2022

@author: user
"""

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.datasets import mnist
from tensorflow.python.keras.utils import np_utils
from matplotlib import pyplot as plt


def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    #reshape:為配合後面cnn，做格式轉換。原size為(60000,28,28)轉為(60000,28,28,1))
    #shape[0]為影像的張數
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    
    #類別辨識
    #由於此範例的目的為分類(0~9), 在此使用np_utils.to_categorical來將整數轉為10種類型
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)
    
    return (x_train, y_train), (x_test, y_test)

(x_train, y_train), (x_test, y_test) = load_data()



#建立cnn模型

model = Sequential()
#filters:濾鏡的個數
#kernel_size:濾鏡的大小
#input_shape:輸入的影像大小，前兩個表示影像尺寸，第三個表示影像的頻道數(gray=1,RGB=3)
model.add(Conv2D(filters=3, kernel_size=(3,3), activation='relu',
                 input_shape=(28,28,1), data_format="channels_last"))
#MaxPooling2D:(2,2)表示將2x2個像素縮為1個像素
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=10, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
#第一層，20顆神經元
model.add(Dense(units=20, activation = 'relu'))
#第二層，20顆神經元
model.add(Dense(units=20, activation = 'relu'))
#最後一層，10顆神經元(有0~9共10顆))
#softmax : 將結果轉為機率
model.add(Dense(units=10, activation='softmax'))
#目的是辨識(分類)，分類的運算通常以crossentropy為loss function
#accuracy:辨識的準確度
model.compile(loss='categorical_crossentropy', optimizer='adagrad', metrics=['accuracy'])




#訓練模型
#進行訓練, 訓練過程會存在 train_history 變數中
train_history = model.fit(x=x_train, y=y_train, epochs=5, batch_size=500)

#顯示測試成果(testing data)
result = model.evaluate(x_test, y_test)
print("\nAccuracy of testing data = ", result)











