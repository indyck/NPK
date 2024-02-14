import os
import time
print(tf.config.list_physical_devices('GPU'))
#подключение нужных библиотек
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation, BatchNormalization, AveragePooling2D
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
import tensorflow_datasets as tfds  
import tensorflow as tf
import logging
import numpy as np

EPOCHS = 50
BATCH_SIZE = 64


def mnist_make_model(image_w: int, image_h: int):
   #создание модели типа входной_слой=>скрытый_слой=>выходной_слой с ReLU функции активации 
   model = Sequential()
   model.add(Dense(784, activation='relu', input_shape=(image_w*image_h,)))
   model.add(Dense(10, activation='softmax'))
   model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
   return model

def mnist_mlp_train(model):
   (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
   # x_train: 60000x28x28 array, x_test: 10000x28x28 array
   image_size = x_train.shape[1]
   train_data = x_train.reshape(x_train.shape[0], image_size*image_size)
   test_data = x_test.reshape(x_test.shape[0], image_size*image_size)
   train_data = train_data.astype('float32')
   test_data = test_data.astype('float32')
   train_data /= 255.0
   test_data /= 255.0
   #У нас есть 10 цифр, каждый будет представляться так
   # 3 -> [0 0 0 1 0 0 0 0 0 0], 5 -> [0 0 0 0 0 1 0 0 0 0]
   num_classes = 10
   train_labels_cat = keras.utils.to_categorical(y_train, num_classes)
   test_labels_cat = keras.utils.to_categorical(y_test, num_classes)
   print("Тренировка модели...")
   t_start = time.time()
   #Обучение нейронной сети
   model.fit(train_data, train_labels_cat, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1, validation_data=(test_data, test_labels_cat))
   print("Закончила тренировку за:", time.time() - t_start)

model = mnist_make_model(image_w=28, image_h=28)
mnist_mlp_train(model)
model.save('MLP.h5')


