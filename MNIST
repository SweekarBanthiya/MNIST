import tensorflow as tf
from keras.datasets import mnist
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
%matplotlib inline

(trainX, trainy), (testX, testy) = mnist.load_data()
for i in range(0, 15):
    plt.subplot(5,3,1 + i) #subplot(no. of rows for stacking images, no. of cols, image number)
    plt.imshow(trainX[i], cmap = plt.get_cmap('gray')) #plt.cm.binary

trainX = trainX.reshape(trainX.shape[0], 28, 28, 1)
testX = testX.reshape(testX.shape[0], 28, 28, 1)

from tensorflow.keras.utils import to_categorical
trainy = to_categorical(trainy)
testy = to_categorical(testy)

trainX = trainX.astype('float32') # convert pixel int values to float32
testX = testX.astype('float32')
trainX = trainX/255.0 # max pixel value is 255 and min is 0. Hence normalization formula is used here
testX = testX/255.0

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as k
from tensorflow.keras.optimizers import SGD

model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu', kernel_initializer='he_uniform', input_shape=(28,28,1)))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(64, (3,3), activation='relu', kernel_initializer='he_uniform', input_shape=(28,28,1)))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(64, (3,3), activation='relu', kernel_initializer='he_uniform', input_shape=(28,28,1)))
# model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dense(100, activation='relu', kernel_initializer='he_uniform')) # this is 1st hidden layer with 100 neurons
model.add(Dense(10, activation='softmax')) # this is output layer

opt = SGD(learning_rate=0.005, momentum=0.9) # lr is learning rate
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

model_history = model1.fit(trainX, trainy, batch_size = 128, epochs=5, verbose=1, validation_data=(testX, testy), callbacks=EarlyStopping(monitor='loss'))

plt.plot(model_history.history['accuracy'], color='blue',label='train')
plt.plot(model_history.history['val_accuracy'], color='green',label='test')
