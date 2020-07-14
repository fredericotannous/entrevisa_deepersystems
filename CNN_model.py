# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 15:54:47 2020

@author: Frederico
"""

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from sklearn.model_selection import train_test_split
import cv2
import pandas as pd
import glob


from keras.optimizers import adam
from keras.callbacks import Callback

from keras.utils import np_utils #transfor labels into categorical
from keras.datasets import cifar10

import numpy as np
import matplotlib.pyplot as plt

from keras import backend as K
K.common.set_image_dim_ordering('th')

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

#statistics
mean = np.mean(x_train, axis=(0, 1, 2, 3))
std = np.std(x_train, axis = (0, 1, 2, 3))
x_train = (x_train-mean)/(std+1e-7)
x_test = (x_test-mean)/(std+1e-7)

#classes
nClasses = 10
y_train = np_utils.to_categorical(y_train, nClasses)
y_test = np_utils.to_categorical(y_test, nClasses)

print(x_train.shape) #(images, resolution, resolution, rgb)
print(y_train.shape) #(images, classes)

input_shape = (32, 32, 3)

def createModel():
    model = Sequential()
    
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4)) #prevents overfitting
    
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))
    
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))
    
    model.add(Flatten())
    model.add(Dense(512, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nClasses, activation = 'softmax'))
    
    
    return model

K.clear_session()
model = createModel()

#optimizer
AdamOpt = adam(lr = 0.001)
model.compile(optimizer=AdamOpt, loss = 'categorical_crossentropy',
              metrics = ['accuracy'])

model.summary()

# CustomCallback class for logging 

class CustomCallback(Callback):
    def on_epoch_end(self, epoch , logs={}):
        if (epoch % 5 == 0):
            print("Just finished epoch", epoch)
            print('-------------------------')
            print('Loss evaluated on the validation dataset = ', 
                  logs.get('val_loss'))
            print('Accuracy reached train is', 
                  logs.get('acc'))
            print('Accuracy reached Val is', 
                  logs.get('val_acc'))
            return 

# Training of the network

int_batch_size = 256
int_epochs = 50

custom_callbacks = CustomCallback()
fitted_model = model.fit(x_train, y_train, batch_size=int_batch_size, epochs=int_epochs,
                         verbose=0, 
                         validation_data = (x_test, y_test), 
                         callbacks=[custom_callbacks])
