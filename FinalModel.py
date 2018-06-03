from load import input_file

import keras
from keras.layers import Merge,Dense,Conv2D,Reshape,Flatten
#from keras.models import model_from_json
from keras.models import Sequential
import pandas as pd
import os

from utils import make_train_test_split
import numpy as np
import tensorflow as tf

inp_x, inp_y, Y = input_file(os.getcwd())


#inp_x = np.reshape(inp_x , (inp_x.shape[0],inp_x.shape[1],inp_x.shape[2],1))

inp_x_train,inp_x_test,Y_train,Y_test = make_train_test_split(inp_x,Y)
inp_y_train,inp_y_test,Y_train,Y_test = make_train_test_split(inp_y,Y)

inp_x_train = np.reshape(inp_x_train , (1,inp_x_train.shape[1],inp_x_train.shape[2],inp_x_train.shape[0]))
inp_x_test = np.reshape(inp_x_test , (1,inp_x_test.shape[1],inp_x_test.shape[2],inp_x_test.shape[0]))
inp_y_train = np.reshape(inp_y_train , (1,inp_y_train.shape[1],inp_y_train.shape[0]))
inp_y_test = np.reshape(inp_y_test , (1,inp_y_test.shape[1],inp_y_test.shape[0]))
Y_train = np.reshape(Y_train , (1,Y_train.shape[1],Y_train.shape[0]))
Y_test = np.reshape(Y_test , (1,Y_test.shape[1],Y_test.shape[0]))

model = Sequential()
model.add(Conv2D(24, kernel_size=(12,10), strides=(1, 1), padding='valid', input_shape=(1024,32,8), activation='relu'))
model.add(Conv2D(48, kernel_size=(14,12), strides=(1, 1), padding='valid', activation='relu'))
model.add(Conv2D(96, kernel_size=(12,4), strides=(2, 2), padding='valid', activation='relu'))
model.add(Conv2D(192, kernel_size=(15,3), strides=(2, 2), padding='valid', activation='relu'))
model.add(Conv2D(384, kernel_size=(21,2), strides=(2, 2), padding='valid', activation='relu'))
model.add(Conv2D(768, kernel_size=(11,1), strides=(4, 4), padding='valid', activation='relu'))
model.add(Conv2D(1024, kernel_size=(14,1), strides=(4, 4), padding='valid', activation='relu'))
a = model.add(Conv2D(1024, kernel_size=(4,1), strides=(1, 1), padding='valid', activation='relu'))
a = model.add(Reshape((1024,8)))(a)
#adding inp_y explicitly to the network
#
#first = Sequential()
#first.add(Dense(8, input_shape=(1024,1), activation='relu'))


#total = Sequential()

#model.add(Merge([model, first],mode='concat'))
model.add(Dense(512,activation='relu'))
model.add(Dense(512,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(100,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(10,activation='relu'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

#history = AccuracyHistory()

model.fit(inp_x_train, Y_train, batch_size=None, epochs=1)
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))

score = model.evaluate([inp_x_test,inp_y_test], Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
##no: of params
