# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 04:49:56 2022

@author: arsene
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from random import randint

path_to_model = "../Models/model_small_b32.h5"

path_xtest = "../Models/validation_set/validation_x_set.npy"
path_ytest = "../Models/validation_set/validation_y_set.npy"

model = keras.models.load_model(path_to_model)

X_test = np.load(path_xtest).astype(dtype=np.float32)
Y_test = np.load(path_ytest).astype(dtype=np.float32)

rand_sample = randint(0, X_test.shape[0]-1)
x_sample = X_test[rand_sample]
s_sample = x_sample/255
y_sample = Y_test[rand_sample]
tmp = y_sample.argmax(axis=0)
#print(y_sample)
print("Chosen input's corresponding label is "+str(tmp)+" according to y_test")


x_sample = x_sample.reshape(1, 80, 45, 3)#80, 45 pour le small  320, 180
#print(x_sample)
dataset = tf
ret = model.predict(x_sample,
                    batch_size=None)
print(ret)
