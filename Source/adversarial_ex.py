# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 04:49:56 2022

@author: arsene
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from random import randint
import matplotlib.pyplot as plt
from PIL import Image
tf.compat.v1.enable_eager_execution()
#from tensorflow.python.ops.numpy_ops import np_config
#np_config.enable_numpy_behavior()

path_to_model = "../Models/model_small_b32.h5"
#medium
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


x_sample = x_sample.reshape(1, 80, 45, 3)

#Pour le medium
#for i in range(80):
#    for j in range(45):
#        for l in range(3):
#            for i2 in range(4):
#                for j2 in range(4):
#                    new[0][i*4+i2][j*4+j2][l]=x_sample[0][i][j][l]
#pour le small

new=x_sample

dataset = tf
ret = model.predict(new, batch_size=None)
print(ret)



def create_adversarial_pattern(input_image, input_label):
  with tf.GradientTape() as tape:
    tape.watch(input_image)
    prediction = model(input_image)#, batch_size=None)
    loss = tf.keras.losses.CategoricalCrossentropy()#input_label, prediction[0])
    loss= loss(input_label, prediction)
    print(tape)

  # Get the gradients of the loss w.r.t to the input image.
  #loss= tf.convert_to_tensor(loss)
  gradient = tape.gradient(loss, input_image)
  #print(gradient)
  # Get the sign of the gradients to create the perturbation
  signed_grad = tf.sign(gradient)
  return signed_grad

x_tensor = tf.convert_to_tensor(new )
y_tensor = tf.convert_to_tensor([y_sample] )
perturbations = create_adversarial_pattern(x_tensor, y_tensor)
#print(perturbations)
#plt.imshow(perturbations[0] * 0.5 + 0.5);



def display_images(image, description):
  prediction = model.predict(image, batch_size=None)
  label = prediction[0].argmax(axis=0)
  confidence = prediction[0][label]
  plt.figure()
  new = np.array(image)
  new = new[0].astype(np.uint8)
  image2 = Image.fromarray(new)
  plt.imshow(image2)
  print("label", label)
  print("confidence", confidence)
  #plt.title('{} \n {} : {:.2f}% Confidence'.format(description,label, confidence*100))
  plt.show()

display_images(new,"normal")
display_images(perturbations,"perturbation")
display_images(new+perturbations,"image perturbé")
print(X_test.shape)