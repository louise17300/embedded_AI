# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 10:59:18 2022

@author: arsene
"""


from tensorflow import keras
import tensorflow_model_optimization as tfmot

path_to_model_small = "../Models/model_small_b32.h5"
path_to_model_medium = "../Models/model_medium_b32.h5"

model_small = keras.models.load_model(path_to_model_small)
model_medium = keras.models.load_model(path_to_model_medium)

model_small_pruned = tfmot.sparsity.keras.strip_pruning(model_small)
model_medium_pruned = tfmot.sparsity.keras.strip_pruning(model_medium)

model_small_pruned.save("../Models/model_small_pruned.h5")
model_medium_pruned.save("../Models/model_medium_pruned.h5")

# feature_extractor = keras.Model(
#     inputs=model_small,
#     outputs=[layer.output for layer in model_small.layers])

print("done")
