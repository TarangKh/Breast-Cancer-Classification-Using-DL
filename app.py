# -*- coding: utf-8 -*-

## Importing Required Packages
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import os
import pickle
import random

## Loading the Models
path = os.getcwd()
modelname = "Models\model.keras"
scalername = "Models\scaler.pkl"
modelpath = os.path.join(path, modelname)
scalerpath = os.path.join(path, scalername)
model = keras.models.load_model(modelpath)

with open(scalerpath, "rb") as sc:
  scaler = pickle.load(sc)


## Prediction System
data = load_breast_cancer()
target = data.target
random_index = random.randint(0, len(data))
random_input = data.data[random_index]
actual_label = target[random_index]
nparray = np.asarray(random_input)
nparray_reshaped = nparray.reshape(1, -1)

standard_random_input = scaler.transform(nparray_reshaped)
pred = model.predict(standard_random_input)

predicted_label = np.argmax(pred)

print("---- BREAST CANCER CLASSIFICATION ----")
print("FEATURES")
print(data.feature_names)
print("RANDOM SAMPLE CORRESPONDING TO FEATURES")
print(random_input)

print("ACTUAL RESULT: ", end="")
print("The tumor is Benign.") if actual_label == 1 else print("The tumor is Malignant.")
print("\nPREDICTED RESULT: ", end="")
print("The tumor is Benign.") if predicted_label == 1 else print("The tumor is Malignant.")

if actual_label == predicted_label:
  print("MODEL PREDICTION IS CORRECT.")
else:
  print("MODEL PREDICTION IS INCORRECT.")



# df = load_breast_cancer(as_frame = True).frame
# X = df.drop("target",axis = 1)
# y = df["target"]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 42)

# X_test_std = scaler.transform(X_test)
# print(model.evaluate(X_test_std, y_test))

