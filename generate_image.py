import numpy as np
import cv2
import tensorflow as tf
import math
import random
from tensorflow import keras
from keras.layers import Flatten,Dense,Dropout
from keras.models import Sequential
from keras.activations import relu,softmax
from sklearn.model_selection import train_test_split
from keras.datasets import mnist


# image=np.random.normal(size=(500,500))*255


# cv2.imshow("Image",image)

# cv2.waitKey(0)

noise=np.array([np.random.normal() for r in range(10)])

model=Sequential()
model.add(Flatten(input_shape=(1,10)))
model.add(Dense(128,activation=relu))
model.add(Dense(160000,activation=relu))

noise=noise.reshape(1,10)

#print(noise.shape)

prediction=model.predict(np.array([noise]))
image=prediction.reshape(400,400)*255

cv2.imshow("Image",image)

cv2.waitKey(0)




