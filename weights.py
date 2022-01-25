import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense,Flatten,Convolution2D,Dropout,MaxPooling2D
from keras.models import  Sequential
from keras.activations import relu,softmax,sigmoid
import numpy as np
import math
import matplotlib.pyplot as plt
import cv2
import random
from tensorflow.keras.utils import to_categorical

image=np.random.uniform(size=(1,2))

print(image)

X=np.array([image])
y=np.array([1])

y=to_categorical(y)



model=Sequential()
model.add(Flatten(input_shape=image.shape))
model.add(Dense(128,activation=relu))
model.add(Dense(2,activation=softmax))
# model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
# model.fit(X,y,epochs=50)