import numpy as np
import math
import random
import nn
import cv2
import matplotlib.pyplot as plt

nn=nn.NeuralNetwork(2,100,2500)

image=np.array(nn.feedFoward([1,4])).reshape(50,50)*255

cv2.imshow("Image",image)

cv2.waitKey(0)



