import numpy as np
import cv2


image=np.random.normal(size=(500,500))*255


cv2.imshow("Image",image)

cv2.waitKey(0)