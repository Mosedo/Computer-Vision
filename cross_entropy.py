import math
import numpy as np
import random


def single_input_crossentropy_loss(output,y):
    # -1(y*log(output)+(1-y)*log(1-output))
    loss= -(y*math.log10(output)+(1-y)*math.log10(1-output))
    return loss

def cross_entropy_all_points():

    outputs=[
        0.999,
        0.1,
        0.9999
    ]

    y=[1,0,1]

    #summation -1(y*log(output)+(1-y)*log(1-output))/n where n is number of inputs, above n==3
    total_loss=0
    for index,output in enumerate(outputs):
        total_loss+=single_input_crossentropy_loss(output,y[index])
    
    return total_loss


print(cross_entropy_all_points())