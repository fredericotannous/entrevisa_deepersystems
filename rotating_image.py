# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 23:38:45 2020

@author: Frederico
"""


from keras.models import load_model
import pandas as pd
import cv2
import numpy as np
import glob

def rotate(model,image):
    image_pred = image.reshape(1,64,64,3)
    output = model.predict(image_pred)
    value = output.argmax()
    angle = 0
    
    if value  == 1:     
        angle = 90

    elif value == 0:    
        angle = 270

    elif value == 3:
        angle = 180