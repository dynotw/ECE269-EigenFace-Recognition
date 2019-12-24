from __future__ import print_function
import os
import sys
import cv2
import numpy as np
from sklearn.metrics import mean_squared_error as mse
import matplotlib.pyplot as plt
import pandas as pd

im=cv2.imread('1a.jpg')
im_1=cv2.imread('1a.jpg', flags=cv2.IMREAD_GRAYSCALE)
print('shape of im is', np.shape(im))
print(im_1)
print(im)
