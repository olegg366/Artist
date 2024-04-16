import os
print(os.system('python compile_cython.py build_ext --inplace'))

from accelerated_trajectory import *
import numpy as np
from skimage.morphology import label
from skimage.measure import regionprops
import matplotlib.pyplot as plt
from imageio.v3 import imread, imwrite
from time import sleep

img = imread('colors.png')
var = [[i, j] for i in range(-1, 2) for j in range(-1, 2) if not (i == 0 and j == 0)]
f = (img == 0).sum(axis=2) == 3
f = ~f
imwrite('test/in.png', f.astype('uint8') * 255)
lb = label(~f)
rgs = regionprops(lb)
for rg in rgs:
    if rg.area < 30:
        f, img = bfs(*rg.coords[0], f, img, var)
imwrite('test/out.png', f.astype('uint8') * 255)
imwrite('test/imout.png', img)