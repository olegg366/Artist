import os
os.system('python compile_cython.py build_ext --inplace')

from accelerated_trajectory import fill, compute_image
import numpy as np
from random import randint
from skimage.morphology import label
from skimage.measure import regionprops
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from imageio.v3 import imread, imwrite
from time import sleep

def get_max_divs(x):
    i = 1
    mxdiv = 1
    while i * i <= x:
        if x % i == 0:
            mxdiv = i
        i += 1
    return mxdiv, x // mxdiv

img = imread('colors_circle.png')
f = (img == 0).sum(axis=2) == 3
f = ~f
lb = label(~f)
rgs = regionprops(lb)
for rg in rgs:
    if rg.area < 30:
        f, img = fill(*rg.coords[0], f, img)

clrs = np.unique(img.reshape(-1, img.shape[-1]), axis=0)
idx = 0
for i in range(1, len(clrs)):
    clr = clrs[i]
    f = (img == clr).sum(axis=2) == 3
    lb = label(f)
    rgs = regionprops(lb)
    for reg in rgs:
        idx += 1
        print('image number', idx)
        regimg = reg.image
        plt.imshow(regimg)
        plt.show()
        cords = compute_image(regimg, 10, *reg.bbox[:2])
        ax = plt.subplot()
        ax.imshow(img)
        ax.add_line(Line2D(cords[:, 1], cords[:, 0], color='black', lw=1))
        plt.show()
