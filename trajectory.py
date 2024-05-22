import os
os.system('CC=nvc python compile_cython.py build_ext --inplace')

from accelerated_trajectory import fill, compute_image
import numpy as np
from random import randint
from skimage.morphology import label
from skimage.measure import regionprops
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Arrow
from imageio.v3 import imread, imwrite
from time import sleep  

SERVO_DOWN = ''
SERVO_UP = ''

img = imread('colors.png')
f = (img == 0).sum(axis=2) == 3
f = ~f
lb = label(~f)
rgs = regionprops(lb)
for rg in rgs:
    if rg.area < 30:
        f, img = fill(*rg.coords[0], f, img)

clrs = np.unique(img.reshape(-1, img.shape[-1]), axis=0)
idx = 0
all = []
for i in range(1, len(clrs)):
    clr = clrs[i]
    f = (img == clr).sum(axis=2) == 3
    lb = label(f)
    rgs = regionprops(lb)
    for reg in rgs:
        idx += 1
        print('image number', idx)
        regimg = reg.image
        cords = compute_image(regimg, 10, *reg.bbox[:2])
        all.append('down')
        all.extend(cords)
        all.append('up')
        ax = plt.subplot()
        ax.imshow(img)
        ax.add_line(Line2D(cords[:, 1], cords[:, 0], lw=1, color='black'))
        plt.show()

