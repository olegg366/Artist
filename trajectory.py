import os
os.system('python compile_cython.py build_ext --inplace')

from accelerated_trajectory import bfs, get_borders
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
lb = label(~f)
rgs = regionprops(lb)
for rg in rgs:
    if rg.area < 30:
        f, img = bfs(*rg.coords[0], f, img, var)

clrs = np.unique(img.reshape(-1, img.shape[2]), axis=0)
clr = clrs[2]
f = (img == clr).sum(axis=2) == 3
lb = label(f)
rgs = regionprops(lb)
ret = np.zeros_like(img)
for reg in rgs:
    regimg = reg.image
    cords = get_borders(regimg, np.zeros((*regimg.shape, 2), dtype='bool'))
    dx, dy = reg.bbox[:2]
    cords = np.transpose(np.nonzero(cords))
    cords[:, 0] += dx
    cords[:, 1] += dy
    for cord in cords:
        ret[cord[0], cord[1]] = 1
imwrite('out/out.png', ret.astype('uint8') * 255)