import os
os.system('python compile_cython.py build_ext --inplace')

from accelerated_trajectory import fill, get_borders, get_trajectory
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
        f, img = fill(*rg.coords[0], f, img, var)

clrs = np.unique(img.reshape(-1, img.shape[2]), axis=0)
clr = clrs[2]
f = (img == clr).sum(axis=2) == 3
lb = label(f)
rgs = regionprops(lb)
ret = np.zeros((*img.shape[:-1], ), dtype='float32')
for reg in rgs:
    regimg = reg.image
    dx, dy, x1, y1 = reg.bbox
    cords, x, y = get_borders(regimg)
    cords, nimg = get_trajectory(x, y, 2, cords, regimg, var)
    # ret[dx:x1, dy:y1] = nimg
    plt.imshow(nimg, cmap='gray')
    plt.show()
imwrite('out/out.png', (ret * 255).astype('uint8'))