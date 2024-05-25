
import numpy as np
from numpy.ctypeslib import ndpointer 

import matplotlib.pyplot as plt
from accelerated_trajectory import mark

from tqdm import trange, tqdm
from imageio import imread, imsave
from skimage.measure import label, regionprops
from skimage.color import rgb2hsv, hsv2rgb
from skimage.filters import threshold_otsu
from skimage.transform import resize
from skimage.morphology import binary_dilation, square, remove_small_objects
import cv2

def dispersion(x):
    return ((x - x.mean()) ** 2).sum() / x.size

def get_colors(img):
    r, g, b = [img[:, :, i] for i in range(3)]
    dr, dg, db = map(dispersion, [r, g, b])
    mx = max(dr, dg, db)
    if mx == dr:
        t = r < threshold_otsu(r)
    elif mx == dg:
        t = g < threshold_otsu(g)
    else:
        t = b < threshold_otsu(b)

    lb = label(t)
    big1 = lb == 1
    big2 = lb == 0

    s1 = big1[0].sum() + big1[-1].sum() + big1[:, 0].sum() + big1[:, -1].sum()
    s2 = big2[0].sum() + big2[-1].sum() + big2[:, 0].sum() + big2[:, -1].sum()

    if s1 > s2:
        msk = lb == 1
    else:
        msk = lb == 0

    newmsk = np.zeros((img.shape[0], img.shape[1], 3))
    newmsk[~msk, :] = 1
    newmsk = newmsk.astype('bool')

    img = rgb2hsv(img)

    clrs = np.array([[0., 0., 0.], [0., 0., 1.]])

    clrmsk = mark(img[~msk], clrs)
    nimg = np.zeros_like(img)
    nimg[~msk] = clrmsk

    img = nimg.copy()
    cv2.imshow('img', img)
    for color in tqdm(clrs):
        if (color != [0, 0, 0]).any():
            f = (img == color).sum(axis=2) == 3
            if len(np.unique(f)) == 1: continue
            nf = f != remove_small_objects(f, 10)
            
            img[nf] = 0
            f = (img == color).sum(axis=2) == 3
            if len(np.unique(f)) == 1: continue
            # f = remove_small_holes(f, 500)
            f = binary_dilation(f, square(5))
            
            img[f] = color
    
    return img
        
# img = imread('triangle.png')

# img = img[:, :, :3]

# img = resize(img, (512, 512))
# imsave('colors_triangle.png', (hsv2rgb(img) * 255).astype('uint8'))