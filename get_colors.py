
import numpy as np
from numpy.ctypeslib import ndpointer 

import matplotlib.pyplot as plt
import numba as nb

from tqdm import trange, tqdm
from imageio import imread, imsave
from skimage.measure import label, regionprops
from skimage.color import rgb2hsv, hsv2rgb
from skimage.filters import threshold_otsu
from skimage.transform import resize
from skimage.morphology import binary_dilation, square, remove_small_objects

img = imread('triangle.png')

img = img[:, :, :3]

img = resize(img, (512, 512))

def dispersion(x):
    return ((x - x.mean()) ** 2).sum() / x.size

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

newmsk = np.zeros((512, 512, 3))
newmsk[~msk, :] = 1
newmsk = newmsk.astype('bool')

img = rgb2hsv(img)

clrs = np.array([[h/50, 0.7, 0.7] for h in range(50)])
clrs = np.vstack((clrs, [[0., 0., 0.], [0., 0., 1.]]))

@nb.njit
def mark(img, itsa):
    nimg = np.zeros_like(img)
    for i in nb.prange(img.shape[0]):
        px = img[i]
        if (px == np.array([0, 0, 0], dtype='float64')).sum() != 3:
            nimg[i] = itsa[np.argmin(np.abs(px - itsa).sum(axis=1))]
    return nimg

clrmsk = mark(img[~msk], clrs)
nimg = np.zeros_like(img)
nimg[~msk] = clrmsk

nnimg = nimg.copy()
for color in tqdm(clrs):
    if (color != [0, 0, 0]).any():
        f = (nnimg == color).sum(axis=2) == 3
        if len(np.unique(f)) == 1: continue
        nf = f != remove_small_objects(f, 10)
        
        nnimg[nf] = 0
        f = (nnimg == color).sum(axis=2) == 3
        if len(np.unique(f)) == 1: continue
        # f = remove_small_holes(f, 500)
        f = binary_dilation(f, square(5))
        
        nnimg[f] = color
imsave('colors_triangle.png', (hsv2rgb(nnimg) * 255).astype('uint8'))