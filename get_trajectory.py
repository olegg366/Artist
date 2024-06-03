import os
os.system('./set_cython.sh')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from cython_files.accelerated_trajectory import mark, fill, compute_image, remove_intersections

from PIL import Image
from imageio.v2 import imread, imwrite
from time import sleep

from skimage.measure import label, regionprops
from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu
from skimage.morphology import binary_dilation, square, remove_small_objects

import cv2
import pickle

from serial_control import get_gcode, send_gcode

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
    for color in clrs:
        if (color != [0, 0, 0]).any():
            f = (img == color).sum(axis=2) == 3
            if len(np.unique(f)) == 1: continue
            nf = f != remove_small_objects(f, 10)
            
            img[nf] = 0
            f = (img == color).sum(axis=2) == 3
            if len(np.unique(f)) == 1: continue
            f = binary_dilation(f, square(5))
            
            img[f] = color
            
    f = (img == 0).sum(axis=2) == 3
    f = ~f
    lb = label(~f)
    rgs = regionprops(lb)
    for rg in rgs:
        if rg.area < 30:
            f, img = fill(*rg.coords[0], f, img)
    return img
        
def draw_img(img: Image):
    img = np.array(img)
    # print('getting colors..')
    # img = get_colors(img)
    # print('got colors')
    
    print('getting trajectory...')
    clrs = np.unique(img.reshape(-1, img.shape[-1]), axis=0)
    idx = 0
    all = []
    trajectory = []
    for i in range(1, len(clrs)):
        clr = clrs[i]
        f = (img == clr).sum(axis=2) == 3
        lb = label(f)
        rgs = regionprops(lb)
        for reg in rgs:
            idx += 1
            regimg = reg.image
            cords = compute_image(regimg, 10, *reg.bbox[:2])
            all.append('down')
            all.extend(cords)
            all.append('up')
            trajectory.extend(cords)
    print('got trajectory')
    # print(trajectory)
    trajectory = np.array(trajectory)
    ax = plt.subplot()
    ax.imshow(img)
    ax.add_line(Line2D(trajectory[:, 1], trajectory[:, 0], lw=1, color='white'))
    ax.add_line(Line2D([trajectory[0, 1], trajectory[-1, 1]], [trajectory[0, 0], trajectory[-1, 0]], lw=1, color='white'))
    plt.show()
    
    with open('last_trajectory.lst', 'wb') as f:
        pickle.dump(all, f)
    
    # print('sending gcode...')
    # gcode = get_gcode(all)
    # send_gcode(gcode)
    # print('sent gcode')
    # print(remove_intersections([[-5, 1], [-3, 4], [3, 5], [-1, 6], [2, 3], [1, 0]]))
    
if __name__ == '__main__':
    img = imread('images/colors_square.png')
    draw_img(img)
    sleep(10000)