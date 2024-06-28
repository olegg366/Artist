import os

src = open('trajectory.cpp', 'r')
dst = open('lib/trajectory.prev', 'r')
s, d = src.read(), dst.read()
try:
    if s != d:
        print('compiling...')
        dst.close()
        
        err1 = os.system('nvc++ -fPIC -stdpar -Iinclude-stdpar -gpu=managed,cuda11.8,cc61 -std=c++17 -c trajectory.cpp -o lib/trajectory.o')
        err2 = os.system('nvc++ -shared -gpu=managed,cuda11.8,cc61 -stdpar lib/trajectory.o -o lib/trajectory.so')
        
        if err1: exit(err1)
        if err2: exit(err2)
        
        dst = open('lib/trajectory.prev', 'w')
        dst.write(s)
        print('compiled')
finally:
    dst.close()
    src.close()


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from PIL import Image
from imageio.v2 import imread, imwrite
from time import sleep

from skimage.measure import label, regionprops
from skimage.feature import canny
from skimage.transform import resize, rotate
from skimage.morphology import binary_dilation, square

import ctypes as c

from serial_control import get_gcode, send_gcode

lib = c.CDLL('/home/olegg/Artist/lib/trajectory.so')

DPOINTER2D = np.ctypeslib.ndpointer(dtype=np.float128,
                                   ndim=2,
                                   flags='C')

DPOINTER3D = np.ctypeslib.ndpointer(dtype=np.float128,
                                   ndim=3,
                                   flags='C')

IPOINTER2D = np.ctypeslib.ndpointer(dtype=np.int32,
                                   ndim=2,
                                   flags='C')

lib.pmark.argtypes = [DPOINTER2D, DPOINTER2D, c.c_size_t, c.c_size_t, c.c_size_t, c.c_size_t]
lib.pmark.restype = None

lib.pfill.argtypes = [c.c_int32, c.c_int32, IPOINTER2D, DPOINTER3D, c.c_size_t, c.c_size_t, c.c_size_t]
lib.pfill.restype = None

lib.pcompute_image.argtypes = [IPOINTER2D, c.c_size_t, c.c_size_t, c.c_int32, c.c_int32, c.c_int32]
lib.pcompute_image.restype = c.POINTER(c.c_int32)

def mark(img, clrs):
    lib.pmark(img.astype('float128', order='C'), clrs.astype('float64', order='C'), *img.shape, *clrs.shape)
    return img

def fill(x, y, vis, img):
    lib.pfill(x, y, vis.astype('int32', order='C'), img.astype('float64', order='C'), *img.shape)
    return vis, img

def compute_image(img, d, sx, sy):
    res = lib.pcompute_image(img.astype('int32', order='C'), *img.shape, d, sx, sy)
    ans = []
    sz = res[0]
    for i in range(sz):
        ans.append([res[i * 2 + 1], res[i * 2 + 2]])

    lib.cleanup(res)
    
    return ans

def dispersion(x):
    return ((x - x.mean()) ** 2).sum() / x.size

def get_colors(img, crop):
    r, g, b = [img[:, :, i] for i in range(3)]
    dr, dg, db = map(dispersion, [r, g, b])
    mx = max(dr, dg, db)
    if mx == dr:
        t = canny(r)
    elif mx == dg:
        t = canny(g)
    else:
        t = canny(b)
        
    t = binary_dilation(t, square(5))
    if crop:
        nz = np.nonzero(t)
        t = t[max(0, np.min(nz[0]) - 10):min(t.shape[0], np.max(nz[0]) + 10), max(0, np.min(nz[1]) - 10):min(t.shape[1], np.max(nz[1]) + 10)]
    return t
        
def draw_img(img, crop=False, **kwargs):
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    img = rotate(img, 90, mode='edge')[:, ::-1]
    print('getting colors..')
    img = get_colors(img, crop)
    print('got colors')
    
    print('getting trajectory...')
    idx = 0
    trajectory = []
    lb = label(img)
    rgs = regionprops(lb)
    for reg in rgs:
        idx += 1
        regimg = reg.image
        cords = compute_image(regimg, 4, *reg.bbox[:2])
        trajectory.extend(cords)
        if trajectory[-1][0] != 1e9:
            trajectory.append([1e9, 1e9])
    print('got trajectory')
    # print(trajectory)
    # trajectory = np.array(trajectory)
    # ax = plt.subplot()
    # ax.imshow(img, cmap='gray')
    # i = 0
    # end = 0
    # while i < len(trajectory):
    #     while i < len(trajectory) and trajectory[i, 0] != -1e9:
    #         i += 1
    #     i += 1
    #     end = i
    #     while i < len(trajectory) and trajectory[i, 0] != 1e9:
    #         i += 1
    #     ax.add_line(Line2D(trajectory[end:i, 1], trajectory[end:i, 0], lw=1, color='blue'))
    # plt.get_current_fig_manager().full_screen_toggle()
    # plt.show()
    
    print('sending gcode...')
    gcode = get_gcode(trajectory, **kwargs)
    send_gcode(gcode)
    print('sent gcode')
    
if __name__ == '__main__':
    img = imread('images/what\'s_this_1719511707.8593214.png')
    # img = resize(img, (512, img.shape[1] * (512 / img.shape[0])))
    draw_img(img, k=512/370)