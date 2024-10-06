import os

src = open('trajectory.cpp', 'r')
dst = open('lib/trajectory.prev', 'r')
s, d = src.read(), dst.read()
try:
    if s != d:
        print('compiling...')
        dst.close()
        
        err1 = os.system('nvc++ -fPIC -stdpar -Iinclude-stdpar -gpu=managed,cuda11.8,cc86 -std=c++17 -c trajectory.cpp -o lib/trajectory.o')
        err2 = os.system('nvc++ -shared -gpu=managed,cuda11.8,cc86 -stdpar lib/trajectory.o -o lib/trajectory.so')
        
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
from time import sleep, time

from skimage.measure import label, regionprops
from skimage.feature import canny
from skimage.transform import resize, rotate
from skimage.morphology import binary_dilation, square

import ctypes as c

from serial_control import get_gcode, send_gcode

import cv2
from math import sin, cos, acos, sqrt, pi
from detect_paper import detect_paper

from serial import Serial

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

def shift_coords(sx: float, sy: float, angle: float, matrix: np.ndarray):
    rm = np.array([[cos(angle), sin(angle)],
                   [-sin(angle), cos(angle)]])
    rotated = np.matmul(matrix, rm)
    shifted = rotated + [sx, sy]
    return shifted

def get_angle(x1, y1, x2, y2):
    return acos((x1 * x2 + y1 * y2) / (sqrt(x1 * x1 + y1 * y1) * sqrt(x2 * x2 + y2 * y2))) 

def dist(x1, y1, x2, y2):
    return sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        
def draw_img(img, crop=False):
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    img = img[::-1]
    #img = rotate(img, 90, mode='edge')[::-1]
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
    trajectory = np.array(trajectory)
    points = []
    vid = cv2.VideoCapture(0)
    vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cnt = 0
    print("Get ready...")
    serial = Serial('/dev/ttyACM0', 115200)
    sleep(2)
    serial.write(b'G90\n')
    serial.write(b'G1 X450 F16000\n')
    serial.read_until(b'ok\n')
    sleep(2)
    try:
        while True:
            ret, frame = vid.read()
            if not ret:
                print('Cant')
                continue
            points, frame = detect_paper(frame, warp=True)
            if points is not None:
                cnt += 1
            if cnt == 50:
                break
            # cv2.imshow('img', frame)
            # cv2.waitKey(1)
    except KeyboardInterrupt:
        vid.release()
        serial.write(b'G1 X0 F16000\n')
        serial.close()
        raise(KeyboardInterrupt)
    vid.release()
    serial.write(b'G1 X0 F16000\n')
    serial.read_until(b'ok\n')
    sleep(1)
    serial.close()
    w, h = sorted([dist(*points[0], *points[1]), dist(*points[1], *points[2])])
    for i in range(1, len(points)):
        x = dist(*points[i], *points[(i + 1) % len(points)])
        y = dist(*points[i], *points[i - 1])
        if x == h and y == w:
            idx = i
            break
    points = points[[(idx + i) % len(points) for i in range(len(points))]] 
    sx, sy = points[0]
    mx, my = points[1]
    angle = get_angle(0, 1, mx - sx, my - sy)
    index = np.any(np.abs(trajectory) != [1e9, 1e9], axis=1)
    
    k = w / max(img.shape)
    trajectory[index] *= (k / 1.1)
    if np.max(shift_coords(sx, sy, pi / 2 - angle, trajectory[index]) > max(frame.shape)): angle = pi/2 + angle
    else: angle = pi / 2 - angle
    trajectory[index] = shift_coords(sx, sy, angle, trajectory[index])
    # print(trajectory.tolist())
    # ax = plt.subplot()
    # ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
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
    gcode = get_gcode(trajectory)
    send_gcode(gcode)
    print('sent gcode')
    
if __name__ == '__main__':
    img = imread('images/gen.png')
    img = rotate(resize(img, (512, img.shape[1] * (512 / img.shape[0]))), 0)
    draw_img(img, True)
