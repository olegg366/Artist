# distutils: language=c++

cimport numpy as np
import math
import numpy as pnp
from time import sleep
from random import choice

cdef list var = [[i, j] for i in range(-1, 2) for j in range(-1, 2) if not (i != 0 and j != 0) and not (i == 0 and j == 0)]
var += [[i, j] for i in range(-1, 2) for j in range(-1, 2) if (i != 0 and j != 0)]

cdef class point: 
    cdef public double x, y
    def __cinit__(self, double x, double y): 
        self.x = x 
        self.y = y  

    def __repr__(self):
        return f'point({self.x}, {self.y})'

    def __eq__(self, other):
        return dist(self, other) < 1e-6

    def __str__(self):
        return str(self.x) + ' ' + str(self.y)

    def __add__(self, point b):
        return point(self.x + b.x, self.y + b.y)

    def __mod__(self, point b):
        return self.x * b.y - self.y * b.x
    
    def __mul__(self, b):
        if isinstance(b, point):
            return self.x * b.x + self.y * b.y
        else:
            return point(self.x * b, self.y * b)
    
    cpdef double len(self):
        return (self.x ** 2 + self.y ** 2) ** 0.5

cdef point convert(x):
    if not isinstance(x, point):
        x = point(x[0], x[1])
    return x

cdef double dist(a, b):
    a, b = convert(a), convert(b)
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5

cdef point vec(a, b):
    a, b = convert(a), convert(b)
    return point(b.x - a.x, b.y - a.y)

cdef double get_angle(a, b):
    a, b = convert(a), convert(b)
    return math.atan2(a % b, a * b)


cdef bint in_image(int x, int y, list shape):
    if x >= 0 and y >= 0 and x < shape[0] and y < shape[1]:
        return True
    return False

def mark(np.ndarray img, np.ndarray itsa):
    nimg = pnp.zeros_like(img)
    for i in range(img.shape[0]):
        px = img[i]
        if (px == pnp.array([0, 0, 0], dtype='float64')).sum() != 3:
            nimg[i] = itsa[pnp.argmin(pnp.abs(px - itsa).sum(axis=1))]
    return nimg

cpdef list fill(int x, int y, np.ndarray vis, np.ndarray img):
    cdef np.ndarray vis1 = vis.copy()
    vis[x, y] = 1
    cdef list ln = [[x, y, 0, [0, 0, 0]]]
    cdef int xn, yn
    cdef list shp = [img.shape[0], img.shape[1]]
    while len(ln) != 0:
        for x, y, flg, clr in ln[:]:
            ln.pop(0)
            for dx, dy in var:
                xn = x + dx
                yn = y + dy
                if in_image(xn, yn, shp):
                    if flg: 
                        img[xn, yn] = clr
                        if not vis1[xn, yn]:
                            ln.append([xn, yn, flg, clr])
                            vis1[xn, yn] = 1
                    elif (img[xn, yn] == (0, 0, 0)).sum() != 3:
                        clr = img[xn, yn]
                        img[x, y] = clr
                        flg = 1
                        ln.append([x, y, flg, clr])
                        vis1[x, y] = 1
                    elif not vis[xn, yn]:
                        vis[xn, yn] = 1
                        ln.append([xn, yn, flg, clr])
    cdef list ans = [vis, img]
    return ans

cdef tuple get_borders(np.ndarray img, double sx, double sy):
    cdef int val
    img = img.astype('int')
    cdef np.ndarray ans = pnp.zeros((img.shape[0], img.shape[1], 2), dtype='bool')
    for x in range(img.shape[0]):
        if img[x, 0]:
            ans[x, 0, 0] = 1
        y = 0
        while y < img.shape[1]:
            val = img[x, y]
            while y < img.shape[1] and img[x, y] == val:
                y += 1
            if val:
                ans[x, min(y - 1, img.shape[1] - 1), 0] = 1
            elif y < img.shape[1]:
                ans[x, y, 0] = 1
    for y in range(img.shape[1]):
        if img[0, y]:
            ans[0, y, 1] = 1
        x = 0
        while x < img.shape[0]:
            val = img[x, y]
            while x < img.shape[0] and img[x, y] == val:
                x += 1
            if val:
                ans[min(x - 1, img.shape[0] - 1), y, 1] = 1
            elif x < img.shape[0]:
                ans[x, y, 1] = 1
    
    cdef list l 
    cdef list ansp = [], holes = []
    cdef int xn, yn
    ans = ans[:, :, 0] | ans[:, :, 1]
    cdef np.ndarray vis = pnp.zeros_like(ans)
    cdef int it = 0
    while (vis == ans).sum().sum() != ans.shape[0] * ans.shape[1]:
        l = [nonzero(ans & (~vis))[0]]
        comp = []
        comp.append(point(l[0][0] + sx, l[0][1] + sy))
        while len(l) != 0:
            x, y = l[0]
            vis[x, y] = 1
            l = l[1:]
            for dx, dy in var:
                xn, yn = x + dx, y + dy
                if in_image(xn, yn, [img.shape[0], img.shape[1]]) and ans[xn, yn] and not vis[xn, yn]:
                    comp.append(point(xn + sx, yn + sy))
                    l.append([xn, yn])
                    vis[xn, yn] = 1
                    break
        if it >= 1:
            holes.append(comp)
        else:
            ansp.extend(comp)
    return ansp, holes

cdef bint check_ray(point a, point b, point p):
    return (vec(a, b) % vec(a, p) == 0) and (vec(a, b) * vec(a, p) >= 0)

cdef bint check_seg(point a, point b, point p):
    a, b, p = convert(a), convert(b), convert(p)
    return check_ray(a, b, p) and check_ray(b, a, p)

cdef bint in_polygon(a, list p, list h=[]):
    a = convert(a)
    cdef double anglep = 0
    cdef int j
    cdef point v1, v2
    for i in range(len(p)):
        j = (i + 1) % len(p)
        if check_seg(p[i], p[j], a):
            return False
        v1 = vec(a, p[i])
        v2 = vec(a, p[j])
        anglep += get_angle(v1, v2)
    anglep = abs(anglep)

    cdef double angleh = 0
    for i in range(len(h)):
        j = (i + 1) % len(h)
        if check_seg(h[i], h[j], a):
            return anglep >= math.pi
        v1 = vec(a, h[i])
        v2 = vec(a, h[j])
        angleh += get_angle(v1, v2)
    angleh = abs(angleh)
    return anglep >= math.pi and angleh < math.pi

cdef list nonzero(np.ndarray x):
    return list(zip(*pnp.nonzero(x)))

cdef bint check_near(int x, int y, list deltas, np.ndarray bimage, list borders=None, list holes=None):
    cdef int xn, yn, dtx, dty
    for dtx, dty in deltas:
        xn, yn = x + dtx, y + dty
        if in_image(xn, yn, [bimage.shape[0], bimage.shape[1]]):
            if borders is None:
                if bimage[xn, yn]:
                    return False
            else:
                if not in_polygon((xn, yn), borders, holes):
                    return False
    return True

cdef tuple random_point(np.ndarray bimage, list deltas, list borders=None, list holes=None):
    cdef int x, y 
    x, y = -1, -1
    cdef int cnt = len(nonzero(bimage)) - 1
    cdef list nz = nonzero(bimage)
    while cnt >= 0 and not check_near(x, y, deltas, bimage, borders, holes):
        x, y = nz[cnt]
        cnt -= 1
    return x, y

cdef list get_path(a, b):
    cdef int x, y, tx, ty, xn, yn
    x, y = a
    tx, ty = b
    cdef list l = [[x, y]]
    cdef list ans = []
    cdef double prevd, d
    while True:
        for dx, dy in var:
            xn, yn = x + dx, y + dy
            if xn == tx and yn == ty:
                return ans + [[tx, ty]]
            d = dist((xn, yn), (tx, ty))
            if d < prevd:
                x, y = xn, yn
                prevd = d
                ans.append([x, y])
                break

cpdef list compute_image(np.ndarray bimage, int d, double sx, double sy):
    cdef list deltas2 = [[i, j] for i in range(-d, d + 1) for j in range(-d, d + 1) if abs(dist((i, j), (0, 0)) - d) <= 0.5]
    cdef list mxdeltas = [[i, j] for i in range(-d // 2, d // 2 + 1) for j in range(-d // 2, d // 2 + 1) if abs(dist((i, j), (0, 0)) - d // 2) <= 0.5]
    cdef list deltas = [[i, j] for i in range(-d // 2, d // 2 + 1) for j in range(-d // 2, d // 2 + 1) if dist((i, j), (0, 0)) <= d // 2 + 0.5]
    
    cdef list borders, holes
    borders, holes = get_borders(bimage, sx, sy)

    cdef list ans = []

    cdef int x, y, xn, yn, dtx, dty, xp, yp, it
    cdef bint change
    x, y = random_point(bimage, mxdeltas)
    xp, yp = 0, 0
    it = 0

    while it != 100:
        print('deltas')
        change = False
        for dtx, dty in deltas2:
            xn, yn = x + dtx, y + dty
            if check_near(xn, yn, mxdeltas, bimage):
                x, y = xn, yn
                ans.append([x, y])
                change = True
                break
        print('end')
        if not change:
            x, y = random_point(bimage, mxdeltas)
            # ans.append([-1e6, -1e6])
            ans.append([x, y])
            # ans.append([1e6, 1e6])
            if x == -1:
                x, y = random_point(bimage, mxdeltas, borders, holes)
        
        if it != 0:
            for xn, yn in get_path((xp, yp), (x, y)):
                for dtx, dty in deltas:
                    xnn, ynn = xn + dtx, yn + dty
                    if in_image(xnn, ynn, [bimage.shape[0], bimage.shape[1]]):
                        bimage[xnn, ynn] = 0
        else:
            for dtx, dty in deltas:
                xn, yn = x + dtx, y + dty
                if in_image(xn, yn, [bimage.shape[0], bimage.shape[1]]):
                    bimage[xn, yn] = 0
        xp, yp = x, y
        it += 1
    return ans
