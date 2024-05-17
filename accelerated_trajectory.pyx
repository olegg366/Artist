cimport numpy as np
from time import sleep
import math
import numpy as pnp

# define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

cdef bint check(int x, int y, list shape):
    if x >= 0 and y >= 0 and x < shape[0] and y < shape[1]:
        return True
    return False

cpdef list fill(int x, int y, np.ndarray vis, np.ndarray img, list var):
    cdef np.ndarray vis1 = vis.copy()
    vis[x, y] = 1
    cdef list ln = [[x, y, 0, [0, 0, 0]]]
    cdef int xn, yn
    cdef list shp = [512, 512]
    while len(ln) != 0:
        for x, y, flg, clr in ln[:]:
            ln.pop(0)
            for dx, dy in var:
                xn = x + dx
                yn = y + dy
                if check(xn, yn, shp):
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

cpdef np.ndarray get_borders(np.ndarray img):
    cdef int val
    img = img.astype('int')
    cdef np.ndarray ans = pnp.zeros((img.shape[0], img.shape[1], 2), dtype='bool')
    for x in range(img.shape[0]):
        if img[x, 0]:
            ans[x, 0][0] = 1
        val = img[x, 0]
        y = 0
        while y < img.shape[1]:
            val = img[x, y]
            while y < img.shape[1] and img[x, y] == val:
                y += 1
            if val:
                ans[x, min(y, img.shape[1] - 1)][0] = 1
            elif y < img.shape[1]:
                ans[x, y][0] = 1
    for y in range(img.shape[1]):
        if img[0, y]:
            ans[0, y, 1] = 1
        val = img[0, y]
        x = 0
        while x < img.shape[0]:
            val = img[x, y]
            while x < img.shape[0] and img[x, y] == val:
                x += 1
            if val:
                ans[min(x, img.shape[0] - 1), y, 1] = 1
            elif x < img.shape[0]:
                ans[x, y, 1] = 1
    ans = ans[:, :, 0] | ans[:, :, 1]
    return ans

cdef list compute_shift(int x1, int y1, int x2, int y2, double d, np.ndarray fil):
    cdef double ln = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
    cdef double xd = (y2 - y1) / ln * d + x1
    cdef double yd = (x1 - x2) / ln * d + y1
    if not check(int(xd), int(yd), [fil.shape[0], fil.shape[1]]) or not fil[int(xd), int(yd)]:
        xd = (y1 - y2) / ln * d + x1
        yd = (x2 - x1) / ln * d + y1
    return [xd, yd]

cdef class point: 
    cdef public double x, y
    def __init__(self, double x, double y): 
        self.x = x 
        self.y = y  

    def __eq__(self, other):
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2) < 1e-6

    def __str__(self):
        return str(self.x) + ' ' + str(self.y)

cpdef list get_trajectory(double d, np.ndarray img, np.ndarray fil, list var):
    cdef list l = []
    cdef list ans = []
    cdef int xn, yn, prevln = 0, x, y
    cdef list shp = [img.shape[0], img.shape[1]]
    cdef double xf, yf
    cdef np.ndarray nimg = img.astype('float32'), nxt = pnp.zeros_like(img), cur
    cdef np.ndarray vis = pnp.zeros_like(img)
    while (vis == img).sum().sum() != shp[0] * shp[1]:
        # print('difference:', round((shp[0] * shp[1] - (vis == img).sum().sum()) / (shp[0] * shp[1]) * 100, 2))
        cur = img.copy()
        ans.append('down')
        while True:
            print(ans)
            sleep(1)
            prevln = len(ans)
            l = list(zip(*pnp.nonzero((vis != cur) & cur)))
            if not l:
                break
            l = l[:1]
            while len(l) != 0:
                x, y = l[0]
                l.pop(0)
                for dx, dy in var:
                    xn, yn = x + dx, y + dy
                    if check(xn, yn, shp) and cur[xn, yn] and not vis[xn, yn]:
                        vis[xn, yn] = 1
                        l.append([xn, yn])
                        xf, yf = compute_shift(xn, yn, x, y, d, fil)
                        if check(int(xf), int(yf), shp) and fil[int(xf), int(yf)]:
                            ans.append(point(xf, yf))
                            x, y = map(int, [xf, yf])
                            nimg[x, y] = 0.7
                            nxt[x, y] = 1
            cur = nxt.copy()
            nxt = pnp.zeros_like(img)
        ans.append('up')
    ans = remove_intersections(ans)
    return [ans, nimg]

cdef list remove_intersections(list line):
    cdef point intcord
    cdef bint flag
    cdef int i = 0, j
    # print('length:', len(line))
    while i < len(line) - 1:
        if not isinstance(line[i], str) and not isinstance(line[i + 1], str):
            j = i + 1
            while j < len(line) - 1:
                if not isinstance(line[j], str) and not isinstance(line[j + 1], str):
                    intcord, flag = intersect(line[i], line[i + 1], line[j], line[j + 1])
                    if flag:
                        # print('intersection found on', intcord, '\nsegments:', line[i], '--', line[i + 1], '|', line[j], '--', line[j + 1])
                        # print('indexes:', i, j)
                        if len(line) - (j - i) > j - i and j - i < 10:
                            # print('mode: 0')
                            for d in range(i + 1, j + 1):
                                line.pop(i + 1)
                            line.insert(i + 1, intcord)
                            j = i + 1
                        elif len(line) - (j - i) < 10:
                            # print('mode: 1')
                            for d in range(i + 1):
                                line.pop(0)
                            for d in range(j + 1, len(line)):
                                line.pop(j - i)
                            i = 0
                            j = i
                j += 1
        i += 1
    return line

cdef bint on_segment(point p, point q, point r): 
    if (q.x <= max(p.x, r.x)) and (q.x >= min(p.x, r.x)) and (q.y <= max(p.y, r.y)) and (q.y >= min(p.y, r.y)): 
        return True
    return False

cdef int orientation(point p, point q, point r): 
    cdef double val = ((q.y - p.y) * (r.x - q.x)) - ((q.x - p.x) * (r.y - q.y)) 
    if (val > 0): 
        return 1
    elif (val < 0): 
        return 2    
    else: 
        return 0

cdef list intersect(point p1, point q1, point p2, point q2): 
    cdef int o1 = orientation(p1, q1, p2) 
    cdef int o2 = orientation(p1, q1, q2) 
    cdef int o3 = orientation(p2, q2, p1) 
    cdef int o4 = orientation(p2, q2, q1) 
    cdef point cord = point(0, 0)

    if ((o1 != o2) and (o3 != o4)): 
        cord = get_intersection(p1, q1, p2, q2)
        if cord == p1 or cord == p2 or cord == q1 or cord == q2:
            return [cord, False]
        return [cord, True]

    if ((o1 == 0) and on_segment(p1, p2, q1)): 
        cord = get_intersection(p1, q1, p2, q2)
        if cord == p1 or cord == p2 or cord == q1 or cord == q2:
            return [cord, False]
        return [cord, True]

    if ((o2 == 0) and on_segment(p1, q2, q1)): 
        cord = get_intersection(p1, q1, p2, q2)
        if cord == p1 or cord == p2 or cord == q1 or cord == q2:
            return [cord, False]
        return [cord, True]

    if ((o3 == 0) and on_segment(p2, p1, q2)): 
        cord = get_intersection(p1, q1, p2, q2)
        if cord == p1 or cord == p2 or cord == q1 or cord == q2:
            return [cord, False]
        return [cord, True]

    if ((o4 == 0) and on_segment(p2, q1, q2)): 
        cord = get_intersection(p1, q1, p2, q2)
        if cord == p1 or cord == p2 or cord == q1 or cord == q2:
            return [cord, False]
        return [cord, True]
    return [cord, False]

cdef double det(point a, point b):
    return a.x * b.y - a.y * b.x

cdef point get_intersection(point p1, point q1, point p2, point q2):
    cdef point xdiff = point(p1.x - q1.x, p2.x - q2.x)
    cdef point ydiff = point(p1.y - q1.y, p2.y - q2.y)
    cdef double divv = det(xdiff, ydiff)
    if divv == 0:
        return p1
    cdef point d = point(det(p1, q1), det(p2, q2))
    cdef double x = det(d, xdiff) / divv
    cdef double y = det(d, ydiff) / divv
    return point(x, y)
