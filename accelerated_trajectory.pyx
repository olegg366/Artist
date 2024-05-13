cimport numpy as np
import math
import numpy as pnp

# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

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

cpdef list get_borders(np.ndarray img):
    cdef int val
    img = img.astype('int')
    cdef int xi, yi
    cdef np.ndarray ans = pnp.zeros((img.shape[0], img.shape[1], 2), dtype='bool')
    for x in range(img.shape[0]):
        if img[x, 0]:
            ans[x, 0][0] = 1
            xi, yi = x, 0
        val = img[x, 0]
        y = 0
        while y < img.shape[1]:
            val = img[x, y]
            while y < img.shape[1] and img[x, y] == val:
                y += 1
            if val:
                ans[x, min(y, img.shape[1] - 1)][0] = 1
                xi, yi = x, min(y, img.shape[1] - 1)
            elif y < img.shape[1]:
                ans[x, y][0] = 1
    for y in range(img.shape[1]):
        if img[0, y]:
            ans[0, y, 1] = 1
            xi, yi = 0, y
        val = img[0, y]
        x = 0
        while x < img.shape[0]:
            val = img[x, y]
            while x < img.shape[0] and img[x, y] == val:
                x += 1
            if val:
                ans[min(x, img.shape[0] - 1), y, 1] = 1
                xi, yi = min(x, img.shape[0] - 1), y
            elif x < img.shape[0]:
                ans[x, y, 1] = 1
    ans = ans[:, :, 0] | ans[:, :, 1]
    return [ans, xi, yi]

cdef list compute_shift(int x1, int y1, int x2, int y2, double d, np.ndarray fil):
    cdef double ln = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
    cdef double xd = (y2 - y1) / ln * d + x1
    cdef double yd = (x1 - x2) / ln * d + y1
    if not check(int(xd), int(yd), [fil.shape[0], fil.shape[1]]) or not fil[int(xd), int(yd)]:
        xd = (y1 - y2) / ln * d + x1
        yd = (x2 - x1) / ln * d + y1
    return [xd, yd]

cdef class point: 
    cdef double x, y
    def __init__(self, double x, double y): 
        self.x = x 
        self.y = y  

    def __eq__(self, point o):
        return math.sqrt((self.x - o.x) ** 2 + (self.y - o.y) ** 2) < 1e-6

    def __str__(self):
        return str(self.x) + ' ' + str(self.y)

cpdef list get_trajectory(int x, int y, double d, np.ndarray img, np.ndarray fil, list var):
    cdef list l = [[x, y]]
    cdef list ans = []
    cdef int xn, yn
    cdef list shp = [img.shape[0], img.shape[1]]
    cdef double xf, yf
    cdef np.ndarray nimg = img.astype('float32')
    cdef np.ndarray vis = pnp.zeros_like(img)
    while len(l) != 0:
        x, y = l[0]
        l.pop(0)
        for dx, dy in var:
            xn, yn = x + dx, y + dy
            if check(xn, yn, shp) and img[xn, yn] and not vis[xn, yn]:
                vis[xn, yn] = 1
                l.append([xn, yn])
                xf, yf = compute_shift(xn, yn, x, y, d, fil)
                if check(int(xf), int(yf), [fil.shape[0], fil.shape[1]]) and fil[int(xf), int(yf)]:
                    ans.append(point(xf, yf))
                    x, y = map(int, [xf, yf])
                    nimg[x, y] = 0.7
                    break
    ans = remove_intersections(ans)
    return [ans, nimg]

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

cdef list remove_intersections(list line):
    cdef point intcord
    cdef bint flag
    cdef int i = 0, j
    while i < len(line) - 1:
        j = i + 1
        while j < len(line) - 1:
            intcord, flag = intersect(line[i], line[i + 1], line[j], line[j + 1])
            if flag:
                print('intersection found on', intcord, '\nsegments:', line[i], '--', line[i + 1], '|', line[j], '--', line[j + 1])
            j += 1
        i += 1
    return line
