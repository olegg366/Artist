cimport numpy as np
import math
import numpy as pnp

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
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2) < 1e-6

    def __str__(self):
        return str(self.x) + ' ' + str(self.y)

    def __add__(self, point b):
        return point(self.x + b.x, self.y + b.y)

    def __mod__(self, point b):
        return self.x * b.y - self.y * b.x
    
    def __mul__(self, point b):
        return self.x * b.x + self.y * b.y

    cpdef point seg(self, point b):
        return point(b.x - self.x, b.y - self.y)
    
    cpdef double len(self):
        return (self.x ** 2 + self.y ** 2) ** 0.5

    cpdef double dist(self, point b):
        return ((self.x - b.x) ** 2 + (self.y - b.y) ** 2) ** 0.5


cdef bint check(int x, int y, list shape):
    if x >= 0 and y >= 0 and x < shape[0] and y < shape[1]:
        return True
    return False

cpdef list fill(int x, int y, np.ndarray vis, np.ndarray img):
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

cdef list get_borders(np.ndarray img, double sx, double sy):
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
    cdef list ansp = []
    cdef int xn, yn
    ans = ans[:, :, 0] | ans[:, :, 1]
    cdef np.ndarray vis = pnp.zeros_like(ans)
    while (vis == ans).sum().sum() != ans.shape[0] * ans.shape[1]:
        l = [list(zip(*pnp.nonzero(ans & (~vis))))[0]]
        ansp.append(point(l[0][0] + sx, l[0][1] + sy))
        while len(l) != 0:
            x, y = l[0]
            vis[x, y] = 1
            l = l[1:]
            for dx, dy in var:
                xn, yn = x + dx, y + dy
                if check(xn, yn, [img.shape[0], img.shape[1]]) and ans[xn, yn] and not vis[xn, yn]:
                    ansp.append(point(xn + sx, yn + sy))
                    l.append([xn, yn])
                    vis[xn, yn] = 1
                    break
    return ansp

cdef tuple compute_shift(point a, point b, double d, list polygon):
    cdef double ln = a.dist(b)
    if a == b:
        return point(0, 0)
    cdef double xd1 = (b.y - a.y) / ln * d + a.x
    cdef double yd1 = (a.x - b.x) / ln * d + a.y
    cdef double xd2 = (a.y - b.y) / ln * d + a.x
    cdef double yd2 = (b.x - a.x) / ln * d + a.y
    return point(xd1, yd1), point(xd2, yd2)

cdef double cos(point a, point b):
    return a * b / (a.len() * b.len())

cdef double get_angle(point a, point b):
    return math.atan2(a % b, a * b)

cdef bint check_ray(point a, point b, point p):
    return (a.seg(b) % a.seg(p) == 0) and (a.seg(b) * a.seg(p) >= 0)

cdef bint check_seg(point a, point b, point p):
    return check_ray(a, b, p) and check_ray(b, a, p)

cdef bint in_polygon(a, list p):
    cdef double angle = 0
    cdef int j
    cdef point v1, v2
    for i in range(len(p)):
        j = (i + 1) % len(p)
        if check_seg(p[i], p[j], a):
            return False
        v1 = a.seg(p[i])
        v2 = a.seg(p[j])
        angle += get_angle(v1, v2)
    angle = abs(angle)
    return angle >= math.pi 

cdef list component_fill(double d, list cords):
    cdef point shift1, shift2
    cdef list ansl = [], ansr = []
    cdef bint f1, f2
    for i in range(len(cords)):
        shift1, shift2 = compute_shift(cords[i], cords[(i + 1) % len(cords)], d, cords)
        if not ansr:
            ansl.append(shift1)
            ansr.append(shift2)
        else:
            if shift1.dist(ansl[-1]) < shift1.dist(ansr[-1]):
                ansl.append(shift1)
                ansr.append(shift2)
            elif shift1.dist(ansl[-1]) > shift1.dist(ansr[-1]):
                ansl.append(shift2)
                ansr.append(shift1)
            else:
                if shift2.dist(ansl[-1]) <= shift2.dist(ansr[-1]):
                    ansl.append(shift2)
                    ansr.append(shift1)
                else:
                    ansl.append(shift1)
                    ansr.append(shift2)
    f1 = in_polygon(ansl[0], cords)
    f2 = in_polygon(ansr[0], cords)
    if f1 or f2:
        if f1:
            cords.extend(ansl)
            # cords.extend(component_fill(d, ansl))
        else:
            cords.extend(ansr)
            # cords.extend(component_fill(d, ansr))
    return cords

cpdef np.ndarray compute_image(np.ndarray img, int d, double sx, double sy):
    print('getting borders...')
    cdef list cords = get_borders(img, sx, sy)
    cdef list ans = []
    print('filling...')
    cords.extend(component_fill(d, cords))
    cords = [[a.x, a.y] for a in cords]
    return pnp.array(cords)

cdef list remove_intersections(list line):
    cdef point intcord
    cdef bint flag
    cdef int i = 0, j
    while i < len(line) - 1:
        j = i + 1
        while j < len(line) - 1:
            intcord, flag = intersect(line[i], line[i + 1], line[j], line[j + 1])
            if flag:
                if len(line) - (j - i) > j - i and j - i < 10:
                    line = line[:i + 1] + [intcord] + line[j + 1:]
                    j = i + 1
                elif len(line) - (j - i) < 10:
                    line = line[i + 1:j + 1]
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

cdef point get_intersection(point p1, point q1, point p2, point q2):
    cdef point xdiff = point(p1.x - q1.x, p2.x - q2.x)
    cdef point ydiff = point(p1.y - q1.y, p2.y - q2.y)
    cdef double divv = xdiff * ydiff
    if divv == 0:
        return p1
    cdef point d = point(p1 * q1, p2 * q2)
    cdef double x = (d * xdiff) / divv
    cdef double y = (d * ydiff) / divv
    return point(x, y)
