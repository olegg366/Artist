# distutils: language=c++

cimport numpy as np
import math
import numpy as pnp
from time import sleep

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
    
cdef double cos(point a, point b):
    return a * b / (a.len() * b.len())

cdef double get_angle(point a, point b):
    return math.atan2(a % b, a * b)

cdef point vec(point a, point b):
    return point(b.x - a.x, b.y - a.y)

cdef double dist(point a, point b):
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5

cdef point rot(point p, double angle):
    cdef double x = p.x * math.cos(angle) - p.y * math.sin(angle)
    cdef double y = p.x * math.sin(angle) + p.y * math.cos(angle)
    return point(x, y)

cdef bint check(int x, int y, list shape):
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
        l = [list(zip(*pnp.nonzero(ans & (~vis))))[0]]
        comp = []
        comp.append(point(l[0][0] + sx, l[0][1] + sy))
        while len(l) != 0:
            x, y = l[0]
            vis[x, y] = 1
            l = l[1:]
            for dx, dy in var:
                xn, yn = x + dx, y + dy
                if check(xn, yn, [img.shape[0], img.shape[1]]) and ans[xn, yn] and not vis[xn, yn]:
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
    return check_ray(a, b, p) and check_ray(b, a, p)

cdef bint in_polygon(point a, list p):
    cdef double angle = 0
    cdef int j
    cdef point v1, v2
    for i in range(len(p)):
        j = (i + 1) % len(p)
        if check_seg(p[i], p[j], a):
            return False
        v1 = vec(a, p[i])
        v2 = vec(a, p[j])
        angle += get_angle(v1, v2)
    angle = abs(angle)
    return angle >= math.pi 

cdef tuple get_line(point p1, point p2):
    cdef double a, b, c
    a = p2.y - p1.y
    b = p1.x - p2.x
    c = -(a * p1.x + b * p1.y)
    return a, b, c

cdef double distpl(point p, double a, double b, double c):
    return abs(a * p.x + b * p.y + c) / ((a ** 2 + b ** 2) ** 0.5)

cdef tuple intersect_ray_line(double a, double b, double c, point p0, point d):
    if a * d.x + b * d.y == 0:
        return True, p0
    cdef double t = -(a * p0.x + b * p0.y + c) / (a * d.x + b * d.y)
    if t < 0:
        return False, point(-1, -1)
    else:
        return True, d * t + p0

cdef point intersect_lines(double a1, double b1, double c1, double a2, double b2, double c2):
    cdef double x, y
    y = (c2 * a1 - a2 * c1) / (b1 * a2 - a1 * b2)
    x = (c2 * b1 - c1 * b2) / (a1 * b2 - a2 * b1)
    return point(x, y)

cdef bint in_angle(point v1, point v2, point p):
    cdef double s = (-v1.x * (v2.y - v1.y) + v1.y * (v2.x - v1.x)) / 2
    if s < 0:
        v1, v2 = v2, v1
    cdef double s1 = (-v1.x * (p.y - v1.y) + v1.y * (p.x - v1.x)) / 2
    cdef double s2 = (-v2.x * (p.y - v2.y) + v2.y * (p.x - v2.x)) / 2
    return s1 >= 0 and s2 <= 0

cdef tuple bisec(point a, point b):
    cdef double angle = get_angle(a, b) / 2
    cdef point p1 = rot(a, angle)
    cdef point p2 = vec(rot(a, angle), point(0, 0))
    return p1 * (1 / p1.len()), p2 * (1 / p2.len())

cdef tuple shift_segment(int a, int b, int c, int d, list pol, double s):
    cdef point b1a, b1b, b2a, b2b, b1, b2
    b1a, b1b = bisec(vec(pol[b], pol[a]), vec(pol[b], pol[c]))
    b2a, b2b = bisec(vec(pol[c], pol[b]), vec(pol[c], pol[d]))
    if in_polygon(pol[b] + b1a, pol):
        b1 = b1a
    else:
        b1 = b1b
    if in_polygon(pol[c] + b2a, pol):
        b2 = b2a
    else:
        b2 = b2b
    cdef double pa, pb, pc
    pa, pb, pc = get_line(pol[b], pol[c])
    cdef point mv
    mv = point(pa, pb)
    mv = (mv * (1 / mv.len())) * s
    pc += pa * mv.x + pb * mv.y
    cdef bint f1, f2
    cdef point pint1, pint2
    f1, pint1 = intersect_ray_line(pa, pb, pc, pol[b], b1)
    f2, pint2 = intersect_ray_line(pa, pb, pc, pol[c], b2)
    if not f1 or not f2:
        pc -= 2 * (pa * mv.x + pb * mv.y)
        f1, pint1 = intersect_ray_line(pa, pb, pc, pol[b], b1)
        f2, pint2 = intersect_ray_line(pa, pb, pc, pol[c], b2)
    return pint1, pint2

cdef list component_fill(double s, list cords, list holes, int level):
    if level == 1:
        return cords
    cdef list ans = []
    cdef point p1, p2
    cdef bint f 
    for i in range(1, len(cords)):
        p1, p2 = shift_segment(i - 1, i, (i + 1) % len(cords), (i + 2) % len(cords), cords, s)
        if not (p1 == point(-1, -1)) and in_polygon(p1, cords):
            f = True
            for j in range(len(holes)):
                if in_polygon(p1, holes[j]):
                    f = False
                    break
            if f:
                ans.append(p1)
        if not (p2 == point(-1, -1)) and in_polygon(p2, cords):
            f = True
            for j in range(len(holes)):
                if in_polygon(p2, holes[j]):
                    f = False
                    break
            if f:
                ans.append(p2)
    if ans:
        ans = remove_dublicates(ans)
        ans = remove_intersections(ans)
        cords = component_fill(s, ans, holes, level + 1)
    return cords

cpdef list compute_image(np.ndarray img, int d, double sx, double sy):
    cdef list cords, holes
    cords, holes = get_borders(img, sx, sy)
    cords = approximate(cords)
    cords = remove_dublicates(cords)
    cords = component_fill(d, cords, holes, 0)
    cords = [[a.x, a.y] for a in cords]
    return cords

cdef list remove_dublicates(list x):
    i = 0
    while i < len(x):
        while len(x) > 1 and i < len(x) and x[(i + 1) % len(x)] == x[i]:
            x.pop(i)
        i += 1
    return x 

cdef bint check_poly(list cords, double a, double b):
    cdef double s = 0
    for p in cords:
        s = max(s, abs(a * p.x + b - p.y))
    return s < 5

cdef list approximate(list cords):
    cdef int i = 0
    cdef list ncords = []
    cdef list now = []
    cdef double a = 1, b = 0, an, bn, cn
    while i < len(cords):
        now.append(cords[i])
        if len(now) < 2:
            i += 1
            continue
        try:
            res = pnp.polynomial.Polynomial.fit([n.x for n in now], [n.y for n in now], 1).convert().coef
            if len(res) == 1:
                bn = res[0]
                an = 0
            else: bn, an = res
        except pnp.linalg.LinAlgError:
            i += 1
            continue
        if not check_poly(now, an, bn):
            ncords.extend((now[0], now[-2]))
            now = [cords[i]]
        a = an
        b = bn
        i += 1
    if now:
        ncords.extend([now[0], now[-1]])
    return ncords

cdef convert(list l):
    ans = []
    for elem in l:
        ans.append(point(elem[0], elem[1]))
    return ans
    
cpdef list remove_intersections(list line):
    if line and not isinstance(line[0], point):
        line = convert(line)
    cdef point intcord
    cdef bint flag 
    cdef int i = 0, j
    cdef int a, b, c, d
    while i < len(line):
        j = i + 2
        while j < len(line):
            a, b, c, d = i, (i + 1) % len(line), j, (j + 1) % len(line)
            if a not in [b, c, d] and b not in [c, d] and c != d:
                try:
                    intcord, flag = intersect(line[a], line[b], line[c], line[d])
                except ZeroDivisionError:
                    print(i, j, len(line))
                    print(line[a], line[b], line[c], line[d])
                if flag:
                    # if len(line) - (c - a) > c - a:
                    #     line = line[:b] + [intcord] + line[d:]
                    #     j = i + 1
                    # elif len(line) - (c - a) < 5:
                    #     line = line[b:d] + [intcord]
                    #     i = 0
                    #     j = i
                    # else:
                    line = line[:a + 1] + line[c:b + 1:-1] + line[d:]
                    j = i + 3
            j += 1
            # print(line)
        i += 1
        print('end2')
    print('end')
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
        if cord == p1 or cord == p2 or cord == q1 or cord == q2:
            return [cord, False]
        cord = get_intersection(p1, q1, p2, q2)
        return [cord, True]

    if ((o1 == 0) and on_segment(p1, p2, q1)): 
        if cord == p1 or cord == p2 or cord == q1 or cord == q2:
            return [cord, False]
        cord = get_intersection(p1, q1, p2, q2)
        return [cord, True]

    if ((o2 == 0) and on_segment(p1, q2, q1)): 
        if cord == p1 or cord == p2 or cord == q1 or cord == q2:
            return [cord, False]
        cord = get_intersection(p1, q1, p2, q2)
        return [cord, True]

    if ((o3 == 0) and on_segment(p2, p1, q2)): 
        if cord == p1 or cord == p2 or cord == q1 or cord == q2:
            return [cord, False]
        cord = get_intersection(p1, q1, p2, q2)
        return [cord, True]

    if ((o4 == 0) and on_segment(p2, q1, q2)): 
        if cord == p1 or cord == p2 or cord == q1 or cord == q2:
            return [cord, False]
        cord = get_intersection(p1, q1, p2, q2)
        return [cord, True]

    return [cord, False]

cdef point get_intersection(point p1, point q1, point p2, point q2):
    cdef double a1, b1, c1, a2, b2, c2
    a1, b1, c1 = get_line(p1, q1)
    a2, b2, c2 = get_line(p2, q2)
    return intersect_lines(a1, b1, c1, a2, b2, c2)
