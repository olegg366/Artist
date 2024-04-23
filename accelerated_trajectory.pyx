cimport numpy as np

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

cpdef list get_borders(np.ndarray img, np.ndarray ans):
    cdef int val
    img = img.astype('int')
    cdef int xi, yi
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

cdef double cos(double x1, double y1, double x2, double y2):
    cdef double ln1 = (x1 ** 2 + y1 ** 2) ** 0.5
    cdef double ln2 = (x2 ** 2 + y2 ** 2) ** 0.5
    return (x1 * x2 + y1 * y2) / (ln1 * ln2)

cdef list compute_shift(int x1, int y1, int x2, int y2, int xc, int yc, double d):
    cdef double xm = (x1 + x2) / 2
    cdef double ym = (y1 + y2) / 2
    cdef double ln = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
    cdef double xd = (y2 - y1) / ln
    cdef double yd = (x1 - x2) / ln
    if cos(xd, yd, xc, yc) < 0:
        xd = (y1 - y2) / ln
        yd = (x2 - x1) / ln
    xd *= d
    yd *= d
    return [xd + xm, yd + ym]

cpdef get_trajectory(int x, int y, int xc, int yc, int delx, int dely, double d, np.ndarray nimg, np.ndarray img, np.ndarray vis, list var):
    cdef list l = [[x, y]]
    cdef list ans = []
    cdef int xp = -1, yp = -1, xn, yn
    cdef list shp = [img.shape[0], img.shape[1]]
    cdef double xf, yf
    while len(l) != 0:
        x, y = l[0]
        l.pop(0)
        for dx, dy in var:
            xn, yn = x + dx, y + dy
            if check(xn, yn, shp) and img[xn, yn] and not vis[xn, yn]:
                vis[xn, yn] = 1
                l.append([xn, yn])
                if xp != -1 and yp != -1:
                    xf, yf = compute_shift(xn, yn, x, y, xc, yc, d)
                    ans.append([xf, yf])
                    x, y = int(round(xf)), int(round(yf))
                    nimg[x + delx, y + dely] = 0.9
                    break
                xp, yp = xn, yn
    return [ans, nimg]