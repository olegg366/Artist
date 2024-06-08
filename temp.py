import ctypes as c
import numpy as np

import os
os.system('gcc -shared -fPIC -o trajectory.so trajectory.cpp -lstdc++')

lib = c.CDLL('trajectory.so')

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

lib.pcompute_image.argtypes = [IPOINTER2D, c.c_size_t, c.c_size_t, c.c_int32, c.c_longdouble, c.c_longdouble]
lib.pcompute_image.restype = IPOINTER2D

X = np.ones((2, 3), dtype='float128', order="C")
Y = np.ones((5, 3), dtype='float128')