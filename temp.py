from ctypes import CDLL, POINTER
import ctypes
import numpy as np

import os
os.system('gcc -shared -fPIC -o mylib.so trajectory.cpp -lstdc++')

mylib = CDLL("mylib.so")

ND_POINTER_2 = np.ctypeslib.ndpointer(dtype=np.int32, 
                                      ndim=2,
                                      flags="C")

mylib.test_approximation.argtypes = [ND_POINTER_2, ctypes.c_size_t, POINTER(ctypes.c_longdouble), POINTER(ctypes.c_longdouble)]
mylib.test_approximation.restype = None

X = np.array([[-6, 2], [-3, 4], [1, 2], [2, 6], [7, 3]], dtype='int32', order='C')
a = ctypes.c_longdouble(0)
b = ctypes.c_longdouble(0)

vec = mylib.test_approximation(X, X.shape[0], ctypes.byref(a), ctypes.byref(b))
print(a, b)