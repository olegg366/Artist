import numpy as np 
from numpy.ctypeslib import ndpointer 
import ctypes 
import os
from time import sleep

path = 'D:\\Python_progs\\AI\\Artist\\tests\\test_cython\\'

os.system(f'gcc -Wall -pedantic -shared -fPIC -o {path}lib.dll {path}test.c')

tpp = ndpointer(dtype=np.uintp, ndim=1, flags='C') 

dll = ctypes.CDLL(path + 'lib.dll', winmode=0) 

foobar = dll.foobar 
foobar.argtypes = [ctypes.c_int, ctypes.c_int, tpp] 
foobar.restype = None 

def get_args(x: np.ndarray): 
    xpp = (x.__array_interface__['data'][0] + np.arange(x.shape[0]) * x.strides[0]).astype(np.uintp) 
    return *x.shape, xpp

x = np.zeros([2, 2, 2], dtype=np.double)
print(get_args(x))
foobar(*get_args(x))