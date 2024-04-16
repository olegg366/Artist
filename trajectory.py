import os
import numpy as np
from numpy.ctypeslib import ndpointer
import ctypes

path = 'D:\\Python_progs\\AI\\Artist\\'

os.system(f'gcc -Wall -pedantic -shared -fPIC -o {path}trajectory_lib.dll {path}trajectory_code.c')

doublepp = ndpointer(dtype=np.uintp, ndim=1, flags='C') 

dll = ctypes.CDLL(path + 'trajectory_lib.dll', winmode=0) 

get_trajectory = dll.get_trajectory 
get_trajectory.argtypes = [ctypes.c_int, ctypes.c_int, doublepp] 
get_trajectory.restype = None 

def get_args(x: np.ndarray): 
    xpp = (x.__array_interface__['data'][0] + np.arange(x.shape[0]) * x.strides[0]).astype(np.uintp) 
    m = ctypes.c_int(x.shape[0]) 
    n = ctypes.c_int(x.shape[1]) 
    return m, n, xpp

