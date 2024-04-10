import numpy as np 
from numpy.ctypeslib import ndpointer 
import ctypes 

doublepp = ndpointer(dtype=np.uintp, ndim=1, flags='C') 

dll = ctypes.CDLL('dummy.dll', winmode=0) 

foobar = dll.foobar 
foobar.argtypes = [ctypes.c_int, ctypes.c_int, doublepp, doublepp] 
foobar.restype = None 

def get_args(x): 
    xpp = (x.__array_interface__['data'][0] + np.arange(x.shape[0])*x.strides[0]).astype(np.uintp) 
    m = ctypes.c_int(x.shape[0]) 
    n = ctypes.c_int(x.shape[1]) 
    return xpp, m, n 

