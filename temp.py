import os
os.system('./set_cython.sh')

import numpy as np
from cython_lib.accelerated_trajectory import get_trajectory

get_trajectory(np.zeros((2, 2), dtype='bool'), 2, 1, 1)