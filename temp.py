import numpy as np

a = np.array([[0, 1, 0], 
              [0, 0, 0],
              [1, 0, 1]])

b = np.array([[0, 1, 0], 
              [0, 1, 0],
              [1, 0, 1]])

print(list(zip(*np.nonzero(a == b))))