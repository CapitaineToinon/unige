import numpy as np

A = np.array([
    [5, 6, 3],
    [-1, 0, 1],
    [1, 2, -1],
])

print(np.linalg.eigvals(A))