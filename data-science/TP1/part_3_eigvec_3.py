import numpy as np

# Original vector
A = np.array([
    [5, 6, 3],
    [-1, 0, 1],
    [1, 2, -1],
])

# Add the eigen value
B = A - np.identity(3) * 4
print(B)

# Our result we're testing
eig_vec_1 = np.array([1, -(2/9), 1-(8/9)])
print(eig_vec_1)
print(np.dot(B, eig_vec_1))
