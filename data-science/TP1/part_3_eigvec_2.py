import numpy as np

# Original vector
A = np.array([
    [5, 6, 3],
    [-1, 0, 1],
    [1, 2, -1],
])

# Add the eigen value
B = A - np.identity(3) * -2
print(B)

# Our result we're testing
eig_vec_1 = np.array([0, 1, -2])
print(eig_vec_1)
print(np.dot(B, eig_vec_1))