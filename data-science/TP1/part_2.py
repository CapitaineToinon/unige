import numpy as np

# create 2 random vectors of length 7
u = np.random.randn(7)
v = np.random.randn(7)

# Perform some computations
r = np.dot(u, v)

# Creating a vector orthogonal to u and of the same norm
# Step 1: Generate a random vector
w = np.random.randn(7)

# Step 2: Make w orthogonal to u by subtracting its projection on u
proj_u_w = (np.dot(w, u) / np.dot(u, u)) * u
v_orthogonal = w - proj_u_w

# Step 3: Normalize v_orthogonal to make it of the same norm as the original v
v_orthogonal = v_orthogonal / np.linalg.norm(v_orthogonal) * np.linalg.norm(v)

# Check orthogonality and norm
dot_product = np.dot(u, v_orthogonal)
norm_v_orthogonal = np.linalg.norm(v_orthogonal)

print("Dot product (should be close to 0):", dot_product)
print("Norm of v (original):", np.linalg.norm(v))
print("Norm of v (orthogonal):", norm_v_orthogonal)

def cosine_similarity(u: np.array, v: np.array) -> np.float64:
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)

    if norm_u == 0 or norm_v == 0:
        raise ValueError("cannot use vectors with norm 0")

    return np.dot(u, v) / (norm_u * norm_v)

u = np.array([1, 2, 3])
v = np.array([4, 5, 6])
cos_theta = cosine_similarity(u, v)
print(f'Cosine of the angle between u and v: {cos_theta}')
print(type(cos_theta))
