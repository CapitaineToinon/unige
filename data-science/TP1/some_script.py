import numpy as np
import warnings
from datetime import datetime

# Not a codesmell at all nope
warnings.filterwarnings("ignore", category=DeprecationWarning)

print('START - %s' % datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]);

# parameters
N = 2  # the dimension choosen


# functions
def scalar_product(u: np.array, v: np.array) -> np.array:
    """
    takes a pair of vectors (u, v) as input and returns their scalar product
    such that s = sum_{i=1..N}(u_i * v_i)
    """

    return np.dot(u, v)


def deviate_vector(u: np.array, v: np.array) -> np.array:
    """
    takes a pair of vectors (u, v) as input and returns a vector w defined by
    w = v - projection of v on u.
    """

    return v - project_on_first(u, v)


def project_on_first(u: np.array, v: np.array) -> np.array:
    """
    takes a pair of vectors (u, v) as input and returns the vector w which is
    the projection of v on u : w and u are colinear. All vectors are column!
    subspace is a build-in function of matlab which returns the angle between
    2 column vectors.
    (au)^2 + (v-au)^2 = v^2
    a^2u^2 + v^2 -2a(v,u) + a^2u^2 = v^2
    a^2u^2 -a(u,v) = 0
    a = (u,v)/u^2
    """
    return np.dot(u, v) / np.dot(u, u) * u


u = np.zeros(N)
u[np.random.randint(N)] = 1  # random canonic vector of dimension N
v = np.random.randn(N)  # random vector of dimension N
w = deviate_vector(u, v)  # w is a vector, function of u and v based on some geometrics

print(project_on_first(np.array([3, 4]), np.array([1, 2])))
print('Result = %.1f' % scalar_product(u, w));

print('END - %s\n' % datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]);
