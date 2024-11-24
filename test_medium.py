import timeit
from numba import jit
import numpy as np

@jit
def compute_sum(arr):
    total = 0
    for x in arr:
        total += x
    return total
data = np.random.rand(1_000_000)

timeit.timeit(compute_sum(data))



