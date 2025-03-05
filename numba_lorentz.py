# numba_lorentz.py
import numpy as np
from numba import njit, prange, atomic

@njit(parallel=True, nogil=True)
def lorentzian_histogram_numba(n, bins=100, xmin=-10, xmax=10):
    xfac = bins / (xmax - xmin)
    counts = np.zeros(bins, dtype=np.int64)
    for i in prange(n):
        u = np.random.random()
        x = 1.0 / np.tan(np.pi * u)
        ix = int((x - xmin) * xfac)
        if 0 <= ix < bins:
            atomic.add(counts, ix, 1)
    return counts

if __name__ == "__main__":
    import sys
    n = int(sys.argv[1])
    counts = lorentzian_histogram_numba(n)
    print(f"Numba: n={n}, counts={counts}")
