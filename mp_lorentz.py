# mp_lorentz.py
import multiprocessing
import numpy as np
from lorentzian_histogram import lorentzian_histogram
from functools import partial

def run_multiproc(n, n_cores=4, bins=100, xmin=-10, xmax=10):
    chunks = (n // n_cores) * np.ones(n_cores, dtype=int)
    chunks[:n % n_cores] += 1
    lorentzian_hist_func = partial(lorentzian_histogram, bins=bins, xmin=xmin, xmax=xmax)
    with multiprocessing.Pool(n_cores) as pool:
        results = pool.map(lorentzian_hist_func, chunks)
    # results is a list of tuples (counts, bin_edges), so we extract counts:
    counts_list = [r[0] for r in results]
    return np.sum(counts_list, axis=0)

if __name__ == "__main__":
    import sys
    n = int(sys.argv[1])
    counts = run_multiproc(n, n_cores=4)
    print(f"Multiprocessing: n={n}, counts={counts}")
