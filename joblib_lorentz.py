# joblib_lorentz.py
import numpy as np
from joblib import Parallel, delayed
from lorentzian_histogram import lorentzian_histogram

def run_joblib(n, n_jobs=4, bins=100, xmin=-10, xmax=10):
    chunks = (n // n_jobs) * np.ones(n_jobs, dtype=int)
    chunks[:n % n_jobs] += 1
    results = Parallel(n_jobs=n_jobs)(
        delayed(lorentzian_histogram)(chunk, bins, xmin, xmax) for chunk in chunks
    )
    # Each result is a tuple (counts, bin_edges)
    counts_list = [r[0] for r in results]
    return np.sum(counts_list, axis=0)

if __name__ == "__main__":
    import sys
    n = int(sys.argv[1])
    counts = run_joblib(n, n_jobs=4)
    print(f"Joblib: n={n}, counts={counts}")
