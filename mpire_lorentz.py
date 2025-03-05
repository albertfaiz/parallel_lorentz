# mpire_lorentz.py
import numpy as np
from mpire import WorkerPool
from lorentzian_histogram import lorentzian_histogram

def run_mpire(n, n_jobs=4, bins=100, xmin=-10, xmax=10):
    chunks = (n // n_jobs) * np.ones(n_jobs, dtype=int)
    chunks[:n % n_jobs] += 1
    with WorkerPool(n_jobs=n_jobs) as pool:
        results = pool.map(lorentzian_histogram, chunks, bins, xmin, xmax)
    counts_list = [res[0] for res in results]
    return np.sum(counts_list, axis=0)

if __name__ == "__main__":
    import sys
    n = int(sys.argv[1])
    counts = run_mpire(n, n_jobs=4)
    print(f"mpire: n={n}, counts={counts}")
