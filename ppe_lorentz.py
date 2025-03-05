# ppe_lorentz.py
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from lorentzian_histogram import lorentzian_histogram

def run_ppe(n, max_workers=4, bins=100, xmin=-10, xmax=10):
    chunks = (n // max_workers) * np.ones(max_workers, dtype=int)
    chunks[:n % max_workers] += 1
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(lorentzian_histogram, chunk, bins, xmin, xmax)
                   for chunk in chunks]
        results = [f.result()[0] for f in futures]
    return np.sum(results, axis=0)

if __name__ == "__main__":
    import sys
    n = int(sys.argv[1])
    counts = run_ppe(n, max_workers=4)
    print(f"ProcessPoolExecutor: n={n}, counts={counts}")
