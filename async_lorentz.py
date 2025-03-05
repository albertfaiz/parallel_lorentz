# async_lorentz.py
import asyncio
import numpy as np
from lorentzian_histogram import lorentzian_histogram

async def async_lorentzian_histogram(n, bins=100, xmin=-10, xmax=10):
    # Direct call since the function is CPU-bound
    return lorentzian_histogram(n, bins, xmin, xmax)[0]

async def get_counts(n, n_tasks=4, bins=100, xmin=-10, xmax=10):
    chunks = (n // n_tasks) * np.ones(n_tasks, dtype=int)
    chunks[:n % n_tasks] += 1
    tasks = [async_lorentzian_histogram(chunk, bins, xmin, xmax) for chunk in chunks]
    results = await asyncio.gather(*tasks)
    return np.sum(results, axis=0)

def run_async(n, n_tasks=4, bins=100, xmin=-10, xmax=10):
    return asyncio.run(get_counts(n, n_tasks, bins, xmin, xmax))

if __name__ == "__main__":
    import sys
    n = int(sys.argv[1])
    counts = run_async(n, n_tasks=4)
    print(f"AsyncIO: n={n}, counts={counts}")
