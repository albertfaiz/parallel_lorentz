# dask_lorentz.py
import numpy as np
import dask
from dask import delayed

def lorentzian_histogram(n, bins=100, xmin=-10, xmax=10):
    """
    Sample n points from the Lorentzian distribution via inverse transform sampling.
    Returns the histogram counts (ignoring bin edges).
    """
    u = np.random.random(n)            # Uniform(0,1)
    x = 1.0 / np.tan(np.pi * u)          # x = 1/tan(pi*u)
    counts, _ = np.histogram(x, bins=bins, range=(xmin, xmax))
    return counts

@delayed
def delayed_lorentzian_histogram(n, bins=100, xmin=-10, xmax=10):
    return lorentzian_histogram(n, bins, xmin, xmax)

def run_dask(n, n_tasks=4, bins=100, xmin=-10, xmax=10):
    """
    Run the Lorentzian histogram sampling in parallel using Dask.
    Splits n samples among n_tasks and aggregates the histogram.
    """
    # Split n into nearly equal chunks for n_tasks
    chunks = (n // n_tasks) * np.ones(n_tasks, dtype=int)
    chunks[:n % n_tasks] += 1
    tasks = [delayed_lorentzian_histogram(chunk, bins, xmin, xmax) for chunk in chunks]
    results = dask.compute(*tasks)
    return np.sum(results, axis=0)
