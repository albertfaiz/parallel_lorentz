#!/usr/bin/env python
# benchmark_parallel.py
import time
import numpy as np
import matplotlib.pyplot as plt
from dask_lorentz import run_dask, lorentzian_histogram

def benchmark_function(func, n, **kwargs):
    """
    Benchmark the given function 'func' with sample size n.
    Returns the runtime in seconds and the function result.
    """
    start = time.perf_counter()
    result = func(n, **kwargs)
    end = time.perf_counter()
    return end - start, result

def main():
    # Define sample sizes (you can adjust these as needed)
    sample_sizes = [10**i for i in range(5, 8)]  # e.g. 1e5, 1e6, 1e7 samples
    # Concurrency levels to test for the Dask version:
    n_tasks_values = [1, 2, 4, 8]
    bins = 100
    xmin, xmax = -10, 10

    # Lists to store baseline (single-threaded) times
    baseline_times = []
    # Dictionary to store Dask times for each concurrency level
    dask_times = {n_tasks: [] for n_tasks in n_tasks_values}

    print("Starting benchmarking...\n")
    for n in sample_sizes:
        print(f"Testing sample size: n = {n}")
        # Baseline using pure NumPy (runs the base function)
        t_base, _ = benchmark_function(lorentzian_histogram, n, bins=bins, xmin=xmin, xmax=xmax)
        baseline_times.append(t_base)
        print(f"  Baseline (NumPy): {t_base:.4f} s")
        # Benchmark Dask for different numbers of tasks
        for n_tasks in n_tasks_values:
            t_dask, _ = benchmark_function(run_dask, n, n_tasks=n_tasks, bins=bins, xmin=xmin, xmax=xmax)
            dask_times[n_tasks].append(t_dask)
            print(f"  Dask with {n_tasks} tasks: {t_dask:.4f} s")
        print()

    # Plot Runtime vs. Number of Samples
    plt.figure(figsize=(10,6))
    plt.loglog(sample_sizes, baseline_times, marker='o', label='Baseline (NumPy)')
    for n_tasks in n_tasks_values:
        plt.loglog(sample_sizes, dask_times[n_tasks], marker='o', label=f'Dask ({n_tasks} tasks)')
    plt.xlabel('Number of Samples (n)')
    plt.ylabel('Runtime (s)')
    plt.title('Runtime Scaling for Lorentzian Sampling')
    plt.legend()
    plt.grid(True, which="both", linestyle="--")
    plt.savefig("runtime_vs_n.png", dpi=300)
    plt.show()

    # Plot Speedup (Baseline / Dask) vs. Number of Samples
    plt.figure(figsize=(10,6))
    for n_tasks in n_tasks_values:
        speedups = np.array(baseline_times) / np.array(dask_times[n_tasks])
        plt.plot(sample_sizes, speedups, marker='o', label=f'Dask ({n_tasks} tasks)')
    plt.xlabel('Number of Samples (n)')
    plt.ylabel('Speedup (Baseline / Dask)')
    plt.title('Speedup Relative to Baseline for Different Concurrency Levels')
    plt.legend()
    plt.grid(True)
    plt.savefig("speedup_vs_n.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
