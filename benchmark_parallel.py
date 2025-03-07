#!/usr/bin/env python
# benchmark_parallel.py

import time
import numpy as np
import matplotlib.pyplot as plt
from dask_lorentz import run_dask  # Your Dask wrapper function
from lorentzian_histogram import lorentzian_histogram  # Baseline NumPy function

def benchmark_function(func, n, **kwargs):
    start = time.perf_counter()
    result = func(n, **kwargs)
    end = time.perf_counter()
    return end - start, result

def main():
    sample_sizes = [10**7, 10**8, 10**9]  # Adjust as needed
    concurrency_levels = [1, 2, 4, 8]       # Number of tasks/workers
    bins = 100
    xmin, xmax = -10, 10

    # Baseline: pure NumPy
    baseline_times = []
    for n in sample_sizes:
        t, _ = benchmark_function(lorentzian_histogram, n, bins=bins, xmin=xmin, xmax=xmax)
        baseline_times.append(t)
        print(f"Baseline for n={n}: {t:.4f} s")

    # For one concurrency method (Dask in this example)
    dask_times = {lvl: [] for lvl in concurrency_levels}
    for n in sample_sizes:
        for lvl in concurrency_levels:
            t, _ = benchmark_function(run_dask, n, n_tasks=lvl, bins=bins, xmin=xmin, xmax=xmax)
            dask_times[lvl].append(t)
            print(f"Dask with {lvl} tasks for n={n}: {t:.4f} s")

    # Save results to a CSV file
    import pandas as pd
    data = []
    for i, n in enumerate(sample_sizes):
        row = {"n": n, "Baseline": baseline_times[i]}
        for lvl in concurrency_levels:
            row[f"Dask_{lvl}"] = dask_times[lvl][i]
        data.append(row)
    df = pd.DataFrame(data)
    df.to_csv("benchmark_results.csv", index=False)
    print("Benchmark results saved to benchmark_results.csv")

    # Plot Runtime vs. Sample Size (log-log plot)
    plt.figure(figsize=(10,6))
    plt.loglog(sample_sizes, baseline_times, marker='o', label='Baseline (NumPy)')
    for lvl in concurrency_levels:
        plt.loglog(sample_sizes, dask_times[lvl], marker='o', label=f'Dask ({lvl} tasks)')
    plt.xlabel('Number of Samples (n)')
    plt.ylabel('Runtime (s)')
    plt.title('Runtime Scaling for Lorentzian Sampling')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.savefig("runtime_vs_n.png", dpi=300)
    plt.show()

    # Plot Speedup: Baseline time / Dask time
    plt.figure(figsize=(10,6))
    for lvl in concurrency_levels:
        speedup = np.array(baseline_times) / np.array(dask_times[lvl])
        plt.plot(sample_sizes, speedup, marker='o', label=f'Dask ({lvl} tasks)')
    plt.xlabel('Number of Samples (n)')
    plt.ylabel('Speedup (Baseline / Dask)')
    plt.title('Speedup Relative to Baseline')
    plt.legend()
    plt.grid(True)
    plt.savefig("speedup_vs_n.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
