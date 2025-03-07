#!/usr/bin/env python
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dask_lorentz import run_dask  # Dask-based parallel approach
from lorentzian_histogram import lorentzian_histogram  # Baseline NumPy implementation

def benchmark_function(func, n, **kwargs):
    """Measure runtime of func(n, **kwargs) and return (runtime, result)."""
    start = time.perf_counter()
    result = func(n, **kwargs)
    end = time.perf_counter()
    return end - start, result

def main():
    # Define sample sizes (you can adjust these based on your HPC resources)
    sample_sizes = [10**5, 10**6, 10**7, 10**8]
    # Define concurrency levels (number of tasks/workers)
    concurrency_levels = [1, 2, 4, 8, 16]
    bins = 100
    xmin, xmax = -10, 10

    # First, benchmark the baseline (pure NumPy)
    print("Running baseline benchmarks (pure NumPy)...")
    baseline_times = {}
    for n in sample_sizes:
        t, _ = benchmark_function(lorentzian_histogram, n, bins=bins, xmin=xmin, xmax=xmax)
        baseline_times[n] = t
        print(f"Baseline for n={n}: {t:.4f} s")

    # Now, benchmark the Dask version for each concurrency level
    results_list = []
    for n in sample_sizes:
        for n_tasks in concurrency_levels:
            t, _ = benchmark_function(run_dask, n, n_tasks=n_tasks, bins=bins, xmin=xmin, xmax=xmax)
            speedup = baseline_times[n] / t if t > 0 else np.nan
            results_list.append({
                'n': n,
                'n_tasks': n_tasks,
                'Runtime': t,
                'Baseline': baseline_times[n],
                'Speedup': speedup
            })
            print(f"Dask with {n_tasks} tasks for n={n}: {t:.4f} s, Speedup: {speedup:.2f}")

    # Save results to CSV
    df = pd.DataFrame(results_list)
    df.to_csv("benchmark_scaling_results.csv", index=False)
    print("Benchmark results saved to benchmark_scaling_results.csv")

    # Plot: Runtime vs. Number of Samples
    plt.figure(figsize=(10,6))
    for n_tasks in concurrency_levels:
        subset = df[df['n_tasks'] == n_tasks]
        plt.loglog(subset['n'], subset['Runtime'], marker='o', label=f'Dask ({n_tasks} tasks)')
    # Plot baseline as dashed line
    plt.loglog(list(baseline_times.keys()), list(baseline_times.values()), marker='o', linestyle='--', label='Baseline (NumPy)')
    plt.xlabel('Number of Samples (n)')
    plt.ylabel('Runtime (s)')
    plt.title('Runtime Scaling for Lorentzian Sampling')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.savefig("runtime_scaling.png", dpi=300)
    plt.show()

    # Plot: Speedup vs. Number of Samples
    plt.figure(figsize=(10,6))
    for n_tasks in concurrency_levels:
        subset = df[df['n_tasks'] == n_tasks]
        plt.plot(subset['n'], subset['Speedup'], marker='o', label=f'Dask ({n_tasks} tasks)')
    plt.xlabel('Number of Samples (n)')
    plt.ylabel('Speedup (Baseline Runtime / Dask Runtime)')
    plt.title('Speedup Relative to Baseline')
    plt.legend()
    plt.grid(True)
    plt.savefig("speedup_scaling.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
