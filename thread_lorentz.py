# thread_lorentz.py
import threading
import numpy as np
from lorentzian_histogram import lorentzian_histogram

def add_chunk(n, counts, lock, bins=100, xmin=-10, xmax=10):
    local_counts, _ = lorentzian_histogram(n, bins, xmin, xmax)
    with lock:
        counts += local_counts

def run_threaded(n, n_threads=4, bins=100, xmin=-10, xmax=10):
    # Split n into nearly equal chunks
    chunks = (n // n_threads) * np.ones(n_threads, dtype=int)
    chunks[:n % n_threads] += 1
    threads = []
    counts = np.zeros(bins, dtype=int)
    lock = threading.Lock()
    for i in range(n_threads):
        t = threading.Thread(target=add_chunk, args=(chunks[i], counts, lock, bins, xmin, xmax))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    return counts

if __name__ == "__main__":
    import sys
    n = int(sys.argv[1])
    counts = run_threaded(n, n_threads=4)
    print(f"Threading: n={n}, counts={counts}")
