# lorentzian_histogram.py
import numpy as np
import matplotlib.pyplot as plt

def lorentzian_histogram(n, bins=100, xmin=-10, xmax=10):
    """
    Sample n random points from the Lorentzian distribution using inverse transform sampling.
    Returns the histogram counts.
    """
    # Generate n uniform random numbers in (0,1)
    u = np.random.random(n)
    # Transform: x = 1/tan(pi*u)
    x = 1.0 / np.tan(np.pi * u)
    counts, bin_edges = np.histogram(x, bins=bins, range=(xmin, xmax))
    return counts, bin_edges

if __name__ == "__main__":
    import sys
    n = int(sys.argv[1])
    counts, bin_edges = lorentzian_histogram(n)
    print(f"n={n}, counts={counts}")
    
    # For visual verification: plot normalized histogram vs Lorentzian PDF
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    plt.figure(figsize=(8,5))
    plt.hist(1.0 / np.tan(np.pi * np.random.random(n)), bins=100, range=(-10,10), density=True, alpha=0.6, label='Histogram')
    
    # Theoretical PDF: p(x) = 1/(pi*(1+x^2))
    x = np.linspace(-10,10,1000)
    pdf = 1.0/(np.pi*(1+x**2))
    plt.plot(x, pdf, 'r-', label='Theoretical PDF')
    plt.xlabel('x')
    plt.ylabel('Normalized Frequency')
    plt.legend()
    plt.title('Lorentzian Distribution via Inverse Transform Sampling')
    plt.show()
