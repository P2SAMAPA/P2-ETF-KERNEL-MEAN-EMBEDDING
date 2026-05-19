import numpy as np

def rbf_kernel(x, y, gamma=0.5):
    """RBF kernel: exp(-gamma * ||x-y||^2)"""
    diff = x - y
    return np.exp(-gamma * np.dot(diff, diff))

def mmd(samples1, samples2, kernel_func, gamma=0.5):
    """
    Compute unbiased estimate of MMD^2.
    """
    n1 = len(samples1)
    n2 = len(samples2)
    # Use matrix operations for speed
    # Compute pairwise distances and then kernel matrices
    # For 1D samples, we can compute directly
    K_xx = np.zeros((n1, n1))
    for i in range(n1):
        for j in range(n1):
            K_xx[i,j] = kernel_func(samples1[i], samples1[j], gamma=gamma)
    K_yy = np.zeros((n2, n2))
    for i in range(n2):
        for j in range(n2):
            K_yy[i,j] = kernel_func(samples2[i], samples2[j], gamma=gamma)
    K_xy = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            K_xy[i,j] = kernel_func(samples1[i], samples2[j], gamma=gamma)
    term1 = K_xx.mean()
    term2 = K_yy.mean()
    term3 = 2 * K_xy.mean()
    mmd2 = term1 + term2 - term3
    # Numerical issues can cause negative values (should be >=0)
    return max(mmd2, 0.0)

def compute_mmd_score(etf_returns, reference_returns, kernel='rbf', gamma=0.5):
    """
    Compute -MMD as score (higher = closer to reference).
    """
    if len(etf_returns) == 0 or len(reference_returns) == 0:
        return 0.0
    # Use a subset if too large for performance
    max_samples = 500
    if len(etf_returns) > max_samples:
        idx = np.random.choice(len(etf_returns), max_samples, replace=False)
        etf_returns = etf_returns[idx]
    if len(reference_returns) > max_samples:
        idx = np.random.choice(len(reference_returns), max_samples, replace=False)
        reference_returns = reference_returns[idx]
    # Compute MMD
    mmd_val = mmd(etf_returns, reference_returns, rbf_kernel, gamma=gamma)
    # Return negative MMD (so higher score means closer)
    return -mmd_val
