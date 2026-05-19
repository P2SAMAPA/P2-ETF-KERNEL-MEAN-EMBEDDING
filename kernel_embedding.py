import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import kv, gamma

def rbf_kernel(x, y, gamma=0.5):
    """RBF kernel: exp(-gamma * ||x-y||^2)."""
    return np.exp(-gamma * np.sum((x - y)**2))

def matern_kernel(x, y, nu=1.5, length_scale=1.0):
    """Matérn kernel (nu=1.5)."""
    d = np.sqrt(np.sum((x - y)**2))
    if d == 0:
        return 1.0
    sqrt3 = np.sqrt(3)
    return (1 + sqrt3 * d / length_scale) * np.exp(-sqrt3 * d / length_scale)

def mmd(samples1, samples2, kernel_func, **kernel_params):
    """
    Compute Maximum Mean Discrepancy (squared) between two sets of samples.
    MMD^2 = E[k(X,X')] + E[k(Y,Y')] - 2 E[k(X,Y)]
    """
    n1 = len(samples1)
    n2 = len(samples2)
    # Compute kernel matrices
    K_xx = np.zeros((n1, n1))
    K_yy = np.zeros((n2, n2))
    K_xy = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n1):
            K_xx[i,j] = kernel_func(samples1[i], samples1[j], **kernel_params)
    for i in range(n2):
        for j in range(n2):
            K_yy[i,j] = kernel_func(samples2[i], samples2[j], **kernel_params)
    for i in range(n1):
        for j in range(n2):
            K_xy[i,j] = kernel_func(samples1[i], samples2[j], **kernel_params)
    term1 = K_xx.mean()
    term2 = K_yy.mean()
    term3 = 2 * K_xy.mean()
    return term1 + term2 - term3

def mmd_fast(samples1, samples2, kernel_func, **kernel_params):
    """
    Faster MMD using unbiased estimator (only for RBF with same bandwidth).
    Uses random subset if samples large.
    """
    # If samples are large, take a random subset to speed up
    max_samples = 500
    if len(samples1) > max_samples:
        idx1 = np.random.choice(len(samples1), max_samples, replace=False)
        samples1 = samples1[idx1]
    if len(samples2) > max_samples:
        idx2 = np.random.choice(len(samples2), max_samples, replace=False)
        samples2 = samples2[idx2]
    return mmd(samples1, samples2, kernel_func, **kernel_params)

def compute_mmd_score(etf_returns, reference_returns, kernel='rbf', gamma=0.5):
    """
    Compute -MMD as score (higher = closer to reference).
    """
    if kernel == 'rbf':
        kernel_func = rbf_kernel
        kernel_params = {'gamma': gamma}
    else:  # matern
        kernel_func = matern_kernel
        kernel_params = {'length_scale': gamma}
    # Ensure returns are 1D arrays (reshape to 2D for distance)
    # The kernel function expects scalars; we can treat each return as a point in 1D.
    # Reshape to (n,1) for distance computation.
    X = etf_returns.reshape(-1, 1)
    Y = reference_returns.reshape(-1, 1)
    # Compute MMD (squared)
    mmd_val = mmd_fast(X, Y, kernel_func, **kernel_params)
    # Return -MMD so that higher score means closer to reference
    return -mmd_val
