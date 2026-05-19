import numpy as np
from scipy.spatial.distance import pdist, cdist

def rbf_kernel_median(X, Y):
    XY = np.vstack([X, Y])
    dists = pdist(XY)
    median_dist = np.median(dists)
    if median_dist == 0:
        median_dist = 1.0
    gamma = 1.0 / (2 * median_dist**2)
    K = np.exp(-gamma * cdist(X, Y, metric='sqeuclidean'))
    return K, gamma

def mmd_unbiased(X, Y):
    n = len(X)
    m = len(Y)
    K_xx, _ = rbf_kernel_median(X, X)
    K_yy, _ = rbf_kernel_median(Y, Y)
    K_xy, _ = rbf_kernel_median(X, Y)
    term1 = (K_xx.sum() - np.trace(K_xx)) / (n * (n - 1))
    term2 = (K_yy.sum() - np.trace(K_yy)) / (m * (m - 1))
    term3 = 2 * K_xy.mean()
    mmd2 = term1 + term2 - term3
    return max(0.0, mmd2)

def compute_mmd_score(etf_returns, reference_returns):
    if np.std(reference_returns) < 1e-8:
        reference_returns = reference_returns + np.random.normal(0, 1e-6, len(reference_returns))
    X = etf_returns.reshape(-1, 1)
    Y = reference_returns.reshape(-1, 1)
    mmd_val = mmd_unbiased(X, Y)
    return -mmd_val
