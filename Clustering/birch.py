import numpy as np

from ._utils import nearest
from .kmeans import kmeans


def cf_stats(X):
    """Clustering feature of a set of points: (N, LS, SS, centroid, radius)."""
    X = np.asarray(X, dtype=float)
    n = len(X)
    ls = X.sum(axis=0)
    ss = float(np.sum(X ** 2))
    cen = ls / n
    rad = float(np.sqrt(np.sum((X - cen) ** 2) / n))
    return n, ls, ss, cen, rad


def _radius(n, ls, ss):
    # sqrt(SS/N - |LS/N|^2), clipped because rounding can make it slightly negative
    n = np.asarray(n, dtype=float)
    mu = ls / n[..., None]
    return np.sqrt(np.maximum(ss / n - np.sum(mu ** 2, axis=-1), 0.0))


def birch(X, K, threshold, seed=42):
    """Simplified BIRCH: a single flat list of CF entries (no CF-tree),
    then k-means on the CF centroids.

    Returns (labels, cf_counts, cf_centroids, cf_radii).
    """
    X = np.asarray(X, dtype=float)
    n, dim = X.shape

    N = np.zeros(n)
    LS = np.zeros((n, dim))
    SS = np.zeros(n)
    N[0], LS[0], SS[0] = 1, X[0], np.sum(X[0] ** 2)
    m = 1  # number of CF entries so far

    for i in range(1, n):
        cens = LS[:m] / N[:m, None]
        idx = np.argmin(((cens - X[i]) ** 2).sum(axis=1))
        n2, ls2, ss2 = N[idx] + 1, LS[idx] + X[i], SS[idx] + np.sum(X[i] ** 2)
        if _radius(n2, ls2, ss2) > threshold:
            N[m], LS[m], SS[m] = 1, X[i], np.sum(X[i] ** 2)
            m += 1
        else:
            N[idx], LS[idx], SS[idx] = n2, ls2, ss2

    N, LS, SS = N[:m], LS[:m], SS[:m]
    cf_cens = LS / N[:, None]
    cf_radii = _radius(N, LS, SS)

    if m < K:
        raise ValueError(f"only {m} CF entries for K={K}, lower the threshold")
    cf_labels, _, _ = kmeans(cf_cens, K, seed=seed)

    # each point takes the global cluster of its nearest CF entry
    idx, _ = nearest(X, cf_cens)
    return cf_labels[idx], N.astype(int), cf_cens, cf_radii
