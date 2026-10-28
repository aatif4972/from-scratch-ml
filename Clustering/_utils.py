import numpy as np


def sq_dists(A, B):
    # all pairwise squared distances, shape (len(A), len(B)).
    # |a-b|^2 = |a|^2 + |b|^2 - 2a.b avoids building an (n, m, d) array
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    d = (A ** 2).sum(axis=1)[:, None] + (B ** 2).sum(axis=1)[None, :] - 2 * A @ B.T
    return np.maximum(d, 0.0)


def nearest(X, C):
    # index of the closest row of C for every row of X, plus that squared distance
    d = sq_dists(X, C)
    idx = d.argmin(axis=1)
    return idx, d[np.arange(len(X)), idx]
