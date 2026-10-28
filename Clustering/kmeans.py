import numpy as np

from ._utils import nearest, sq_dists


def squared_distance(x_q, x_r):
    return float(np.sum((np.asarray(x_q) - np.asarray(x_r)) ** 2))


def assign_points(X, C):
    """One-hot (N, K) matrix, 1 where the point's nearest centroid is."""
    idx, _ = nearest(X, C)
    ans = np.zeros((len(X), len(C)), dtype=int)
    ans[np.arange(len(X)), idx] = 1
    return ans


def update_centroids(X, assignments, K):
    # assignments can be labels (N,) or a one-hot matrix (N, K)
    if assignments.ndim == 2:
        assignments = assignments.argmax(axis=1)
    glob = X.mean(axis=0)
    cens = np.empty((K, X.shape[1]))
    for k in range(K):
        pts = X[assignments == k]
        # an empty cluster falls back to the global mean
        cens[k] = pts.mean(axis=0) if len(pts) else glob
    return cens


def distortion(X, C, labels):
    return float(np.sum((X - C[labels]) ** 2))


def kmeans_plus_plus(X, K, seed=42, rng=None):
    if rng is None:
        rng = np.random.default_rng(seed)
    n = len(X)
    first = rng.integers(n)
    cens = [X[first]]
    mini = sq_dists(X, cens[-1:])[:, 0]
    for _ in range(1, K):
        prob = mini / mini.sum()
        nxt = rng.choice(n, p=prob)
        cens.append(X[nxt])
        # only the newest centroid can lower a point's min distance
        mini = np.minimum(mini, sq_dists(X, cens[-1:])[:, 0])
    return np.array(cens)


def kmeans(X, K, max_iters=100, seed=42, init="kmeans++"):
    """Lloyd's algorithm. Returns (labels, centroids, distortion per iteration)."""
    X = np.asarray(X, dtype=float)
    n = len(X)
    rng = np.random.default_rng(seed)
    if init == "kmeans++":
        cens = kmeans_plus_plus(X, K, rng=rng)
    elif init == "random":
        cens = X[rng.choice(n, size=K, replace=False)].copy()
    else:
        raise ValueError(f"unknown init {init!r}")

    hist = []
    labels = None
    for _ in range(max_iters):
        new_labels, _ = nearest(X, cens)
        if labels is not None and np.array_equal(new_labels, labels):
            break  # assignments stopped changing, so centroids won't move either
        labels = new_labels
        cens = update_centroids(X, labels, K)
        hist.append(distortion(X, cens, labels))

    return labels, cens, np.array(hist)


def minibatch_update(X, C, batch_indices, counts):
    """Sequential mini-batch update: each point moves its nearest centroid
    by 1/count of the way towards it."""
    cens = C.astype(float).copy()
    cnts = counts.copy()
    for i in batch_indices:
        idx = np.argmin(((cens - X[i]) ** 2).sum(axis=1))
        cnts[idx] += 1
        cens[idx] += (X[i] - cens[idx]) / cnts[idx]
    return cens, cnts


def minibatch_kmeans(X, K, batch_size, max_iters=200, seed=42):
    """Returns (labels, centroids, distortion per iteration, per-centroid counts)."""
    X = np.asarray(X, dtype=float)
    n = len(X)
    rng = np.random.default_rng(seed)
    cens = kmeans_plus_plus(X, K, rng=rng)
    counts = np.zeros(K, dtype=int)

    hist = []
    for _ in range(max_iters):
        batch = rng.choice(n, size=batch_size, replace=False)
        # assign the whole batch first, then apply the updates
        idxs, _ = nearest(X[batch], cens)
        for i, idx in zip(batch, idxs):
            counts[idx] += 1
            cens[idx] += (X[i] - cens[idx]) / counts[idx]

        labels, d = nearest(X, cens)
        hist.append(float(d.sum()))

    return labels, cens, np.array(hist), counts
