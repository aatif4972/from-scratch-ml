from collections import deque

import numpy as np

NOISE = -1
UNSEEN = -2


def _neighbours(X, i, eps):
    # includes i itself, and points exactly eps away
    d = np.sqrt(((X - X[i]) ** 2).sum(axis=1))
    return np.flatnonzero(d <= eps)


def point_types(X, eps, min_pts):
    """2 = core, 1 = border, 0 = noise."""
    X = np.asarray(X, dtype=float)
    n = len(X)
    core = np.array([len(_neighbours(X, i, eps)) >= min_pts for i in range(n)])
    ans = np.zeros(n, dtype=int)
    ans[core] = 2
    core_pts = X[core]
    for i in np.flatnonzero(~core):
        if len(core_pts) and np.any(np.sqrt(((core_pts - X[i]) ** 2).sum(axis=1)) <= eps):
            ans[i] = 1
    return ans


def dbscan(X, eps, min_pts):
    """Labels 0..k-1 for clusters and -1 for noise."""
    X = np.asarray(X, dtype=float)
    n = len(X)
    labels = np.full(n, UNSEEN)
    cid = 0

    for i in range(n):
        if labels[i] != UNSEEN:
            continue
        neighs = _neighbours(X, i, eps)
        if len(neighs) < min_pts:
            labels[i] = NOISE  # may still become a border point later
            continue

        labels[i] = cid
        dq = deque(neighs)
        while dq:
            curr = dq.popleft()
            if labels[curr] == NOISE:
                labels[curr] = cid  # border point, don't expand from it
            if labels[curr] != UNSEEN:
                continue
            labels[curr] = cid
            neighs = _neighbours(X, curr, eps)
            if len(neighs) >= min_pts:
                dq.extend(neighs[labels[neighs] < 0])
        cid += 1

    return labels
