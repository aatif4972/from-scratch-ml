import numpy as np

from ._utils import sq_dists


def _cross_dists(X, labels, a, b):
    return np.sqrt(sq_dists(X[labels == a], X[labels == b]))


def single_linkage(X, labels, a, b):
    return float(_cross_dists(X, labels, a, b).min())


def complete_linkage(X, labels, a, b):
    return float(_cross_dists(X, labels, a, b).max())


def average_linkage(X, labels, a, b):
    return float(_cross_dists(X, labels, a, b).mean())


def lance_williams(d_ac, d_bc, d_ab, n_a, n_b, n_c, linkage):
    """Distance from c to the merged cluster (a+b).
    Works on scalars or on arrays of c's. Ward expects squared distances."""
    if linkage == "single":
        return 0.5 * d_ac + 0.5 * d_bc - 0.5 * np.abs(d_ac - d_bc)
    if linkage == "complete":
        return 0.5 * d_ac + 0.5 * d_bc + 0.5 * np.abs(d_ac - d_bc)
    if linkage == "average":
        return (n_a / (n_a + n_b)) * d_ac + (n_b / (n_a + n_b)) * d_bc
    if linkage == "ward":
        t = n_a + n_b + n_c
        return ((n_a + n_c) / t) * d_ac + ((n_b + n_c) / t) * d_bc - (n_c / t) * d_ab
    raise ValueError(f"unknown linkage {linkage!r}")


def clusters_at_cut(merge_heights, cut):
    # every merge at or below the cut height removes one cluster
    merge_heights = np.asarray(merge_heights)
    return int(len(merge_heights) - np.sum(merge_heights <= cut) + 1)


def agglomerative(X, K, linkage="average"):
    """Bottom-up merging with Lance-Williams updates.

    Returns (labels, merge_history). Each history row is (id_a, id_b, height, size);
    original points have ids 0..n-1 and the j-th merge creates id n+j, same as scipy.
    """
    X = np.asarray(X, dtype=float)
    n = len(X)

    D = sq_dists(X, X)
    if linkage != "ward":
        D = np.sqrt(D)
    np.fill_diagonal(D, np.inf)

    # slot i of D holds one active cluster; when u and v merge, the new
    # cluster takes u's slot and v's slot is switched off
    ids = np.arange(n)
    sizes = np.ones(n)
    active = np.ones(n, dtype=bool)
    owner = np.arange(n)  # which slot each point belongs to
    new_id = n
    history = []

    for _ in range(n - K):
        flat = np.argmin(D)
        u, v = divmod(flat, n)
        d_ab = D[u, v]
        height = np.sqrt(d_ab) if linkage == "ward" else d_ab

        a, b = sorted((ids[u], ids[v]))
        history.append((a, b, height, sizes[u] + sizes[v]))

        others = active.copy()
        others[[u, v]] = False
        upd = lance_williams(D[u, others], D[v, others], d_ab,
                             sizes[u], sizes[v], sizes[others], linkage)
        D[u, others] = upd
        D[others, u] = upd

        D[v, :] = np.inf
        D[:, v] = np.inf
        active[v] = False
        sizes[u] += sizes[v]
        owner[owner == v] = u
        ids[u] = new_id
        new_id += 1

    # number the final clusters in order of their ids
    slots = np.flatnonzero(active)
    rank = np.empty(n, dtype=int)
    rank[slots[np.argsort(ids[slots])]] = np.arange(len(slots))
    labels = rank[owner]
    return labels, np.array(history)
