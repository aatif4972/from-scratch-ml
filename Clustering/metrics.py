import numpy as np


def contingency(labels_pred, labels_true):
    """Rows = predicted clusters, columns = true classes."""
    _, p = np.unique(labels_pred, return_inverse=True)
    _, t = np.unique(labels_true, return_inverse=True)
    ans = np.zeros((p.max() + 1, t.max() + 1), dtype=int)
    np.add.at(ans, (p, t), 1)
    return ans


def silhouette(X, labels):
    """Mean silhouette with Euclidean distance.
    Points in singleton clusters score 0; noise (-1) is treated as a normal cluster."""
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    uniq, lab = np.unique(labels, return_inverse=True)
    if len(uniq) < 2:
        return 0.0
    sizes = np.bincount(lab)
    n = len(X)

    total = 0.0
    for i in range(n):
        d = np.sqrt(((X - X[i]) ** 2).sum(axis=1))
        sums = np.bincount(lab, weights=d, minlength=len(uniq))
        own = lab[i]
        if sizes[own] == 1:
            continue  # singleton, s = 0
        ai = sums[own] / (sizes[own] - 1)  # d[i] = 0 so it doesn't affect the sum
        means = sums / sizes
        means[own] = np.inf
        bi = means.min()
        total += (bi - ai) / max(ai, bi)
    return total / n


def purity(cont):
    n = cont.sum()
    if n == 0:
        return 0.0
    return float(cont.max(axis=1).sum() / n)


def nmi(cont):
    """NMI with geometric-mean normalisation, I(C;Y) / sqrt(H(C) H(Y))."""
    n = cont.sum()
    if n == 0:
        return 0.0
    p = cont / n
    pc = p.sum(axis=1)
    py = p.sum(axis=0)

    hc = -np.sum(pc[pc > 0] * np.log(pc[pc > 0]))
    hy = -np.sum(py[py > 0] * np.log(py[py > 0]))
    if hc == 0 and hy == 0:
        return 1.0
    if hc == 0 or hy == 0:
        return 0.0

    nz = p > 0
    outer = np.outer(pc, py)
    mi = np.sum(p[nz] * np.log(p[nz] / outer[nz]))
    return float(np.clip(mi / np.sqrt(hc * hy), 0.0, 1.0))


def rand_index(labels_true, labels_pred):
    """Fraction of point pairs on which the two labelings agree."""
    n = len(labels_true)
    if n < 2:
        return 1.0
    # count pairs from the contingency table instead of looping over all n^2 pairs
    cont = contingency(labels_pred, labels_true)
    pairs = lambda x: (x * (x - 1) // 2).sum()
    both_same = pairs(cont)                 # same cluster and same class
    same_pred = pairs(cont.sum(axis=1))
    same_true = pairs(cont.sum(axis=0))
    total = n * (n - 1) // 2
    both_diff = total - same_pred - same_true + both_same
    return (both_same + both_diff) / total
