import numpy as np


def pca(X, n_components=2):
    """Project onto the top principal components (eigenvectors of the covariance)."""
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    if not 1 <= n_components <= min(n, d):
        raise ValueError(f"n_components must be in [1, {min(n, d)}]")
    X_c = X - X.mean(axis=0)
    cov = np.cov(X_c, rowvar=False).reshape(d, d)
    vals, vecs = np.linalg.eigh(cov)
    top = np.argsort(vals)[::-1][:n_components]
    return X_c @ vecs[:, top]
