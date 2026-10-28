import numpy as np
import pytest
from scipy.cluster.hierarchy import linkage as scipy_linkage
from scipy.spatial.distance import cdist
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

from clustering.hierarchical import (agglomerative, average_linkage, clusters_at_cut,
                                     complete_linkage, lance_williams, single_linkage)

LINKAGES = ["single", "complete", "average", "ward"]


@pytest.fixture
def data():
    return make_blobs(n_samples=120, centers=3, random_state=4)


def test_linkage_distances(data):
    X, y = data
    d = cdist(X[y == 0], X[y == 1])
    assert single_linkage(X, y, 0, 1) == pytest.approx(d.min())
    assert complete_linkage(X, y, 0, 1) == pytest.approx(d.max())
    assert average_linkage(X, y, 0, 1) == pytest.approx(d.mean())


def test_lance_williams_average_is_exact(data):
    # merging a and b with the LW formula should equal the direct average linkage
    X, y = data
    merged = np.where(y == 1, 0, y)
    lw = lance_williams(average_linkage(X, y, 0, 2), average_linkage(X, y, 1, 2),
                        average_linkage(X, y, 0, 1),
                        np.sum(y == 0), np.sum(y == 1), np.sum(y == 2), "average")
    assert lw == pytest.approx(average_linkage(X, merged, 0, 2))


@pytest.mark.parametrize("method", LINKAGES)
def test_merge_heights_match_scipy(data, method):
    X, _ = data
    _, hist = agglomerative(X, 1, linkage=method)
    ref = scipy_linkage(X, method=method)
    assert np.allclose(np.sort(hist[:, 2]), np.sort(ref[:, 2]))
    assert hist[-1, 3] == len(X)


@pytest.mark.parametrize("method", LINKAGES)
def test_labels_match_sklearn(data, method):
    X, _ = data
    labels, _ = agglomerative(X, 3, linkage=method)
    sk = AgglomerativeClustering(n_clusters=3, linkage=method).fit(X)
    assert adjusted_rand_score(labels, sk.labels_) == pytest.approx(1.0)


def test_clusters_at_cut():
    h = np.array([0.5, 1.0, 2.0, 2.0, 3.5])  # 6 points
    assert clusters_at_cut(h, 0.1) == 6
    assert clusters_at_cut(h, 2.0) == 2   # heights equal to the cut count as merged
    assert clusters_at_cut(h, 10) == 1
    assert clusters_at_cut(np.array([]), 1.0) == 1
