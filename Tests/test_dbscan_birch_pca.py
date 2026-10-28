import numpy as np
import pytest
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs, make_moons
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score

from clustering.birch import birch, cf_stats
from clustering.dbscan import dbscan, point_types
from clustering.pca import pca


@pytest.mark.parametrize("eps,min_pts", [(0.15, 5), (0.2, 4), (0.25, 10)])
def test_dbscan_matches_sklearn(eps, min_pts):
    X, _ = make_moons(n_samples=300, noise=0.07, random_state=2)
    ours = dbscan(X, eps, min_pts)
    sk = DBSCAN(eps=eps, min_samples=min_pts).fit(X)
    assert np.array_equal(ours == -1, sk.labels_ == -1)
    assert adjusted_rand_score(ours, sk.labels_) == pytest.approx(1.0)


def test_point_types_core_matches_sklearn():
    X, _ = make_moons(n_samples=200, noise=0.1, random_state=0)
    types = point_types(X, 0.15, 5)
    sk = DBSCAN(eps=0.15, min_samples=5).fit(X)
    assert set(np.flatnonzero(types == 2)) == set(sk.core_sample_indices_)
    assert np.array_equal(types == 0, sk.labels_ == -1)


def test_cf_stats_and_additivity():
    rng = np.random.default_rng(0)
    A, B = rng.normal(size=(20, 3)), rng.normal(size=(15, 3)) + 2
    nA, lsA, ssA, _, _ = cf_stats(A)
    nB, lsB, ssB, _, _ = cf_stats(B)
    n, ls, ss, cen, rad = cf_stats(np.vstack([A, B]))
    # the whole point of CFs: merging two clusters is just adding their stats
    assert n == nA + nB and np.allclose(ls, lsA + lsB) and ss == pytest.approx(ssA + ssB)
    assert np.allclose(cen, np.vstack([A, B]).mean(axis=0))
    assert rad == pytest.approx(np.sqrt(ss / n - np.sum(cen ** 2)))


def test_birch_finds_blobs():
    X, y = make_blobs(n_samples=400, centers=[[0, 0], [6, 0], [3, 5]], cluster_std=0.7, random_state=5)
    labels, counts, cens, radii = birch(X, 3, threshold=0.8)
    assert adjusted_rand_score(y, labels) > 0.95
    assert counts.sum() == len(X)
    assert np.all(radii <= 0.8 + 1e-9)


def test_pca_matches_sklearn_up_to_sign():
    X = np.random.default_rng(0).normal(size=(80, 5)) @ np.diag([5, 3, 1, 0.5, 0.1])
    ours = pca(X, 3)
    ref = PCA(n_components=3).fit_transform(X)
    for k in range(3):
        assert np.allclose(ours[:, k], ref[:, k]) or np.allclose(ours[:, k], -ref[:, k])


def test_pca_constant_data_is_zero():
    assert np.allclose(pca(np.ones((10, 3)), 2), 0)
