import numpy as np
import pytest
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

from clustering.kmeans import (assign_points, distortion, kmeans, kmeans_plus_plus,
                               minibatch_kmeans, minibatch_update, update_centroids)


@pytest.fixture
def blobs():
    return make_blobs(n_samples=500, centers=4, cluster_std=0.8, random_state=1)


def test_assign_points_one_hot():
    X = np.array([[0.0, 0.0], [5.0, 5.0], [0.2, -0.1]])
    C = np.array([[0.0, 0.0], [5.0, 5.0]])
    assert (assign_points(X, C) == [[1, 0], [0, 1], [1, 0]]).all()


def test_update_centroids_labels_and_onehot_agree():
    X = np.random.default_rng(0).normal(size=(50, 3))
    labels = np.random.default_rng(1).integers(0, 4, size=50)
    onehot = np.eye(4, dtype=int)[labels]
    assert np.allclose(update_centroids(X, labels, 4), update_centroids(X, onehot, 4))


def test_empty_cluster_gets_global_mean():
    X = np.array([[0.0], [2.0]])
    cens = update_centroids(X, np.array([0, 0]), 2)
    assert cens[1, 0] == pytest.approx(1.0)


def test_kmeans_pp_picks_data_points():
    X = np.random.default_rng(0).normal(size=(100, 2))
    C = kmeans_plus_plus(X, 5, seed=3)
    assert all(any(np.array_equal(c, x) for x in X) for c in C)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_kmeans_matches_sklearn(blobs, seed):
    X, _ = blobs
    labels, cens, hist = kmeans(X, 4, seed=seed)
    sk = KMeans(n_clusters=4, n_init=10, random_state=seed).fit(X)
    assert adjusted_rand_score(labels, sk.labels_) == pytest.approx(1.0)
    assert distortion(X, cens, labels) == pytest.approx(sk.inertia_, rel=1e-6)


def test_kmeans_distortion_never_increases(blobs):
    X, _ = blobs
    _, _, hist = kmeans(X, 6, seed=0, init="random")
    assert np.all(np.diff(hist) <= 1e-9)


def test_minibatch_update_is_running_mean():
    # feeding every point of one cluster with counts starting at 0
    # should leave that centroid at exactly their mean
    X = np.random.default_rng(0).normal(size=(30, 2)) + 10
    C = np.array([[10.0, 10.0], [-50.0, -50.0]])
    cens, cnts = minibatch_update(X, C, np.arange(30), np.zeros(2, dtype=int))
    assert np.allclose(cens[0], X.mean(axis=0))
    assert list(cnts) == [30, 0]


def test_minibatch_kmeans_finds_blobs(blobs):
    X, y = blobs
    labels, _, hist, counts = minibatch_kmeans(X, 4, batch_size=50, max_iters=100, seed=0)
    assert adjusted_rand_score(y, labels) > 0.95
    assert counts.sum() == 50 * 100
