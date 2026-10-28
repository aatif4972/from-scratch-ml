import numpy as np
import pytest
from sklearn import metrics as skm
from sklearn.datasets import make_blobs

from clustering import metrics


@pytest.mark.parametrize("seed", range(20))
def test_nmi_matches_sklearn(seed):
    rng = np.random.default_rng(seed)
    n = rng.integers(2, 300)
    a = rng.integers(0, rng.integers(1, 8), size=n)
    b = rng.integers(0, rng.integers(1, 8), size=n)
    ours = metrics.nmi(metrics.contingency(b, a))
    ref = skm.normalized_mutual_info_score(a, b, average_method="geometric")
    assert ours == pytest.approx(ref, abs=1e-9)


@pytest.mark.parametrize("seed", range(20))
def test_rand_index_matches_sklearn(seed):
    rng = np.random.default_rng(seed)
    n = rng.integers(2, 300)
    a = rng.integers(0, 5, size=n)
    b = rng.integers(0, 5, size=n)
    assert metrics.rand_index(a, b) == pytest.approx(skm.rand_score(a, b), abs=1e-12)


@pytest.mark.parametrize("seed", range(5))
def test_silhouette_matches_sklearn(seed):
    X, _ = make_blobs(n_samples=200, centers=4, random_state=seed)
    labels = np.random.default_rng(seed).integers(-1, 4, size=len(X))  # include a "-1" cluster
    assert metrics.silhouette(X, labels) == pytest.approx(skm.silhouette_score(X, labels), abs=1e-9)


def test_silhouette_singleton_scores_zero():
    X = np.array([[0.0, 0.0], [0.0, 0.1], [10.0, 10.0]])
    labels = np.array([0, 0, 1])
    assert metrics.silhouette(X, labels) == pytest.approx(skm.silhouette_score(X, labels))


def test_silhouette_one_cluster_is_zero():
    X = np.random.default_rng(0).normal(size=(10, 2))
    assert metrics.silhouette(X, np.zeros(10)) == 0.0


def test_purity():
    cont = np.array([[5, 1, 0], [1, 4, 0], [0, 0, 3]])
    assert metrics.purity(cont) == pytest.approx(12 / 14)
    assert metrics.purity(np.zeros((2, 2))) == 0.0


def test_nmi_edge_cases():
    assert metrics.nmi(np.zeros((2, 2))) == 0.0
    assert metrics.nmi(np.array([[5]])) == 1.0          # both labelings constant
    assert metrics.nmi(np.array([[3, 2]])) == 0.0       # only one is constant
