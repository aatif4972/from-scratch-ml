"""Public datasets used for the benchmark. Real datasets are standardised."""
import numpy as np
from sklearn import datasets as skd
from sklearn.preprocessing import StandardScaler

SEED = 0


def _std(X):
    return StandardScaler().fit_transform(X)


def blobs():
    return skd.make_blobs(n_samples=450, centers=3, random_state=170)


def aniso():
    X, y = skd.make_blobs(n_samples=450, centers=3, random_state=170)
    return X @ np.array([[0.6, -0.64], [-0.4, 0.85]]), y


def unequal():
    return skd.make_blobs(n_samples=450, centers=3, cluster_std=[1.0, 2.5, 0.5], random_state=170)


def moons():
    return skd.make_moons(n_samples=800, noise=0.06, random_state=SEED)


def circles():
    return skd.make_circles(n_samples=800, factor=0.45, noise=0.05, random_state=SEED)


def iris():
    d = skd.load_iris()
    return _std(d.data), d.target


def wine():
    d = skd.load_wine()
    return _std(d.data), d.target


def penguins():
    from palmerpenguins import load_penguins
    df = load_penguins()
    cols = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
    df = df.dropna(subset=cols)
    y = df["species"].astype("category").cat.codes.to_numpy()
    return _std(df[cols].to_numpy()), y


def digits():
    d = skd.load_digits()
    return _std(d.data), d.target


ALL = {
    "blobs": blobs, "aniso": aniso, "unequal": unequal, "moons": moons, "circles": circles,
    "iris": iris, "wine": wine, "penguins": penguins, "digits": digits,
}
