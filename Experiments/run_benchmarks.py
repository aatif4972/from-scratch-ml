"""Runs every algorithm on every dataset and writes results/results.md and plots/.

    python experiments/run_benchmarks.py
"""
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from clustering import agglomerative, birch, dbscan, kmeans, minibatch_kmeans
from clustering._utils import sq_dists
from clustering.metrics import contingency, nmi, rand_index, silhouette
from datasets import ALL

SEED = 42
BIRCH_THRESHOLD = 0.3


def eps_from_kdist(X, min_pts, q=0.98):
    """For every point take the distance to its min_pts-th nearest neighbour
    (itself included, same as dbscan()). eps is the q-quantile of those, so
    roughly 1-q of the points can end up with too sparse a neighbourhood.
    Uses no labels."""
    d = np.sqrt(sq_dists(X, X))
    kd = np.sort(d, axis=1)[:, min_pts - 1]
    return float(np.quantile(kd, q))


def run_all(X, K):
    min_pts = 2 * X.shape[1]
    eps = eps_from_kdist(X, min_pts)
    out = {
        "k-means": kmeans(X, K, seed=SEED)[0],
        "mini-batch k-means": minibatch_kmeans(X, K, batch_size=min(64, len(X)), seed=SEED)[0],
        "agglomerative (ward)": agglomerative(X, K, "ward")[0],
        "agglomerative (single)": agglomerative(X, K, "single")[0],
        "DBSCAN": dbscan(X, eps, min_pts),
        "BIRCH": birch(X, K, BIRCH_THRESHOLD, seed=SEED)[0],
    }
    return out, eps, min_pts


def main():
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "plots"), exist_ok=True)

    nmi_rows, sil_rows, ri_rows, notes = [], [], [], []
    labels_by_ds = {}
    algos = None
    for name, loader in ALL.items():
        X, y = loader()
        X = StandardScaler().fit_transform(X)
        K = len(np.unique(y))
        t = time.time()
        res, eps, min_pts = run_all(X, K)
        algos = list(res)
        labels_by_ds[name] = (X, y, res)
        nmi_rows.append([nmi(contingency(l, y)) for l in res.values()])
        ri_rows.append([rand_index(y, l) for l in res.values()])
        sil_rows.append([silhouette(X, l) for l in res.values()])
        noise = np.mean(res["DBSCAN"] == -1)
        notes.append(f"| {name} | {X.shape[0]} | {X.shape[1]} | {K} | {eps:.3f} | {min_pts} | {noise:.0%} |")
        print(f"{name:10s} done in {time.time() - t:.1f}s")

    def table(rows, title):
        head = "| dataset | " + " | ".join(algos) + " |"
        sep = "|---|" + "---:|" * len(algos)
        lines = [f"**{title}**", "", head, sep]
        for name, r in zip(ALL, rows):
            best = np.argmax(r)
            cells = [f"**{v:.3f}**" if j == best else f"{v:.3f}" for j, v in enumerate(r)]
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        return "\n".join(lines)

    md = [table(nmi_rows, "NMI against the true labels (higher is better)"), "",
          table(ri_rows, "Rand index against the true labels"), "",
          table(sil_rows, "Mean silhouette (no labels used)"), "",
          "**Data and DBSCAN settings**", "",
          "| dataset | n | d | K | eps (98% k-dist) | min_pts | DBSCAN noise |", "|---|---:|---:|---:|---:|---:|---:|",
          *notes]
    with open(os.path.join(ROOT, "results", "results.md"), "w") as f:
        f.write("\n".join(md) + "\n")

    plot_shapes(labels_by_ds)
    plot_nmi_heatmap(np.array(nmi_rows), algos)


def _scatter(ax, X, labels, title):
    noise = labels == -1
    ax.scatter(X[~noise, 0], X[~noise, 1], c=labels[~noise], cmap="tab10", s=8, vmin=0, vmax=9)
    ax.scatter(X[noise, 0], X[noise, 1], c="lightgrey", s=8, marker="x")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_shapes(labels_by_ds):
    show = ["k-means", "agglomerative (single)", "DBSCAN", "BIRCH"]
    rows = ["moons", "circles", "aniso"]
    fig, axes = plt.subplots(len(rows), len(show) + 1, figsize=(13, 7.5))
    for i, ds in enumerate(rows):
        X, y, res = labels_by_ds[ds]
        _scatter(axes[i, 0], X, y, f"{ds}: true labels")
        for j, a in enumerate(show):
            _scatter(axes[i, j + 1], X, res[a], f"{a}  (NMI {nmi(contingency(res[a], y)):.2f})")
    fig.suptitle("Non-convex shapes: centroid methods vs density/linkage methods (grey x = DBSCAN noise)")
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "plots", "shapes.png"), dpi=130)
    plt.close(fig)


def plot_nmi_heatmap(M, algos):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(M, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(algos)), algos, rotation=30, ha="right")
    ax.set_yticks(range(len(ALL)), list(ALL))
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    color="white" if M[i, j] < 0.6 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, label="NMI")
    ax.set_title("NMI of each algorithm on each dataset")
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "plots", "nmi_heatmap.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
