# classical-ml-from-scratch

Clustering algorithms and evaluation metrics written from scratch in NumPy, tested against scikit-learn and SciPy, and compared on 9 public datasets.

## What's implemented

| file | contents |
|---|---|
| `clustering/kmeans.py` | k-means (Lloyd's algorithm) with k-means++ or random init, mini-batch k-means with per-centroid learning rate 1/count |
| `clustering/hierarchical.py` | agglomerative clustering with single / complete / average / Ward linkage via Lance-Williams updates, dendrogram cuts |
| `clustering/dbscan.py` | DBSCAN (BFS cluster expansion), core / border / noise classification |
| `clustering/birch.py` | clustering features (N, LS, SS), a simplified BIRCH, k-means on the CF centroids |
| `clustering/metrics.py` | silhouette, purity, NMI (geometric), Rand index, contingency table |
| `clustering/pca.py` | PCA via eigendecomposition of the covariance matrix |

## Tests

```
pytest
```

78 tests, all passing. The point of these is to show that "from scratch" also means correct:

- NMI and Rand index match `sklearn.metrics` to 1e-9 on 20 random labelings each, and silhouette matches `silhouette_score` (including a `-1` cluster and singletons).
- Agglomerative merge heights match `scipy.cluster.hierarchy.linkage` for all four linkages, and the final labels match `sklearn.cluster.AgglomerativeClustering` exactly (ARI = 1).
- DBSCAN gives the same noise points as `sklearn.cluster.DBSCAN` and the same clusters up to relabelling; the core points match `core_sample_indices_`.
- k-means reaches the same partition and inertia (rel. 1e-6) as `sklearn.cluster.KMeans` on separated blobs, and its distortion never increases between iterations.
- Smaller checks: the Lance-Williams average update equals the directly computed average linkage, CF statistics are additive, mini-batch updates reproduce the running mean, and PCA matches sklearn up to the sign of each component.

## Results

```
python experiments/run_benchmarks.py
```

Datasets: synthetic `blobs`, `aniso` (stretched blobs), `unequal` (blobs with very different spreads), `moons` and `circles` from scikit-learn, plus `iris`, `wine`, `penguins` and `digits`. Every dataset is standardised and K is set to the true number of classes. Labels are only used for scoring.

NMI against the true labels:

| dataset | k-means | mini-batch k-means | agglomerative (ward) | agglomerative (single) | DBSCAN | BIRCH |
|---|---:|---:|---:|---:|---:|---:|
| blobs | **1.000** | 1.000 | 1.000 | 0.753 | 0.730 | 1.000 |
| aniso | 0.578 | 0.580 | 0.577 | 0.753 | **0.982** | 0.562 |
| unequal | 0.738 | 0.738 | **0.845** | 0.045 | 0.035 | 0.530 |
| moons | 0.378 | 0.381 | 0.554 | **1.000** | 0.976 | 0.302 |
| circles | 0.000 | 0.000 | 0.008 | **1.000** | 0.975 | 0.000 |
| iris | 0.659 | 0.673 | 0.676 | 0.742 | **0.761** | 0.660 |
| wine | 0.876 | **0.892** | 0.786 | 0.058 | 0.000 | 0.876 |
| penguins | 0.786 | 0.786 | **0.901** | 0.782 | 0.782 | 0.828 |
| digits | 0.695 | 0.513 | **0.797** | 0.039 | 0.021 | 0.695 |

Rand index, silhouette and the DBSCAN settings used for each dataset are in [`results/results.md`](results/results.md).

![shapes](plots/shapes.png)

![nmi heatmap](plots/nmi_heatmap.png)

What I took from this:

- **Centroid methods assume round clusters.** k-means, mini-batch k-means and BIRCH (which runs k-means on its CF centroids) split the moons and circles straight down the middle, with an NMI of 0 on circles. Single linkage and DBSCAN follow the shape and get them almost perfectly.
- **Single linkage is very sensitive to chaining.** It is perfect on moons and circles but collapses on wine, digits and the unequal blobs, where one thin bridge of points is enough to merge two classes.
- **DBSCAN has one density threshold for the whole dataset.** On `unequal`, an `eps` big enough for the spread-out cluster also joins the two tight ones, so everything ends up in a single cluster. With this `eps` rule it also returns a single cluster on wine (13-d) and digits (64-d), where pairwise distances are much less spread out than in 2-d.
- **Ward linkage is the most reliable on the real datasets.** It is the best on penguins and digits, and within about 0.1 NMI of the best on iris and wine.
- **Silhouette and the true labels can disagree.** On digits, DBSCAN puts 1784 of 1797 points in one cluster and the other 13 in noise. That gets the highest silhouette in the table (0.675), because the noise is treated as its own far-away cluster, but an NMI of only 0.02. A good internal score does not mean the clustering matches the real classes.

## Implementation notes

- **DBSCAN parameters.** `min_pts = 2 * d`, and `eps` is the 98th percentile of every point's distance to its `min_pts`-th neighbour (the k-distance curve). This uses no labels, so the DBSCAN numbers are not tuned against the answer.
- **Agglomerative clustering.** This keeps a full distance matrix and updates one row per merge with the Lance-Williams formula, so it is O(n²) memory and O(n³) time in the worst case. Fine for a few thousand points; digits (1797 points) takes about 4 s per linkage.
- **BIRCH.** My BIRCH is a flat list of CF entries, not a CF-tree. Each point joins the nearest entry if the radius stays under the threshold, otherwise it starts a new one. The global step is plain k-means on the CF centroids, not weighted by how many points each entry holds.
- **Empty clusters.** In k-means, a cluster that goes empty is reset to the global mean, and the loop stops once assignments stop changing.

## Running it

```
pip install -r requirements.txt
pytest
python experiments/run_benchmarks.py
```

Python 3.10+. The library itself needs only NumPy. scikit-learn, SciPy, matplotlib and palmerpenguins are used for the tests, datasets and plots.
