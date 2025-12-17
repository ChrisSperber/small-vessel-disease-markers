"""Compute k-means cluster analysis on SVD marker in prospective sample.

Highly skewed variables (which includes all counts of lacunes and cmb) were deskewed via
log-transform.

Requirements:
    - the prospective data were prepared with the script in ../data_preparation

Outputs:
    - csv with the assignment of each subject to clusters
    - yaml with additional information on cluster assignments

"""

# %%
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import KMeans

from svd_marker_tools.config import PROSPECTIVE_SAMPLE_CLEAN_CSV, RNG_SEED
from svd_marker_tools.utils import Cols

K_CLUSTER_NUMBER = 5  # adapted from previous work Sperber et al. 2023 in J Neurology
WINSORISE_SD = 5

VARIABLES_CLUSTERING = [
    Cols.LACUNES,
    Cols.CMB_LOBAR,
    Cols.CMB_CENTRAL,
    Cols.CMB_INFRATENTORIAL,
    Cols.EPVS_CS,
    Cols.EPVS_BG,
    Cols.WMH_PV,
    Cols.WMH_DEEP,
]
VARIABLES_FOR_DESKEW = [
    Cols.LACUNES,
    Cols.CMB_LOBAR,
    Cols.CMB_CENTRAL,
    Cols.CMB_INFRATENTORIAL,
]

# %%
# format data
full_data_df = pd.read_csv(PROSPECTIVE_SAMPLE_CLEAN_CSV, sep=";")
svd_data_df = full_data_df[VARIABLES_CLUSTERING].copy()

# log-transform skewed variables
svd_data_df[VARIABLES_FOR_DESKEW] = np.log1p(svd_data_df[VARIABLES_FOR_DESKEW])

svd_data_df = svd_data_df.astype(float)

# store parameters to reconstruct standardised output scores later
means = svd_data_df.mean()
stds = svd_data_df.std(ddof=0)

# z-standardise all variables
svd_data_df[:] = (svd_data_df - svd_data_df.mean()) / svd_data_df.std(ddof=0)
# clip extreme values > 4SDs from the mean
svd_data_df[:] = svd_data_df.clip(lower=-WINSORISE_SD, upper=WINSORISE_SD)

# %%
# cluster analysis
km = KMeans(
    n_clusters=K_CLUSTER_NUMBER,
    init="k-means++",
    n_init=50,
    max_iter=300,
    tol=1e-4,
    random_state=RNG_SEED,
)

labels = km.fit_predict(svd_data_df)  # array of cluster IDs (0..k-1)
centroids = km.cluster_centers_  # shape: (k, n_features)
inertia = km.inertia_  # within-cluster sum of squares

output_df = full_data_df.copy()
output_df = output_df[[Cols.ID, Cols.INSEL_PID]]
output_df["kmeans_cluster"] = labels + 1

# create binary Cluster variables for downstream analyses
cluster_list = ["Cluster1", "Cluster2", "Cluster3", "Cluster4", "Cluster5"]
k = 0
for cluster in cluster_list:
    k = k + 1
    output_df[cluster] = (output_df["kmeans_cluster"] == k).astype(int)


# %%
# cluster sizes
cluster_sizes = np.bincount(labels, minlength=K_CLUSTER_NUMBER)

# compute centroids to original scores
centroids_z = km.cluster_centers_
# undo z-scoring
centroids_unscaled = centroids_z * stds.values + means.values
# undo log1p for deskewed variables
centroids_original = centroids_unscaled.copy()

for j, var in enumerate(VARIABLES_CLUSTERING):
    if var in VARIABLES_FOR_DESKEW:
        centroids_original[:, j] = np.expm1(centroids_original[:, j])

# centroids as "per-cluster dict of feature->value" for readability
centroids_named = []
for i in range(K_CLUSTER_NUMBER):
    centroids_named.append(
        {
            "cluster": int(i + 1),
            "centroid_z": {
                var: float(centroids_z[i, j])
                for j, var in enumerate(VARIABLES_CLUSTERING)
            },
            "centroid_original_scale": {
                var: float(centroids_original[i, j])
                for j, var in enumerate(VARIABLES_CLUSTERING)
            },
        }
    )

summary = {
    "analysis": "kmeans_svd_markers",
    "n_subjects": int(svd_data_df.shape[0]),
    "n_features": int(svd_data_df.shape[1]),
    "included_variables": list(VARIABLES_CLUSTERING),
    "preprocessing": {
        "deskew": {
            "method": "log1p",
            "variables": list(VARIABLES_FOR_DESKEW),
        },
        "standardisation": {"method": "zscore", "ddof": 0},
        "winsorise": {"method": "clip", "lower": -WINSORISE_SD, "upper": WINSORISE_SD},
    },
    "kmeans_params": {
        "n_clusters": int(km.n_clusters),  # pyright: ignore[reportAttributeAccessIssue]
        "init": km.init,  # pyright: ignore[reportAttributeAccessIssue]
        "n_init": int(km.n_init),  # pyright: ignore[reportAttributeAccessIssue]
        "max_iter": int(km.max_iter),  # pyright: ignore[reportAttributeAccessIssue]
        "algorithm": getattr(km, "algorithm", None),
    },
    "fit_results": {
        "inertia": float(km.inertia_),
        "cluster_sizes": {
            f"Cluster{i+1}": int(cluster_sizes[i]) for i in range(K_CLUSTER_NUMBER)
        },
        "centroids": centroids_named,
    },
}

# write yaml
out_yaml = Path("kmeans_svd_markers_summary.yml")
with out_yaml.open("w", encoding="utf-8") as f:
    yaml.safe_dump(summary, f, sort_keys=False, allow_unicode=True)

# write assignments csv (if you want it here too)
out_csv = Path(__file__).with_suffix(".csv")
output_df.to_csv(out_csv, index=False)

# %%
