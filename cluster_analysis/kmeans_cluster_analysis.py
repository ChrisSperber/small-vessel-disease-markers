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

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from svd_marker_tools.config import PROSPECTIVE_SAMPLE_CLEAN_CSV, RNG_SEED
from svd_marker_tools.utils import Cols

K_CLUSTER_NUMBER = 5  # adapted from previous work Sperber et al. 2023 in J Neurology

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
VARIBELS_FOR_DESKEW = [
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
svd_data_df[VARIBELS_FOR_DESKEW] = np.log1p(svd_data_df[VARIBELS_FOR_DESKEW])

svd_data_df = svd_data_df.astype(float)

# z-standardise all variables
svd_data_df[:] = (svd_data_df - svd_data_df.mean()) / svd_data_df.std(ddof=0)

# %%
# cluster analysis
km = KMeans(
    n_clusters=K_CLUSTER_NUMBER,
    init="k-means++",
    n_init=5,
    max_iter=300,
    tol=1e-4,
    random_state=RNG_SEED,
)

labels = km.fit_predict(svd_data_df)  # array of cluster IDs (0..k-1)
centroids = km.cluster_centers_  # shape: (k, n_features)
inertia = km.inertia_  # within-cluster sum of squares

full_data_df["kmeans_cluster"] = labels + 1

# %%
