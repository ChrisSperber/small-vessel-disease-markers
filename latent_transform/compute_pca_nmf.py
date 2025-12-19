"""Compute PCA and Non-Negative Matrix Factorisation (NMF) on SVD markers in prospective sample.

Highly skewed variables (which includes all counts of lacunes and cmb) were deskewed via
log-transform.
Standardisation and clipping is applied only on data for PCA, but not NMF.

Requirements:
    - the prospective data were prepared with the script in ../data_preparation

Outputs:
    - csv with pca/nmf scores per subject

"""

# %%
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF, PCA
from sklearn.preprocessing import MinMaxScaler

from svd_marker_tools.config import PROSPECTIVE_SAMPLE_CLEAN_CSV, RNG_SEED
from svd_marker_tools.utils import Cols

WINSORISE_SD = 5

VARIABLES_SVD = [
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
svd_data_df = full_data_df[VARIABLES_SVD].copy()

# log-transform skewed variables
svd_data_df[VARIABLES_FOR_DESKEW] = np.log1p(svd_data_df[VARIABLES_FOR_DESKEW])

svd_data_df = svd_data_df.astype(float)

# z-standardise all variables
svd_data_z_df = (svd_data_df - svd_data_df.mean()) / svd_data_df.std(ddof=0)
# clip extreme values > xSDs from the mean
svd_data_z_df = svd_data_z_df.clip(lower=-WINSORISE_SD, upper=WINSORISE_SD)

# prepare output df with IDs
output_df: pd.DataFrame = full_data_df[[Cols.ID, Cols.INSEL_PID]].copy()

# %%
# compute PCA
X = svd_data_z_df.to_numpy(dtype=float)
pca = PCA(n_components=2)
scores_pca = pca.fit_transform(X)

pca_scores_df = pd.DataFrame(
    scores_pca,
    index=svd_data_df.index,
    columns=["PCA1", "PCA2"],
)

output_df = pd.concat([output_df, pca_scores_df], axis=1)

# %%
# compute NMF
# NMF scores for the first component differ based on the total number of latents, hence the first
# latent component is once computed alone and once with a second component
X = svd_data_df.to_numpy(dtype=float)

# Min-Max Scale
scaler = MinMaxScaler(feature_range=(0, 1))
X_mm = scaler.fit_transform(X)

# NMF with 2 latents
nmf = NMF(
    n_components=2,
    init="nndsvda",
    solver="cd",
    max_iter=2000,
    random_state=RNG_SEED,
)


W = nmf.fit_transform(X_mm)  # subject scores, shape (n_subjects, 2)

nmf_scores_df = pd.DataFrame(
    W,
    index=svd_data_df.index,
    columns=["NMF1of2", "NMF2of2"],
)

output_df = pd.concat([output_df, nmf_scores_df], axis=1)

# NMF with 1 latent
nmf = NMF(
    n_components=1,
    init="nndsvda",
    solver="cd",
    max_iter=2000,
    random_state=RNG_SEED,
)


W = nmf.fit_transform(X_mm)  # subject scores, shape (n_subjects, 1)

nmf_scores_df = pd.DataFrame(
    W,
    index=svd_data_df.index,
    columns=["NMF1of1"],
)

output_df = pd.concat([output_df, nmf_scores_df], axis=1)

# %%
# store output
out_csv = Path(__file__).with_suffix(".csv")
output_df.to_csv(out_csv, index=False)

# %%
