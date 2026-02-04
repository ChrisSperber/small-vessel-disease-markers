"""Evaluate autoencoder model in comparison to SVD burden score and simple latents.

A first evaluation is done within the autoencoder training sample.
TODO: second sample when NIHSS available

Requirements:
    - an autoencoder has been succesfully trained and its model and preproc.json are referenced to
        in the constants
    - simple latent transform were computed and stored in a csv
    - additional prospective data csv was created via load_additional_prospective_data.py

Outputs:
    - txt file with statistical results

"""

# %%
from pathlib import Path

import pandas as pd
from autoencoder_inference_functions import compute_svd_latent
from scipy.stats import kendalltau
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from svd_marker_tools.config import (
    PROSPECTIVE_SAMPLE_CLEAN_CSV,
    RETROSPECTIVE_SAMPLE_CLEAN_CSV,
)
from svd_marker_tools.utils import LATENT_SVD_SCORE, Cols

MODEL_WEIGHTS = (
    Path(__file__).parent
    / "outputs"
    / "svd_ae_output_20260109_2329"
    / "20260109_2329_best_model_weights.pt"
)
PREPROC_JSON = (
    Path(__file__).parent / "outputs" / "svd_ae_output_20260109_2329" / "preproc.json"
)

SIMPLE_LATENTS_CSV = (
    Path(__file__).parents[1] / "latent_transform" / "compute_pca_nmf.csv"
)

# %%
# load retrospective data and compute latent
retrospective_data_df = pd.read_csv(RETROSPECTIVE_SAMPLE_CLEAN_CSV, sep=";")

retrospective_latents_arr = compute_svd_latent(
    df=retrospective_data_df,
    model_weights_path=MODEL_WEIGHTS,
    preproc_json_path=PREPROC_JSON,
)

retrospective_data_df[LATENT_SVD_SCORE] = retrospective_latents_arr

# %%
# load prospective data and compute latent
prosepctive_data_df = pd.read_csv(PROSPECTIVE_SAMPLE_CLEAN_CSV, sep=";")

prospective_latents_arr = compute_svd_latent(
    df=prosepctive_data_df,
    model_weights_path=MODEL_WEIGHTS,
    preproc_json_path=PREPROC_JSON,
)

prosepctive_data_df[LATENT_SVD_SCORE] = prospective_latents_arr


# %%
# load simple latents
simple_latents_csv = pd.read_csv(SIMPLE_LATENTS_CSV, sep=",")

# %%
# evaluate autoencoder latent in retrospective data
svd_scores = [
    LATENT_SVD_SCORE,
    Cols.SVD_BURDEN_SCORE,
]
y_bin = retrospective_data_df[Cols.NIHSS_24H_BINARY_GT4].astype(int).to_numpy()
y_cont = retrospective_data_df[Cols.NIHSS_24H].to_numpy(dtype=float)

OUT_TXT = Path(__file__).with_suffix(".txt")

lines = []
lines.append("SVD autoencoder evaluation\n")
lines.append("Within-training-sample/retrospective data\n")
lines.append(f"N = {len(retrospective_data_df)}\n")

# --- 1) Logistic regression + AUC ---
lines.append("\nLogistic regression: NIHSS binary ~ SVD score (univariate)\n")
for score_col in svd_scores:
    x = retrospective_data_df[[score_col]].to_numpy(dtype=float)

    clf = LogisticRegression(solver="lbfgs", max_iter=10_000)
    clf.fit(x, y_bin)

    p = clf.predict_proba(x)[:, 1]
    auc = roc_auc_score(y_bin, p)

    beta = float(clf.coef_[0][0])
    intercept = float(clf.intercept_[0])

    lines.append(f"\nScore: {score_col}\n")
    lines.append(f"  coef (beta): {beta:.6f}\n")
    lines.append(f"  intercept:   {intercept:.6f}\n")
    lines.append(f"  AUC:         {auc:.4f}\n")

# --- 2) Kendall's tau SVD scores vs continuous NIHSS ---
lines.append("\nKendall's tau: NIHSS continuous vs SVD score\n")
for score_col in svd_scores:
    x = retrospective_data_df[score_col].to_numpy(dtype=float)
    tau, p = kendalltau(x, y_cont, nan_policy="omit")

    lines.append(f"\nScore: {score_col}\n")
    lines.append(f"  tau: {tau:.4f}\n")
    lines.append(f"  p:   {p:.4g}\n")

OUT_TXT.write_text("".join(lines), encoding="utf-8")
print(f"Wrote: {OUT_TXT}")

# %%
# evaluate latents in prospective data

# TODO

# %%
