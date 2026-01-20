"""Compute the autoencoder-derived latents for the prospective data.

Requirements:
    - an autoencoder has been succesfully trained and its model and preproc.json are referenced to
        in the constants

Outputs:
    - csv with latent SVD variable per subject

"""

# %%
from pathlib import Path

import pandas as pd
from autoencoder_inference_functions import compute_svd_latent

from svd_marker_tools.config import PROSPECTIVE_SAMPLE_CLEAN_CSV
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

# %%
# load data and compute latent
prosepctive_data_df = pd.read_csv(PROSPECTIVE_SAMPLE_CLEAN_CSV, sep=";")

prospective_latents_arr = compute_svd_latent(
    df=prosepctive_data_df,
    model_weights_path=MODEL_WEIGHTS,
    preproc_json_path=PREPROC_JSON,
)

prosepctive_data_df[LATENT_SVD_SCORE] = prospective_latents_arr.round(3)

# %%
# store latents with identifiers
output_df = prosepctive_data_df[[Cols.ID, Cols.INSEL_PID, LATENT_SVD_SCORE]]
outfile = Path(__file__).with_suffix(".csv")

output_df.to_csv(outfile, sep=";", index=False)

# %%
