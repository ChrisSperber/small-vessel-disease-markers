"""Inference helpers for SVD autoencoder."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from autoencoder_model import TabularAutoencoderWithAuxHead
from autoencoder_preproc_dataset import TabularAEPreprocessor


def compute_svd_latent(
    df: pd.DataFrame,
    model_weights_path: str | Path,
    preproc_json_path: str | Path,
    *,
    device: str = "cpu",
    batch_size: int = 1024,
) -> np.ndarray:
    """Compute 1D latent score for each row in df.

    Args:
        df: DataFrame with one row per subject. Must contain the feature columns
            expected by the persisted preprocessor.
        model_weights_path: Path to model state_dict (.pt) containing 'model_state_dict'.
        preproc_json_path: Path to preprocessor JSON saved during training.
        device: "cpu" or "cuda".
        batch_size: Batch size for inference.

    Returns:
        latents: ndarray of shape (n_subjects,) if latent_dim==1,
            otherwise shape (n_subjects, latent_dim).

    """
    device_t = torch.device(device)

    # --- load preprocessing (includes the feature column list) ---
    preproc = TabularAEPreprocessor.load_json(preproc_json_path)
    x = preproc.transform_features(df)  # (N, D) float32

    # --- build model with correct input size and latent dim ---
    # latent_dim is set to current default.
    model = TabularAutoencoderWithAuxHead(
        n_features=x.shape[1],
        latent_dim=1,
        hidden_dim=16,
        dropout_p=0.0,
    ).to(device_t)

    # --- load weights ---
    state = torch.load(Path(model_weights_path), map_location=device_t)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)

    model.eval()

    # --- batched inference ---
    latents: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            xb = torch.from_numpy(x[start : start + batch_size]).to(device_t)
            zb = model.encode_latent(xb)  # (B, latent_dim)
            latents.append(zb.detach().cpu().numpy())

    z = np.concatenate(latents, axis=0)

    # return 1D if latent_dim==1
    if z.shape[1] == 1:
        return z[:, 0]

    return z
