"""Preprocessing pipeline and dataset for SVD autoencoder."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class AEPreprocessConfig:
    """Config for SVD preprocessing pipeline."""

    feature_cols: Sequence[str]  # relevant cols of SVD markers
    nihss_col: str
    count_cols: Sequence[str]  # counts cols to be log1p transformed and standardised
    ordinal_cols: Sequence[str]  # ordinal cols to be standardised only
    standardize_features: bool = True
    standardize_nihss: bool = True


# Preprocessor (fit/transform + persist)
@dataclass
class _AEPreprocessState:
    cfg: AEPreprocessConfig
    feat_mean: list[float] | None = None
    feat_std: list[float] | None = None
    nihss_mean: float | None = None
    nihss_std: float | None = None


class TabularAEPreprocessor:
    """Process SVD dataframe via applicable transformations.

    - Applies log1p to count columns
    - Standardizes features / NIHSS using parameters fit on TRAIN only.
    - Can be saved/loaded as a JSON file for reuse in validation/external cohorts.
    """

    def __init__(self, cfg: AEPreprocessConfig):  # noqa: D107
        self.state = _AEPreprocessState(cfg=cfg)

        feature_set = set(cfg.feature_cols)
        if not set(cfg.count_cols).issubset(feature_set):
            missing = set(cfg.count_cols) - feature_set
            raise ValueError(f"count_cols not in feature_cols: {sorted(missing)}")
        if not set(cfg.ordinal_cols).issubset(feature_set):
            missing = set(cfg.ordinal_cols) - feature_set
            raise ValueError(f"ordinal_cols not in feature_cols: {sorted(missing)}")

    # ---------- persistence ----------
    def save_json(self, path: str | Path) -> None:  # noqa: D102
        path = Path(path)
        payload = {
            "schema_version": 1,
            "state": {
                "cfg": asdict(self.state.cfg),
                "feat_mean": self.state.feat_mean,
                "feat_std": self.state.feat_std,
                "nihss_mean": self.state.nihss_mean,
                "nihss_std": self.state.nihss_std,
            },
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> TabularAEPreprocessor:  # noqa: D102
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))

        if payload.get("schema_version") != 1:
            raise ValueError(
                f"Unsupported schema_version: {payload.get('schema_version')}"
            )

        st = payload["state"]
        cfg = AEPreprocessConfig(**st["cfg"])
        obj = cls(cfg)
        obj.state.feat_mean = st.get("feat_mean")
        obj.state.feat_std = st.get("feat_std")
        obj.state.nihss_mean = st.get("nihss_mean")
        obj.state.nihss_std = st.get("nihss_std")
        return obj

    # ---------- fitting ----------
    def fit(self, df_train: pd.DataFrame) -> TabularAEPreprocessor:  # noqa: D102
        cfg = self.state.cfg

        x = self._transform_features_no_standardize(df_train)

        if cfg.standardize_features:
            mean = x.mean(axis=0)
            std = x.std(axis=0, ddof=0)
            std = np.where(std == 0, 1.0, std)  # avoid /0 for constant columns
            self.state.feat_mean = mean.astype(np.float64).tolist()
            self.state.feat_std = std.astype(np.float64).tolist()

        if cfg.standardize_nihss:
            y = df_train[cfg.nihss_col].to_numpy(dtype=np.float32)
            nihss_mean = float(y.mean())
            nihss_std = float(y.std(ddof=0))
            if nihss_std == 0:
                nihss_std = 1.0
            self.state.nihss_mean = nihss_mean
            self.state.nihss_std = nihss_std

        return self

    # ---------- transforms ----------
    def transform_features(self, df: pd.DataFrame) -> np.ndarray:  # noqa: D102
        cfg = self.state.cfg
        x = self._transform_features_no_standardize(df)

        if cfg.standardize_features:
            if self.state.feat_mean is None or self.state.feat_std is None:
                raise RuntimeError(
                    "Preprocessor not fit() yet (missing feature scaling params)."
                )
            mean = np.asarray(self.state.feat_mean, dtype=np.float32)
            std = np.asarray(self.state.feat_std, dtype=np.float32)
            x = (x - mean) / std

        return x.astype(np.float32, copy=False)

    def transform_nihss(self, df: pd.DataFrame) -> np.ndarray:  # noqa: D102
        cfg = self.state.cfg
        y = df[cfg.nihss_col].to_numpy(dtype=np.float32)

        if cfg.standardize_nihss:
            if self.state.nihss_mean is None or self.state.nihss_std is None:
                raise RuntimeError(
                    "Preprocessor not fit() yet (missing NIHSS scaling params)."
                )
            y = (y - float(self.state.nihss_mean)) / float(self.state.nihss_std)

        return y.astype(np.float32, copy=False)

    # ---------- internals ----------
    def _transform_features_no_standardize(self, df: pd.DataFrame) -> np.ndarray:
        cfg = self.state.cfg

        # Ensure all required columns exist (fail early, clear error)
        missing = [c for c in cfg.feature_cols if c not in df.columns]
        if missing:
            raise KeyError(f"Missing feature columns in df: {missing}")

        cols = list(cfg.feature_cols)
        x = df.loc[:, cols].to_numpy(dtype=np.float32)

        # log1p counts (handles zeros safely)
        if cfg.count_cols:
            idx = [cfg.feature_cols.index(c) for c in cfg.count_cols]
            x[:, idx] = np.log1p(x[:, idx])

        # ordinal cols are left numeric; standardization (if enabled) happens later
        return x


# -----------------------------
# Dataset
# -----------------------------
class SVDTabularDataset(Dataset):
    """SVD marker dataset for pytorch.

    Returns:
      x: features (autoencoder input + reconstruction target)
      y: NIHSS (auxiliary classification target)

    """

    def __init__(self, df: pd.DataFrame, preproc: TabularAEPreprocessor):  # noqa: D107
        self.x = preproc.transform_features(df)
        self.y = preproc.transform_nihss(df)

    def __len__(self) -> int:  # noqa: D105
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:  # noqa: D105
        x = torch.from_numpy(self.x[idx])
        y = torch.tensor(self.y[idx], dtype=torch.float32)
        return x, y
