"""Load and clean small vessel disease data of the retrospective dataset.

Variables derived from the source SVD measures, like the SVD burden score, are added.

Requirements:
    - The original excel file for retrospective data is located at RETROSPECTIVE_SAMPLE_XLS

Outputs:
    - Cleaned and extended data CSV at RETROSPECTIVE_SAMPLE_CLEAN_CSV (not inside the repo)
"""

# %%

import numpy as np
import pandas as pd

from svd_marker_tools.config import (
    MISSING_PLACEHOLDER,
    NIHSS_CUTOFF_MINOR_STROKE,
    RETROSPECTIVE_SAMPLE_CLEAN_CSV,
    RETROSPECTIVE_SAMPLE_XLS,
)
from svd_marker_tools.utils import BooleanDataString, Cols, RetroSourceCols

# %%
# load and clean data
source_data_df = pd.read_excel(RETROSPECTIVE_SAMPLE_XLS)
relevant_cols_list = RetroSourceCols.list_colnames()
data_df = source_data_df[relevant_cols_list].copy()
data_df = data_df.fillna(MISSING_PLACEHOLDER)

# remove rows with missing SVD or NIHSS 24h data
required_cols = [
    RetroSourceCols.NIHSS_24H,
    RetroSourceCols.LACUNES,
    RetroSourceCols.CMB_TOTAL,
    RetroSourceCols.EPVS_BG_COUNT,
    RetroSourceCols.EPVS_CS_COUNT,
    RetroSourceCols.WMH_DEEP,
    RetroSourceCols.WMH_PV,
]
data_df = data_df[~data_df[required_cols].eq(MISSING_PLACEHOLDER).any(axis=1)]

# unify values
ivt_mapping = {
    "yes": BooleanDataString.YES.value,
    "no": BooleanDataString.NO.value,
    "started before admission": BooleanDataString.YES.value,
}
data_df[RetroSourceCols.INTRAVENOUS_THROMBOLYSIS] = data_df[
    RetroSourceCols.INTRAVENOUS_THROMBOLYSIS
].map(ivt_mapping)

mt_mapping = {
    1.0: BooleanDataString.YES.value,
    0.0: BooleanDataString.NO.value,
}
data_df[RetroSourceCols.MECHANICAL_THROMBECTOMY] = data_df[
    RetroSourceCols.MECHANICAL_THROMBECTOMY
].map(mt_mapping)

# set counts/ratings to integer
data_df = data_df.astype({col: "int" for col in data_df.select_dtypes("float").columns})
cols_to_set_to_int = [
    RetroSourceCols.EPVS_BG_COUNT,
    RetroSourceCols.EPVS_CS_COUNT,
    RetroSourceCols.WMH_DEEP,
    RetroSourceCols.WMH_PV,
    RetroSourceCols.CMB_TOTAL,
    RetroSourceCols.LACUNES,
]
data_df[cols_to_set_to_int] = data_df[cols_to_set_to_int].astype(int)

# map column names
colname_mapping = {
    RetroSourceCols.ID: Cols.ID,
    RetroSourceCols.PRESTROKE_MRS: Cols.PRESTROKE_MRS,
    RetroSourceCols.FOLLOWUP_MRS: Cols.FOLLOWUP_MRS,
    RetroSourceCols.FOLLOWUP_RESTROKE: Cols.FOLLOWUP_RESTROKE,
    RetroSourceCols.NIHSS_ADMISSION: Cols.NIHSS_ADMISSION,
    RetroSourceCols.NIHSS_24H: Cols.NIHSS_24H,
    RetroSourceCols.INTRAVENOUS_THROMBOLYSIS: Cols.INTRAVENOUS_THROMBOLYSIS,
    RetroSourceCols.MECHANICAL_THROMBECTOMY: Cols.MECHANICAL_THROMBECTOMY,
    RetroSourceCols.LACUNES: Cols.LACUNES,
    RetroSourceCols.CMB_TOTAL: Cols.CMB_TOTAL,
    RetroSourceCols.EPVS_CS_COUNT: Cols.EPVS_CS_COUNT,
    RetroSourceCols.EPVS_BG_COUNT: Cols.EPVS_BG_COUNT,
    RetroSourceCols.WMH_PV: Cols.WMH_PV,
    RetroSourceCols.WMH_DEEP: Cols.WMH_DEEP,
    RetroSourceCols.STROKE_TOAST: Cols.STROKE_TOAST,
    RetroSourceCols.MED_HIST_TIA: Cols.MED_HIST_TIA,
    RetroSourceCols.MED_HIST_HYPERTENSION: Cols.MED_HIST_HYPERTENSION,
    RetroSourceCols.MED_HIST_DIABETES: Cols.MED_HIST_DIABETES,
    RetroSourceCols.MED_HIST_HYPERLIPID: Cols.MED_HIST_HYPERLIPID,
    RetroSourceCols.MED_HIST_SMOKING: Cols.MED_HIST_SMOKING,
    RetroSourceCols.MED_HIST_CHD: Cols.MED_HIST_CHD,
    RetroSourceCols.MED_HIST_ATRIALFIBR: Cols.MED_HIST_ATRIALFIBR,
}
data_df = data_df.rename(columns=colname_mapping)

# %%
# compute derived scores
# EPVS ratings; based on Doubal et al., 2010 in Stroke
cols = [
    (Cols.EPVS_BG_COUNT, Cols.EPVS_BG),
    (Cols.EPVS_CS_COUNT, Cols.EPVS_CS),
]

choices = [0, 1, 2, 3, 4]

for count_col, score_col in cols:
    c = data_df[count_col]

    conditions = [
        c == 0,
        (c >= 1) & (c <= 10),  # noqa: PLR2004
        (c >= 11) & (c <= 20),  # noqa: PLR2004
        (c >= 21) & (c <= 40),  # noqa: PLR2004
        c > 40,  # noqa: PLR2004
    ]

    data_df[score_col] = np.select(conditions, choices, default=np.nan).astype(int)

# binary scores
# cutoffs from Staals et al 2014, EPVS BG are the criterion for SVD burden
CUTOFF_EPVS_BG_BINARY = 2
CUTOFF_WMH_DEEP_BINARY = 2
WMH_PV_BINARY_VAL = 3

data_df[Cols.LACUNES_BINARY] = (data_df[Cols.LACUNES] > 0).astype(int)
data_df[Cols.CMB_BINARY] = (data_df[Cols.CMB_TOTAL] > 0).astype(int)
data_df[Cols.EPVS_BINARY] = (data_df[Cols.EPVS_BG] >= CUTOFF_EPVS_BG_BINARY).astype(int)
data_df[Cols.WMH_BINARY] = (
    (data_df[Cols.WMH_PV] == WMH_PV_BINARY_VAL)
    | (data_df[Cols.WMH_DEEP] >= CUTOFF_WMH_DEEP_BINARY)
).astype(int)

# SVD burden score
data_df[Cols.SVD_BURDEN_SCORE] = data_df[
    [
        Cols.LACUNES_BINARY,
        Cols.CMB_BINARY,
        Cols.EPVS_BINARY,
        Cols.WMH_BINARY,
    ]
].sum(axis=1)

# Binary NIHSS > 4
data_df[Cols.NIHSS_24H_BINARY_GT4] = (
    data_df[Cols.NIHSS_24H] > NIHSS_CUTOFF_MINOR_STROKE
).astype(int)

# %%
# store csv
data_df.to_csv(RETROSPECTIVE_SAMPLE_CLEAN_CSV, index=False, sep=";")

# %%
