"""Load and clean small vessel disease data of the prospective dataset.

Variables derived from the source SVD measures, like the SVD burden score, are added.

Requirements:
    - The original excel file for prospective data is located at PROSPECTIVE_SAMPLE_XLS

Outputs:
    - Cleaned and extended data CSV at PROSPECTIVE_SAMPLE_CLEAN_CSV (not inside the repo)
"""

# %%

import pandas as pd

from svd_marker_tools.config import (
    MISSING_PLACEHOLDER,
    PROSPECTIVE_SAMPLE_CLEAN_CSV,
    PROSPECTIVE_SAMPLE_XLS,
)
from svd_marker_tools.utils import ProspCols, SourceCols

SOURCE_DATA_MISSING_PLACEHOLDER = -999

# %%
# load and clean data
source_data_df = pd.read_excel(PROSPECTIVE_SAMPLE_XLS)
relevant_cols_list = SourceCols.list_colnames()
data_df = source_data_df[relevant_cols_list].copy()

# remove rows with missing data
data_df = data_df[data_df[SourceCols.LACUNES] != SOURCE_DATA_MISSING_PLACEHOLDER]
# for one row, missing values were not coded
data_df = data_df[data_df[SourceCols.WMH_PV].notna()]

colname_mapping = {
    SourceCols.ID: ProspCols.ID,
    SourceCols.INSEL_PID: ProspCols.INSEL_PID,
    SourceCols.MRI_DATE: ProspCols.MRI_DATE,
    SourceCols.LACUNES: ProspCols.LACUNES,
    SourceCols.CMB_LOBAR: ProspCols.CMB_LOBAR,
    SourceCols.CMB_CENTRAL: ProspCols.CMB_CENTRAL,
    SourceCols.CMB_INFRATENTORIAL: ProspCols.CMB_INFRATENTORIAL,
    SourceCols.SUPERFICIAL_SIDEROSIS: ProspCols.SUPERFICIAL_SIDEROSIS,
    SourceCols.EPVS_CS: ProspCols.EPVS_CS,
    SourceCols.EPVS_BG: ProspCols.EPVS_BG,
    SourceCols.WMH_PV: ProspCols.WMH_PV,
    SourceCols.WMH_DEEP: ProspCols.WMH_DEEP,
}

data_df = data_df.rename(columns=colname_mapping)
data_df = data_df.fillna(MISSING_PLACEHOLDER)

# convert date to simple string
col_dt = pd.to_datetime(
    data_df[ProspCols.MRI_DATE].where(
        data_df[ProspCols.MRI_DATE] != MISSING_PLACEHOLDER
    ),
    errors="coerce",
)
# format the datetime values; keep placeholder untouched
data_df[ProspCols.MRI_DATE] = col_dt.dt.strftime("%Y_%m_%d").fillna(MISSING_PLACEHOLDER)

# set all numbers to integer
data_df = data_df.astype({col: "int" for col in data_df.select_dtypes("float").columns})
cols_to_set_to_int = [
    ProspCols.EPVS_BG,
    ProspCols.EPVS_CS,
    ProspCols.WMH_DEEP,
    ProspCols.WMH_PV,
]
data_df[cols_to_set_to_int] = data_df[cols_to_set_to_int].astype(int)

# %%
# compute derived scores
# total microbleeds
data_df[ProspCols.CMB_TOTAL] = data_df[
    [ProspCols.CMB_CENTRAL, ProspCols.CMB_INFRATENTORIAL, ProspCols.CMB_LOBAR]
].sum(axis=1)

# binary scores
# cutoffs from Staals et al 2014, EPVS BG are the criterion for SVD burden
CUTOFF_EPVS_BG_BINARY = 2
CUTOFF_WMH_DEEP_BINARY = 2
WMH_PV_BINARY_VAL = 3

data_df[ProspCols.LACUNES_BINARY] = (data_df[ProspCols.LACUNES] > 0).astype(int)
data_df[ProspCols.CMB_BINARY] = (data_df[ProspCols.CMB_TOTAL] > 0).astype(int)
data_df[ProspCols.EPVS_BINARY] = (
    data_df[ProspCols.EPVS_BG] >= CUTOFF_EPVS_BG_BINARY
).astype(int)
data_df[ProspCols.WMH_BINARY] = (
    (data_df[ProspCols.WMH_PV] == WMH_PV_BINARY_VAL)
    | (data_df[ProspCols.WMH_DEEP] >= CUTOFF_WMH_DEEP_BINARY)
).astype(int)

# SVD burden score
data_df[ProspCols.SVD_BURDEN_SCORE] = data_df[
    [
        ProspCols.LACUNES_BINARY,
        ProspCols.CMB_BINARY,
        ProspCols.EPVS_BINARY,
        ProspCols.WMH_BINARY,
    ]
].sum(axis=1)

# %%
# store csv
data_df.to_csv(PROSPECTIVE_SAMPLE_CLEAN_CSV, index=False, sep=";")

# %%
