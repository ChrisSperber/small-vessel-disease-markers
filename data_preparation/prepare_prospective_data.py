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
from svd_marker_tools.utils import Cols, ProspSourceCols

SOURCE_DATA_MISSING_PLACEHOLDER = -999

# %%
# load and clean data
source_data_df = pd.read_excel(PROSPECTIVE_SAMPLE_XLS)
relevant_cols_list = ProspSourceCols.list_colnames()
data_df = source_data_df[relevant_cols_list].copy()

# remove rows with missing data
data_df = data_df[data_df[ProspSourceCols.LACUNES] != SOURCE_DATA_MISSING_PLACEHOLDER]
# for one row, missing values were not coded
data_df = data_df[data_df[ProspSourceCols.WMH_PV].notna()]

colname_mapping = {
    ProspSourceCols.ID: Cols.ID,
    ProspSourceCols.INSEL_PID: Cols.INSEL_PID,
    ProspSourceCols.MRI_DATE: Cols.MRI_DATE,
    ProspSourceCols.LACUNES: Cols.LACUNES,
    ProspSourceCols.CMB_LOBAR: Cols.CMB_LOBAR,
    ProspSourceCols.CMB_CENTRAL: Cols.CMB_CENTRAL,
    ProspSourceCols.CMB_INFRATENTORIAL: Cols.CMB_INFRATENTORIAL,
    ProspSourceCols.SUPERFICIAL_SIDEROSIS: Cols.SUPERFICIAL_SIDEROSIS,
    ProspSourceCols.EPVS_CS: Cols.EPVS_CS,
    ProspSourceCols.EPVS_BG: Cols.EPVS_BG,
    ProspSourceCols.WMH_PV: Cols.WMH_PV,
    ProspSourceCols.WMH_DEEP: Cols.WMH_DEEP,
}

data_df = data_df.rename(columns=colname_mapping)
data_df = data_df.fillna(MISSING_PLACEHOLDER)

# convert date to simple string
col_dt = pd.to_datetime(
    data_df[Cols.MRI_DATE].where(data_df[Cols.MRI_DATE] != MISSING_PLACEHOLDER),
    errors="coerce",
)
# format the datetime values; keep placeholder untouched
data_df[Cols.MRI_DATE] = col_dt.dt.strftime("%Y_%m_%d").fillna(MISSING_PLACEHOLDER)

# set all numbers to integer
cols_to_set_to_int = [
    Cols.EPVS_BG,
    Cols.EPVS_CS,
    Cols.WMH_DEEP,
    Cols.WMH_PV,
]
data_df[cols_to_set_to_int] = data_df[cols_to_set_to_int].astype(int)

# %%
# compute derived scores
# total microbleeds
data_df[Cols.CMB_TOTAL] = data_df[
    [Cols.CMB_CENTRAL, Cols.CMB_INFRATENTORIAL, Cols.CMB_LOBAR]
].sum(axis=1)

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

# %%
# store csv
data_df.to_csv(PROSPECTIVE_SAMPLE_CLEAN_CSV, index=False, sep=";")


# %%
