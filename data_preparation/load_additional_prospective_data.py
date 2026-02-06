"""Load additional data for the prospective dataset.

Further non-SVD data are loaded from an extended data file, notably the NIHSS.

Outputs:
    - csv with relevant data for validation analysis

"""

# %%
import numpy as np
import pandas as pd

from svd_marker_tools.config import (
    ADDITIONAL_PROSPECTIVE_SAMPLE_XLS,
    DATA_FOLDER,
    MISSING_PLACEHOLDER,
    MRS_CUTOFF_POOR,
    NIHSS_CUTOFF_MINOR_STROKE,
)
from svd_marker_tools.utils import Cols

RELEVANT_COLS_MAIN = [
    "record_id",
    "pat_id",
    "age",
    "biol_sex",
    "yoe",
    "med_tia",
    "med_hypert",
    "med_diab",
    "med_dysl",
    "med_smoking",
    "med_af",
    "med_chd",
    "med_renal",
    "med_comorb",
    "therapy",
    "nihss_acute",
    "nihss_24h",
    "prestroke_mrs",
]
RELEVANT_COLS_FOLLOWUP = [
    "record_id",
    "recurrent_stroke_fu3",
    "mrs_fu",
    "moca_tot_fu3",
]

COLNAME_MAP = {
    "pat_id": Cols.ID,
    "age": Cols.AGE,
    "biol_sex": Cols.SEX,
    "yoe": Cols.YEAR_OF_EDUCATION,
    "med_tia": Cols.MED_HIST_TIA,
    "med_hypert": Cols.MED_HIST_HYPERTENSION,
    "med_diab": Cols.MED_HIST_DIABETES,
    "med_dysl": Cols.MED_HIST_HYPERLIPID,
    "med_smoking": Cols.MED_HIST_SMOKING,
    "med_af": Cols.MED_HIST_ATRIALFIBR,
    "med_chd": Cols.MED_HIST_CHD,
    "med_renal": Cols.MED_HIST_RENAL,
    "nihss_acute": Cols.NIHSS_ADMISSION,
    "nihss_24h": Cols.NIHSS_24H,
    "prestroke_mrs": Cols.PRESTROKE_MRS,
    "recurrent_stroke_fu3": Cols.FOLLOWUP_RESTROKE,
    "mrs_fu": Cols.FOLLOWUP_MRS,
    "moca_tot_fu3": Cols.FOLLOWUP_MOCA,
}

# %%
# load data, sheets 1 (main data) and sheet 2 (follow up data)
data_main = pd.read_excel(ADDITIONAL_PROSPECTIVE_SAMPLE_XLS, sheet_name="baseline")
data_main = data_main[RELEVANT_COLS_MAIN]
data_followup = pd.read_excel(
    ADDITIONAL_PROSPECTIVE_SAMPLE_XLS, sheet_name="FU 3 months"
)
data_followup = data_followup[RELEVANT_COLS_FOLLOWUP]

# %%
data_df = data_main.merge(
    data_followup, how="left", on="record_id", validate="one_to_one"
)

# %%
# clean table - unify missing placeholder, format numbers
int_cols = [
    "age",
    "biol_sex",
    "med_tia",
    "med_hypert",
    "med_diab",
    "med_dysl",
    "med_smoking",
    "med_af",
    "med_chd",
    "med_renal",
    "med_comorb",
    "therapy",
    "prestroke_mrs",
    "recurrent_stroke_fu3",
    "mrs_fu",
    "moca_tot_fu3",
]

MISSING_MARKERS = [-999, "-999", -999.0]

# normalize missing markers + empty/whitespace strings
data_df[int_cols] = (
    data_df[int_cols]
    .replace(MISSING_MARKERS, pd.NA)
    .replace(r"^\s*$", pd.NA, regex=True)
)
# ensure columns are numeric
num = data_df[int_cols].apply(pd.to_numeric, errors="coerce")

# debug check: fail loudly if any non-missing values are not integer-valued
nonint = num.notna() & (num % 1 != 0)
if nonint.any().any():
    for c in nonint.columns[nonint.any(axis=0)]:
        bad = data_df.loc[nonint[c], c].unique()[:20]
        raise ValueError(f"Non-integer values in {c}: {bad}")

data_df[int_cols] = (
    data_df[int_cols].astype("Int64").astype("string").fillna(MISSING_PLACEHOLDER)
)

# rename columns
data_df = data_df.rename(columns=COLNAME_MAP)

# set missing placeholder for non int columns
data_df[Cols.NIHSS_24H] = data_df[Cols.NIHSS_24H].replace(MISSING_MARKERS, pd.NA)
data_df[Cols.YEAR_OF_EDUCATION] = data_df[Cols.YEAR_OF_EDUCATION].replace(
    MISSING_MARKERS, pd.NA
)
data_df.loc[data_df[Cols.YEAR_OF_EDUCATION].isna(), Cols.YEAR_OF_EDUCATION] = (
    MISSING_PLACEHOLDER
)

# %%
# check/derive main target variables
# NIHSS 24h was target training criterion and is central to validation

data_df = data_df[data_df[Cols.NIHSS_24H].notna()]
data_df[Cols.NIHSS_24H_BINARY_GT4] = (
    data_df[Cols.NIHSS_24H] > NIHSS_CUTOFF_MINOR_STROKE
).astype(int)

# mRS is a secondary target variable
# cutoff >1 is chosen to create less uneven groups

num = pd.to_numeric(data_df[Cols.FOLLOWUP_MRS], errors="coerce")

data_df[Cols.FOLLOWUP_MRS_BINARY_GT1] = (
    num.gt(MRS_CUTOFF_POOR)
    .astype("Int64")
    .astype(str)
    .replace("<NA>", MISSING_PLACEHOLDER)
)

# MoCa is a secondary target variable
# cutoff is derived according to Gallucci et al 2024 accounting for age and years of education
MOCA_YOE_CUTOFF = 12
MOCA_AGE_CUTOFF_LOW = 55
MOCA_AGE_CUTOFF_HIGH = 70

# re-format data types
for col in [
    Cols.FOLLOWUP_MOCA,
    Cols.YEAR_OF_EDUCATION,
    Cols.AGE,
]:
    data_df[col] = pd.to_numeric(data_df[col], errors="coerce")

conditions = [
    (data_df[Cols.YEAR_OF_EDUCATION] <= MOCA_YOE_CUTOFF)
    & (data_df[Cols.AGE] < MOCA_AGE_CUTOFF_LOW),
    (data_df[Cols.YEAR_OF_EDUCATION] <= MOCA_YOE_CUTOFF)
    & (data_df[Cols.AGE].between(MOCA_AGE_CUTOFF_LOW, MOCA_AGE_CUTOFF_HIGH)),
    (data_df[Cols.YEAR_OF_EDUCATION] <= MOCA_YOE_CUTOFF)
    & (data_df[Cols.AGE] > MOCA_AGE_CUTOFF_HIGH),
    (data_df[Cols.YEAR_OF_EDUCATION] > MOCA_YOE_CUTOFF)
    & (data_df[Cols.AGE] < MOCA_AGE_CUTOFF_LOW),
    (data_df[Cols.YEAR_OF_EDUCATION] > MOCA_YOE_CUTOFF)
    & (data_df[Cols.AGE].between(MOCA_AGE_CUTOFF_LOW, MOCA_AGE_CUTOFF_HIGH)),
    (data_df[Cols.YEAR_OF_EDUCATION] > MOCA_YOE_CUTOFF)
    & (data_df[Cols.AGE] > MOCA_AGE_CUTOFF_HIGH),
]

cutoffs = [26, 26, 22, 28, 26, 25]

cutoff = np.select(conditions, cutoffs, default=np.nan)

data_df[Cols.FOLLOWUP_MOCA_BINARY] = (data_df[Cols.FOLLOWUP_MOCA] < cutoff).astype(
    "Int64"
)

# %%
# store data
outfile = DATA_FOLDER / "additional_prospective_data.csv"
data_df.to_csv(outfile, sep=";", index=False)

# %%
