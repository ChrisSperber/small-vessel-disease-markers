"""Config for small vessel disease marker project."""

from pathlib import Path

DATA_FOLDER = Path(__file__).parents[3] / "Data"
PROSPECTIVE_SAMPLE_XLS = DATA_FOLDER / "source_prospective_svd_data.xlsx"
PROSPECTIVE_SAMPLE_CLEAN_CSV = DATA_FOLDER / "clean_prospective_svd_data.csv"
ADDITIONAL_PROSPECTIVE_SAMPLE_XLS = (
    DATA_FOLDER / "LB1559_CogStroke_REDCap_data_manually_cleaned.xlsx"
)
RETROSPECTIVE_SAMPLE_XLS = DATA_FOLDER / "Data_full_09_2023.xlsx"
RETROSPECTIVE_SAMPLE_CLEAN_CSV = DATA_FOLDER / "clean_retrospective_svd_data.csv"

MISSING_PLACEHOLDER = "Not_Available"

RNG_SEED = 9001

NIHSS_CUTOFF_MINOR_STROKE = 4  # ensure using >NIHSS_CUTOFF_MINOR_STROKE, not >=
