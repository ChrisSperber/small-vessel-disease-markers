"""Utility objects for small vessel disease marker project."""

from dataclasses import dataclass, fields
from enum import Enum


@dataclass(frozen=True)
class ProspSourceCols:
    """Source data column names."""

    ID: str = "ID-Nummer"
    INSEL_PID: str = "PID insel"
    MRI_DATE: str = "MRI date (24 h)"
    LACUNES: str = "Nr. Lacunes >3mm, <20mm"
    CMB_LOBAR: str = "Nr CMB lobar"  # CMB = cerebral microbleeds
    CMB_CENTRAL: str = "Nr CMB central"
    CMB_INFRATENTORIAL: str = "Nr CMB infratentorial"
    SUPERFICIAL_SIDEROSIS: str = "superficial siderosis"
    EPVS_CS: str = (
        "centrum semiovale EPVS (dobal score 0-4)"  # EPVS = enlarged perivascular spaces
    )
    EPVS_BG: str = "basal ganglia EPVS (doubal score 0-4)"
    WMH_PV: str = (
        "Periventricular WMH Fazekas"  # WMH = white matter hyperintensities, 0-3 rating
    )
    WMH_DEEP: str = "Deep WHM Fazekas"

    @classmethod
    def list_colnames(cls) -> list:
        """Get all column names.

        Returns:
            list: All column names.

        """
        return [getattr(cls, f.name) for f in fields(cls)]


@dataclass(frozen=True)
class Cols:
    """Cleaned data column names for repo-internal use."""

    ID: str = "ID"
    INSEL_PID: str = "PID"
    AGE: str = "Age"
    SEX: str = "Sex"
    MRI_DATE: str = "MRI_date"
    PRESTROKE_MRS: str = "PrestrokeMRS"
    FOLLOWUP_MRS: str = "mRS_3mon"
    FOLLOWUP_RESTROKE: str = "mRS_prestroke"
    NIHSS_ADMISSION: str = "NIH_admission"
    NIHSS_24H: str = "NIHSS_24h"
    NIHSS_24H_BINARY_GT4: str = "NIHSS_24h_binary_gt4"
    INTRAVENOUS_THROMBOLYSIS: str = "intervention_ivt"
    MECHANICAL_THROMBECTOMY: str = "intervention_mt"
    STROKE_TOAST: str = "EtiologyTOAST"
    MED_HIST_TIA: str = "MedHistTIA"  # Transient Ischemic Attack
    MED_HIST_HYPERTENSION: str = "MedHistHypertension"
    MED_HIST_DIABETES: str = "MedHistDiabetes"
    MED_HIST_HYPERLIPID: str = "MedHistHyperlipidemia"
    MED_HIST_SMOKING: str = "MedHistSmoking"
    MED_HIST_ATRIALFIBR: str = "MedHistAtrialFibr"
    MED_HIST_CHD: str = "MedHistCHD"  # Coronoary Heart Disease
    LACUNES: str = "Lacunes_Nr"
    CMB_LOBAR: str = "CMB_lobar_Nr"
    CMB_CENTRAL: str = "CMB_central_Nr"
    CMB_INFRATENTORIAL: str = "CMB_infratent_Nr"
    SUPERFICIAL_SIDEROSIS: str = "superf_siderosis"
    EPVS_CS: str = "EPVS_centrum_semiovale"
    EPVS_BG: str = "EPVS_basal_ganglia"
    EPVS_CS_COUNT: str = "EPVS_centrum_semiovale_Nr"
    EPVS_BG_COUNT: str = "EPVS_basal_ganglia_Nr"
    WMH_PV: str = "WMH_periventricular"
    WMH_DEEP: str = "WHM_deep"
    CMB_TOTAL: str = "CMB_total_Nr"
    LACUNES_BINARY: str = "Lacunes_binary"
    CMB_BINARY: str = "CMB_binary"
    WMH_BINARY: str = "WMH_binary"
    EPVS_BINARY: str = "EPVS_binary"
    SVD_BURDEN_SCORE: str = "SVD_burden_score"


@dataclass(frozen=True)
class RetroSourceCols:
    """Source data column names."""

    ID: str = "CaseID"
    PRESTROKE_MRS: str = "PrestrokedisabilityRankinTM"
    FOLLOWUP_MRS: str = "threeMonmRS"
    FOLLOWUP_RESTROKE: str = "x3MStroke"
    NIHSS_ADMISSION: str = "NIHonadmission"
    NIHSS_24H: str = "NIHSStwentyfourh"
    INTRAVENOUS_THROMBOLYSIS: str = "IVTwithrtPA"
    MECHANICAL_THROMBECTOMY: str = "mt"
    LACUNES: str = "LacunestotalFW"
    CMB_TOTAL: str = "MicrobleedingFW"
    EPVS_CS_COUNT: str = "CentrumsemiovaleEPVSFW"
    EPVS_BG_COUNT: str = "BasalgangliaEPVSFW"
    WMH_PV: str = "PeriventricularWMHFazekasFW"
    WMH_DEEP: str = "DeepWMHFazekasFW"
    STROKE_TOAST: str = "EtiologyTOAST"
    MED_HIST_TIA: str = "MedHistTIA"  # Transient Ischemic Attack
    MED_HIST_HYPERTENSION: str = "MedHistHypertension"
    MED_HIST_DIABETES: str = "MedHistDiabetes"
    MED_HIST_HYPERLIPID: str = "MedHistHyperlipidemia"
    MED_HIST_SMOKING: str = "MedHistSmoking"
    MED_HIST_ATRIALFIBR: str = "MedHistAtrialFibr"
    MED_HIST_CHD: str = "MedHistCHD"  # Coronoary Heart Disease

    @classmethod
    def list_colnames(cls) -> list:
        """Get all column names.

        Returns:
            list: All column names.

        """
        return [getattr(cls, f.name) for f in fields(cls)]


class BooleanDataString(Enum):
    """Enum to define yes/no data strings."""

    YES = "yes"
    NO = "no"
