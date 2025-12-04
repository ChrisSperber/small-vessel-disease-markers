"""Utility objects for small vessel disease marker project."""

from dataclasses import dataclass, fields


@dataclass(frozen=True)
class SourceCols:
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
class ProspCols:
    """Cleaned data column names for repo-internal use."""

    ID: str = "ID"
    INSEL_PID: str = "PID"
    MRI_DATE: str = "MRI_date"
    LACUNES: str = "Lacunes_Nr"
    CMB_LOBAR: str = "CMB_lobar_Nr"
    CMB_CENTRAL: str = "CMB_central_Nr"
    CMB_INFRATENTORIAL: str = "CMB_infratent_Nr"
    SUPERFICIAL_SIDEROSIS: str = "superf_siderosis"
    EPVS_CS: str = "EPVS_centrum_semiovale"
    EPVS_BG: str = "EPVS_basal_ganglia"
    WMH_PV: str = "WMH_periventricular"
    WMH_DEEP: str = "WHM_deep"
    CMB_TOTAL: str = "CMB_total_Nr"
    LACUNES_BINARY: str = "Lacunes_binary"
    CMB_BINARY: str = "CMB_binary"
    WMH_BINARY: str = "WMH_binary"
    EPVS_BINARY: str = "EPVS_binary"
    SVD_BURDEN_SCORE: str = "SVD_burden_score"
