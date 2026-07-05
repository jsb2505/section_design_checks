"""
Unit definitions and conversions for structural engineering.

All calculations use SI base units internally:
- Length: mm
- Force: N (kN for larger values)
- Stress: MPa (N/mm²)
- Temperature: °C
"""

from enum import Enum
from typing import Literal


class LengthUnit(str, Enum):
    """Length units."""
    MM = "mm"
    M = "m"
    CM = "cm"
    IN = "in"
    FT = "ft"


class StressUnit(str, Enum):
    """Stress/pressure units."""
    MPA = "MPa"
    PA = "Pa"
    KPA = "kPa"
    GPA = "GPa"
    PSI = "psi"
    KSI = "ksi"


class ForceUnit(str, Enum):
    """Force units."""
    N = "N"
    KN = "kN"
    MN = "MN"
    LBF = "lbf"
    KIPS = "kips"


class MomentUnit(str, Enum):
    """Moment units."""
    NMM = "N·mm"
    NM = "N·m"
    KNM = "kN·m"


# Standard units used throughout the library
StandardLengthUnit = Literal["mm"]
StandardStressUnit = Literal["MPa"]
StandardForceUnit = Literal["kN"]
StandardMomentUnit = Literal["kN·m"]


# Conversion factors to standard units
LENGTH_TO_MM = {
    LengthUnit.MM: 1.0,
    LengthUnit.CM: 10.0,
    LengthUnit.M: 1000.0,
    LengthUnit.IN: 25.4,
    LengthUnit.FT: 304.8,
}

STRESS_TO_MPA = {
    StressUnit.MPA: 1.0,
    StressUnit.PA: 1e-6,
    StressUnit.KPA: 1e-3,
    StressUnit.GPA: 1e3,
    StressUnit.PSI: 0.00689476,
    StressUnit.KSI: 6.89476,
}

FORCE_TO_KN = {
    ForceUnit.N: 1e-3,
    ForceUnit.KN: 1.0,
    ForceUnit.MN: 1e3,
    ForceUnit.LBF: 4.44822e-3,
    ForceUnit.KIPS: 4.44822,
}
