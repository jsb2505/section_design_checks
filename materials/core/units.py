"""
Unit definitions and conversions for structural engineering.

All calculations use SI base units internally:
- Length: mm
- Force: N (kN for larger values)
- Stress: MPa (N/mm²)
- Moment: kN·m
- Temperature: °C
"""

from collections.abc import Mapping
from enum import Enum
from types import MappingProxyType
from typing import Literal

__all__ = [
    "LengthUnit",
    "StressUnit",
    "ForceUnit",
    "MomentUnit",
    "StandardLengthUnit",
    "StandardStressUnit",
    "StandardForceUnit",
    "StandardMomentUnit",
    "STANDARD_LENGTH_UNIT",
    "STANDARD_STRESS_UNIT",
    "STANDARD_FORCE_UNIT",
    "STANDARD_MOMENT_UNIT",
    "LENGTH_TO_MM",
    "STRESS_TO_MPA",
    "FORCE_TO_KN",
    "MOMENT_TO_KNM",
    "to_mm",
    "to_mpa",
    "to_kn",
    "to_knm",
    "from_mm",
    "from_mpa",
    "from_kn",
    "from_knm",
]


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


# Standard units used throughout the library (type-level aliases)
StandardLengthUnit = Literal["mm"]
StandardStressUnit = Literal["MPa"]
StandardForceUnit = Literal["kN"]
StandardMomentUnit = Literal["kN·m"]

# Standard units as runtime constants
STANDARD_LENGTH_UNIT: LengthUnit = LengthUnit.MM
STANDARD_STRESS_UNIT: StressUnit = StressUnit.MPA
STANDARD_FORCE_UNIT: ForceUnit = ForceUnit.KN
STANDARD_MOMENT_UNIT: MomentUnit = MomentUnit.KNM


# Canonical conversion constant: 1 psi = 6894.757293168 Pa
_PSI_TO_PA = 6894.757293168
_PSI_TO_MPA = _PSI_TO_PA * 1e-6
_KSI_TO_MPA = 1000.0 * _PSI_TO_MPA

# Canonical conversion constant: 1 lbf = 4.4482216152605 N
_LBF_TO_N = 4.4482216152605
_LBF_TO_KN = _LBF_TO_N * 1e-3
_KIPS_TO_KN = 1000.0 * _LBF_TO_KN


# Conversion factors to standard units (immutable)
LENGTH_TO_MM: Mapping[LengthUnit, float] = MappingProxyType({
    LengthUnit.MM: 1.0,
    LengthUnit.CM: 10.0,
    LengthUnit.M: 1000.0,
    LengthUnit.IN: 25.4,
    LengthUnit.FT: 304.8,
})

STRESS_TO_MPA: Mapping[StressUnit, float] = MappingProxyType({
    StressUnit.MPA: 1.0,
    StressUnit.PA: 1e-6,
    StressUnit.KPA: 1e-3,
    StressUnit.GPA: 1e3,
    StressUnit.PSI: _PSI_TO_MPA,
    StressUnit.KSI: _KSI_TO_MPA,
})

FORCE_TO_KN: Mapping[ForceUnit, float] = MappingProxyType({
    ForceUnit.N: 1e-3,
    ForceUnit.KN: 1.0,
    ForceUnit.MN: 1e3,
    ForceUnit.LBF: _LBF_TO_KN,
    ForceUnit.KIPS: _KIPS_TO_KN,
})

MOMENT_TO_KNM: Mapping[MomentUnit, float] = MappingProxyType({
    MomentUnit.NMM: 1e-6,
    MomentUnit.NM: 1e-3,
    MomentUnit.KNM: 1.0,
})


# --- Conversion helper functions ---

def to_mm(value: float, unit: LengthUnit | str) -> float:
    """Convert a length value to millimetres.

    Accepts a LengthUnit enum or its string value (e.g. ``"m"``).
    """
    unit = LengthUnit(unit)
    return value * LENGTH_TO_MM[unit]


def to_mpa(value: float, unit: StressUnit | str) -> float:
    """Convert a stress value to megapascals.

    Accepts a StressUnit enum or its string value (e.g. ``"GPa"``).
    """
    unit = StressUnit(unit)
    return value * STRESS_TO_MPA[unit]


def to_kn(value: float, unit: ForceUnit | str) -> float:
    """Convert a force value to kilonewtons.

    Accepts a ForceUnit enum or its string value (e.g. ``"N"``).
    """
    unit = ForceUnit(unit)
    return value * FORCE_TO_KN[unit]


def to_knm(value: float, unit: MomentUnit | str) -> float:
    """Convert a moment value to kilonewton-metres.

    Accepts a MomentUnit enum or its string value (e.g. ``"N·m"``).
    """
    unit = MomentUnit(unit)
    return value * MOMENT_TO_KNM[unit]


# --- Reverse conversion helper functions (from standard units) ---

def from_mm(value: float, unit: LengthUnit | str) -> float:
    """Convert a length value from millimetres to the specified unit.

    Accepts a LengthUnit enum or its string value (e.g. ``"m"``).
    """
    unit = LengthUnit(unit)
    return value / LENGTH_TO_MM[unit]


def from_mpa(value: float, unit: StressUnit | str) -> float:
    """Convert a stress value from megapascals to the specified unit.

    Accepts a StressUnit enum or its string value (e.g. ``"GPa"``).
    """
    unit = StressUnit(unit)
    return value / STRESS_TO_MPA[unit]


def from_kn(value: float, unit: ForceUnit | str) -> float:
    """Convert a force value from kilonewtons to the specified unit.

    Accepts a ForceUnit enum or its string value (e.g. ``"N"``).
    """
    unit = ForceUnit(unit)
    return value / FORCE_TO_KN[unit]


def from_knm(value: float, unit: MomentUnit | str) -> float:
    """Convert a moment value from kilonewton-metres to the specified unit.

    Accepts a MomentUnit enum or its string value (e.g. ``"N·m"``).
    """
    unit = MomentUnit(unit)
    return value / MOMENT_TO_KNM[unit]
