"""
Tests for core.units module.
"""

import pytest
from materials.core.units import (
    LengthUnit,
    StressUnit,
    ForceUnit,
    MomentUnit,
    LENGTH_TO_MM,
    STRESS_TO_MPA,
    FORCE_TO_KN,
    MOMENT_TO_KNM,
    STANDARD_LENGTH_UNIT,
    STANDARD_STRESS_UNIT,
    STANDARD_FORCE_UNIT,
    STANDARD_MOMENT_UNIT,
    to_mm,
    to_mpa,
    to_kn,
    to_knm,
    from_mm,
    from_mpa,
    from_kn,
    from_knm,
)


# --- Enum value tests (serialisation/display strings) ---

@pytest.mark.parametrize("member,expected", [
    (LengthUnit.MM, "mm"),
    (LengthUnit.M, "m"),
    (LengthUnit.CM, "cm"),
    (LengthUnit.IN, "in"),
    (LengthUnit.FT, "ft"),
])
def test_length_unit_values(member, expected):
    assert member.value == expected


@pytest.mark.parametrize("member,expected", [
    (StressUnit.MPA, "MPa"),
    (StressUnit.PA, "Pa"),
    (StressUnit.KPA, "kPa"),
    (StressUnit.GPA, "GPa"),
    (StressUnit.PSI, "psi"),
    (StressUnit.KSI, "ksi"),
])
def test_stress_unit_values(member, expected):
    assert member.value == expected


@pytest.mark.parametrize("member,expected", [
    (ForceUnit.N, "N"),
    (ForceUnit.KN, "kN"),
    (ForceUnit.MN, "MN"),
    (ForceUnit.LBF, "lbf"),
    (ForceUnit.KIPS, "kips"),
])
def test_force_unit_values(member, expected):
    assert member.value == expected


@pytest.mark.parametrize("member,expected", [
    (MomentUnit.NMM, "N·mm"),
    (MomentUnit.NM, "N·m"),
    (MomentUnit.KNM, "kN·m"),
])
def test_moment_unit_values(member, expected):
    assert member.value == expected


# --- Conversion factor tests ---

@pytest.mark.parametrize("unit,expected", [
    (LengthUnit.MM, 1.0),
    (LengthUnit.CM, 10.0),
    (LengthUnit.M, 1000.0),
    (LengthUnit.IN, 25.4),
    (LengthUnit.FT, 304.8),
])
def test_length_conversion_factors(unit, expected):
    assert LENGTH_TO_MM[unit] == pytest.approx(expected)


@pytest.mark.parametrize("unit,expected", [
    (StressUnit.MPA, 1.0),
    (StressUnit.PA, 1e-6),
    (StressUnit.KPA, 1e-3),
    (StressUnit.GPA, 1e3),
])
def test_stress_conversion_factors(unit, expected):
    assert STRESS_TO_MPA[unit] == pytest.approx(expected)


@pytest.mark.parametrize("unit,expected", [
    (ForceUnit.N, 1e-3),
    (ForceUnit.KN, 1.0),
    (ForceUnit.MN, 1e3),
])
def test_force_conversion_factors(unit, expected):
    assert FORCE_TO_KN[unit] == pytest.approx(expected)


@pytest.mark.parametrize("unit,expected", [
    (MomentUnit.NMM, 1e-6),
    (MomentUnit.NM, 1e-3),
    (MomentUnit.KNM, 1.0),
])
def test_moment_conversion_factors(unit, expected):
    assert MOMENT_TO_KNM[unit] == pytest.approx(expected)


# --- Relationship tests (derived from canonical constants, not hardcoded decimals) ---

def test_ksi_is_1000_psi():
    assert STRESS_TO_MPA[StressUnit.KSI] == pytest.approx(
        1000.0 * STRESS_TO_MPA[StressUnit.PSI]
    )


def test_kips_is_1000_lbf():
    assert FORCE_TO_KN[ForceUnit.KIPS] == pytest.approx(
        1000.0 * FORCE_TO_KN[ForceUnit.LBF]
    )


def test_ft_is_12_in():
    assert LENGTH_TO_MM[LengthUnit.FT] == pytest.approx(
        12.0 * LENGTH_TO_MM[LengthUnit.IN]
    )


def test_gpa_is_1000_mpa():
    assert STRESS_TO_MPA[StressUnit.GPA] == pytest.approx(
        1000.0 * STRESS_TO_MPA[StressUnit.MPA]
    )


def test_mn_is_1000_kn():
    assert FORCE_TO_KN[ForceUnit.MN] == pytest.approx(
        1000.0 * FORCE_TO_KN[ForceUnit.KN]
    )


# --- Mapping completeness tests ---

def test_length_map_covers_all_units():
    assert set(LENGTH_TO_MM.keys()) == set(LengthUnit)


def test_stress_map_covers_all_units():
    assert set(STRESS_TO_MPA.keys()) == set(StressUnit)


def test_force_map_covers_all_units():
    assert set(FORCE_TO_KN.keys()) == set(ForceUnit)


def test_moment_map_covers_all_units():
    assert set(MOMENT_TO_KNM.keys()) == set(MomentUnit)


# --- All factors positive ---

@pytest.mark.parametrize("mapping", [LENGTH_TO_MM, STRESS_TO_MPA, FORCE_TO_KN, MOMENT_TO_KNM])
def test_all_factors_positive(mapping):
    assert all(v > 0 for v in mapping.values())


# --- Mapping immutability ---

def test_mappings_are_immutable():
    with pytest.raises(TypeError):
        LENGTH_TO_MM[LengthUnit.MM] = 2.0  # type: ignore[index]
    with pytest.raises(TypeError):
        STRESS_TO_MPA[StressUnit.MPA] = 2.0  # type: ignore[index]
    with pytest.raises(TypeError):
        FORCE_TO_KN[ForceUnit.KN] = 2.0  # type: ignore[index]
    with pytest.raises(TypeError):
        MOMENT_TO_KNM[MomentUnit.KNM] = 2.0  # type: ignore[index]


# --- Standard unit runtime constants ---

def test_standard_unit_constants():
    assert STANDARD_LENGTH_UNIT is LengthUnit.MM
    assert STANDARD_STRESS_UNIT is StressUnit.MPA
    assert STANDARD_FORCE_UNIT is ForceUnit.KN
    assert STANDARD_MOMENT_UNIT is MomentUnit.KNM


# --- Helper function tests (compare against mappings, not hardcoded values) ---

@pytest.mark.parametrize("x,unit", [
    (1.0, LengthUnit.M),
    (12.0, LengthUnit.IN),
    (2.5, LengthUnit.FT),
    (100.0, LengthUnit.CM),
])
def test_to_mm_matches_mapping(x, unit):
    assert to_mm(x, unit) == pytest.approx(x * LENGTH_TO_MM[unit])


@pytest.mark.parametrize("x,unit", [
    (1.0, StressUnit.GPA),
    (1000.0, StressUnit.PSI),
    (10.0, StressUnit.KSI),
])
def test_to_mpa_matches_mapping(x, unit):
    assert to_mpa(x, unit) == pytest.approx(x * STRESS_TO_MPA[unit])


@pytest.mark.parametrize("x,unit", [
    (1000.0, ForceUnit.N),
    (1.0, ForceUnit.KIPS),
    (5.0, ForceUnit.LBF),
])
def test_to_kn_matches_mapping(x, unit):
    assert to_kn(x, unit) == pytest.approx(x * FORCE_TO_KN[unit])


@pytest.mark.parametrize("x,unit", [
    (1e6, MomentUnit.NMM),
    (1000.0, MomentUnit.NM),
    (1.0, MomentUnit.KNM),
])
def test_to_knm_matches_mapping(x, unit):
    assert to_knm(x, unit) == pytest.approx(x * MOMENT_TO_KNM[unit])


# --- String coercion tests ---

def test_to_mm_accepts_string():
    assert to_mm(1.0, "m") == 1000.0


def test_to_mpa_accepts_string():
    assert to_mpa(1.0, "GPa") == 1000.0


def test_to_kn_accepts_string():
    assert to_kn(1000.0, "N") == 1.0


def test_to_knm_accepts_string():
    assert to_knm(1.0, "kN·m") == 1.0


def test_to_mm_rejects_invalid_string():
    with pytest.raises(ValueError):
        to_mm(1.0, "furlongs")


def test_to_mpa_rejects_invalid_string():
    with pytest.raises(ValueError):
        to_mpa(1.0, "bananas")


# --- Reverse helper function tests (from standard units) ---

@pytest.mark.parametrize("x,unit", [
    (1000.0, LengthUnit.M),
    (304.8, LengthUnit.IN),
    (25.4, LengthUnit.CM),
    (1.0, LengthUnit.MM),
])
def test_from_mm_matches_mapping(x, unit):
    assert from_mm(x, unit) == pytest.approx(x / LENGTH_TO_MM[unit])


@pytest.mark.parametrize("x,unit", [
    (1000.0, StressUnit.GPA),
    (1.0, StressUnit.MPA),
    (100.0, StressUnit.KPA),
])
def test_from_mpa_matches_mapping(x, unit):
    assert from_mpa(x, unit) == pytest.approx(x / STRESS_TO_MPA[unit])


@pytest.mark.parametrize("x,unit", [
    (1.0, ForceUnit.N),
    (5.0, ForceUnit.LBF),
    (1.0, ForceUnit.KN),
])
def test_from_kn_matches_mapping(x, unit):
    assert from_kn(x, unit) == pytest.approx(x / FORCE_TO_KN[unit])


@pytest.mark.parametrize("x,unit", [
    (1.0, MomentUnit.NMM),
    (1.0, MomentUnit.NM),
    (1.0, MomentUnit.KNM),
])
def test_from_knm_matches_mapping(x, unit):
    assert from_knm(x, unit) == pytest.approx(x / MOMENT_TO_KNM[unit])


# --- Round-trip tests (to → from and back) ---

@pytest.mark.parametrize("value,unit", [
    (2.5, LengthUnit.M),
    (12.0, LengthUnit.IN),
    (3.0, LengthUnit.FT),
])
def test_length_round_trip(value, unit):
    assert from_mm(to_mm(value, unit), unit) == pytest.approx(value)


@pytest.mark.parametrize("value,unit", [
    (30.0, StressUnit.GPA),
    (1000.0, StressUnit.PSI),
])
def test_stress_round_trip(value, unit):
    assert from_mpa(to_mpa(value, unit), unit) == pytest.approx(value)


@pytest.mark.parametrize("value,unit", [
    (500.0, ForceUnit.N),
    (2.0, ForceUnit.KIPS),
])
def test_force_round_trip(value, unit):
    assert from_kn(to_kn(value, unit), unit) == pytest.approx(value)


@pytest.mark.parametrize("value,unit", [
    (1e6, MomentUnit.NMM),
    (50.0, MomentUnit.NM),
])
def test_moment_round_trip(value, unit):
    assert from_knm(to_knm(value, unit), unit) == pytest.approx(value)


# --- Reverse helper string coercion ---

def test_from_mm_accepts_string():
    assert from_mm(1000.0, "m") == 1.0


def test_from_mpa_accepts_string():
    assert from_mpa(1000.0, "GPa") == 1.0


def test_from_kn_accepts_string():
    assert from_kn(1.0, "N") == 1000.0


def test_from_knm_accepts_string():
    assert from_knm(1.0, "N·m") == 1000.0


def test_from_mm_rejects_invalid_string():
    with pytest.raises(ValueError):
        from_mm(1.0, "furlongs")


def test_from_kn_rejects_invalid_string():
    with pytest.raises(ValueError):
        from_kn(1.0, "bananas")
