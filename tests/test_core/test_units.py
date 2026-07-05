"""
Tests for core.units module.
"""

import pytest

from section_design_checks.core.units import (
    FORCE_TO_KN,
    LENGTH_TO_MM,
    MOMENT_TO_KNM,
    STANDARD_FORCE_UNIT,
    STANDARD_LENGTH_UNIT,
    STANDARD_MOMENT_UNIT,
    STANDARD_STRESS_UNIT,
    STRESS_TO_MPA,
    ForceUnit,
    LengthUnit,
    MomentUnit,
    StressUnit,
    from_kn,
    from_knm,
    from_mm,
    from_mpa,
    to_kn,
    to_knm,
    to_mm,
    to_mpa,
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
    """Length enum values should match canonical lowercase unit strings."""
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
    """Stress enum values should match canonical stress-unit labels."""
    assert member.value == expected


@pytest.mark.parametrize("member,expected", [
    (ForceUnit.N, "N"),
    (ForceUnit.KN, "kN"),
    (ForceUnit.MN, "MN"),
    (ForceUnit.LBF, "lbf"),
    (ForceUnit.KIPS, "kips"),
])
def test_force_unit_values(member, expected):
    """Force enum values should match canonical force-unit labels."""
    assert member.value == expected


@pytest.mark.parametrize("member,expected", [
    (MomentUnit.NMM, "N·mm"),
    (MomentUnit.NM, "N·m"),
    (MomentUnit.KNM, "kN·m"),
])
def test_moment_unit_values(member, expected):
    """Moment enum values should match canonical moment-unit labels."""
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
    """Length conversion table entries should equal expected factors to mm."""
    assert LENGTH_TO_MM[unit] == pytest.approx(expected)


@pytest.mark.parametrize("unit,expected", [
    (StressUnit.MPA, 1.0),
    (StressUnit.PA, 1e-6),
    (StressUnit.KPA, 1e-3),
    (StressUnit.GPA, 1e3),
])
def test_stress_conversion_factors(unit, expected):
    """Stress conversion table entries should equal expected factors to MPa."""
    assert STRESS_TO_MPA[unit] == pytest.approx(expected)


@pytest.mark.parametrize("unit,expected", [
    (ForceUnit.N, 1e-3),
    (ForceUnit.KN, 1.0),
    (ForceUnit.MN, 1e3),
])
def test_force_conversion_factors(unit, expected):
    """Force conversion table entries should equal expected factors to kN."""
    assert FORCE_TO_KN[unit] == pytest.approx(expected)


@pytest.mark.parametrize("unit,expected", [
    (MomentUnit.NMM, 1e-6),
    (MomentUnit.NM, 1e-3),
    (MomentUnit.KNM, 1.0),
])
def test_moment_conversion_factors(unit, expected):
    """Moment conversion table entries should equal expected factors to kN·m."""
    assert MOMENT_TO_KNM[unit] == pytest.approx(expected)


# --- Relationship tests (derived from canonical constants, not hardcoded decimals) ---

def test_ksi_is_1000_psi():
    """Derived stress factors should preserve ksi = 1000 psi."""
    assert STRESS_TO_MPA[StressUnit.KSI] == pytest.approx(
        1000.0 * STRESS_TO_MPA[StressUnit.PSI]
    )


def test_kips_is_1000_lbf():
    """Derived force factors should preserve kips = 1000 lbf."""
    assert FORCE_TO_KN[ForceUnit.KIPS] == pytest.approx(
        1000.0 * FORCE_TO_KN[ForceUnit.LBF]
    )


def test_ft_is_12_in():
    """Derived length factors should preserve ft = 12 in."""
    assert LENGTH_TO_MM[LengthUnit.FT] == pytest.approx(
        12.0 * LENGTH_TO_MM[LengthUnit.IN]
    )


def test_gpa_is_1000_mpa():
    """Derived stress factors should preserve GPa = 1000 MPa."""
    assert STRESS_TO_MPA[StressUnit.GPA] == pytest.approx(
        1000.0 * STRESS_TO_MPA[StressUnit.MPA]
    )


def test_mn_is_1000_kn():
    """Derived force factors should preserve MN = 1000 kN."""
    assert FORCE_TO_KN[ForceUnit.MN] == pytest.approx(
        1000.0 * FORCE_TO_KN[ForceUnit.KN]
    )


# --- Mapping completeness tests ---

def test_length_map_covers_all_units():
    """Length mapping should include every declared length unit."""
    assert set(LENGTH_TO_MM.keys()) == set(LengthUnit)


def test_stress_map_covers_all_units():
    """Stress mapping should include every declared stress unit."""
    assert set(STRESS_TO_MPA.keys()) == set(StressUnit)


def test_force_map_covers_all_units():
    """Force mapping should include every declared force unit."""
    assert set(FORCE_TO_KN.keys()) == set(ForceUnit)


def test_moment_map_covers_all_units():
    """Moment mapping should include every declared moment unit."""
    assert set(MOMENT_TO_KNM.keys()) == set(MomentUnit)


# --- All factors positive ---

@pytest.mark.parametrize("mapping", [LENGTH_TO_MM, STRESS_TO_MPA, FORCE_TO_KN, MOMENT_TO_KNM])
def test_all_factors_positive(mapping):
    """All conversion factors should be strictly positive."""
    assert all(v > 0 for v in mapping.values())


# --- Mapping immutability ---

def test_mappings_are_immutable():
    """Conversion-factor mappings should be read-only."""
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
    """Runtime standard-unit constants should point to expected enum members."""
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
    """`to_mm` should apply the factor from `LENGTH_TO_MM`."""
    assert to_mm(x, unit) == pytest.approx(x * LENGTH_TO_MM[unit])


@pytest.mark.parametrize("x,unit", [
    (1.0, StressUnit.GPA),
    (1000.0, StressUnit.PSI),
    (10.0, StressUnit.KSI),
])
def test_to_mpa_matches_mapping(x, unit):
    """`to_mpa` should apply the factor from `STRESS_TO_MPA`."""
    assert to_mpa(x, unit) == pytest.approx(x * STRESS_TO_MPA[unit])


@pytest.mark.parametrize("x,unit", [
    (1000.0, ForceUnit.N),
    (1.0, ForceUnit.KIPS),
    (5.0, ForceUnit.LBF),
])
def test_to_kn_matches_mapping(x, unit):
    """`to_kn` should apply the factor from `FORCE_TO_KN`."""
    assert to_kn(x, unit) == pytest.approx(x * FORCE_TO_KN[unit])


@pytest.mark.parametrize("x,unit", [
    (1e6, MomentUnit.NMM),
    (1000.0, MomentUnit.NM),
    (1.0, MomentUnit.KNM),
])
def test_to_knm_matches_mapping(x, unit):
    """`to_knm` should apply the factor from `MOMENT_TO_KNM`."""
    assert to_knm(x, unit) == pytest.approx(x * MOMENT_TO_KNM[unit])


# --- String coercion tests ---

def test_to_mm_accepts_string():
    """Length helper should accept string unit tokens."""
    assert to_mm(1.0, "m") == 1000.0


def test_to_mpa_accepts_string():
    """Stress helper should accept string unit tokens."""
    assert to_mpa(1.0, "GPa") == 1000.0


def test_to_kn_accepts_string():
    """Force helper should accept string unit tokens."""
    assert to_kn(1000.0, "N") == 1.0


def test_to_knm_accepts_string():
    """Moment helper should accept string unit tokens."""
    assert to_knm(1.0, "kN·m") == 1.0


def test_to_mm_rejects_invalid_string():
    """Length helper should reject unsupported unit strings."""
    with pytest.raises(ValueError):
        to_mm(1.0, "furlongs")


def test_to_mpa_rejects_invalid_string():
    """Stress helper should reject unsupported unit strings."""
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
    """`from_mm` should invert factors from `LENGTH_TO_MM`."""
    assert from_mm(x, unit) == pytest.approx(x / LENGTH_TO_MM[unit])


@pytest.mark.parametrize("x,unit", [
    (1000.0, StressUnit.GPA),
    (1.0, StressUnit.MPA),
    (100.0, StressUnit.KPA),
])
def test_from_mpa_matches_mapping(x, unit):
    """`from_mpa` should invert factors from `STRESS_TO_MPA`."""
    assert from_mpa(x, unit) == pytest.approx(x / STRESS_TO_MPA[unit])


@pytest.mark.parametrize("x,unit", [
    (1.0, ForceUnit.N),
    (5.0, ForceUnit.LBF),
    (1.0, ForceUnit.KN),
])
def test_from_kn_matches_mapping(x, unit):
    """`from_kn` should invert factors from `FORCE_TO_KN`."""
    assert from_kn(x, unit) == pytest.approx(x / FORCE_TO_KN[unit])


@pytest.mark.parametrize("x,unit", [
    (1.0, MomentUnit.NMM),
    (1.0, MomentUnit.NM),
    (1.0, MomentUnit.KNM),
])
def test_from_knm_matches_mapping(x, unit):
    """`from_knm` should invert factors from `MOMENT_TO_KNM`."""
    assert from_knm(x, unit) == pytest.approx(x / MOMENT_TO_KNM[unit])


# --- Round-trip tests (to → from and back) ---

@pytest.mark.parametrize("value,unit", [
    (2.5, LengthUnit.M),
    (12.0, LengthUnit.IN),
    (3.0, LengthUnit.FT),
])
def test_length_round_trip(value, unit):
    """Length conversions should round-trip through standard units."""
    assert from_mm(to_mm(value, unit), unit) == pytest.approx(value)


@pytest.mark.parametrize("value,unit", [
    (30.0, StressUnit.GPA),
    (1000.0, StressUnit.PSI),
])
def test_stress_round_trip(value, unit):
    """Stress conversions should round-trip through standard units."""
    assert from_mpa(to_mpa(value, unit), unit) == pytest.approx(value)


@pytest.mark.parametrize("value,unit", [
    (500.0, ForceUnit.N),
    (2.0, ForceUnit.KIPS),
])
def test_force_round_trip(value, unit):
    """Force conversions should round-trip through standard units."""
    assert from_kn(to_kn(value, unit), unit) == pytest.approx(value)


@pytest.mark.parametrize("value,unit", [
    (1e6, MomentUnit.NMM),
    (50.0, MomentUnit.NM),
])
def test_moment_round_trip(value, unit):
    """Moment conversions should round-trip through standard units."""
    assert from_knm(to_knm(value, unit), unit) == pytest.approx(value)


# --- Reverse helper string coercion ---

def test_from_mm_accepts_string():
    """Reverse length helper should accept string unit tokens."""
    assert from_mm(1000.0, "m") == 1.0


def test_from_mpa_accepts_string():
    """Reverse stress helper should accept string unit tokens."""
    assert from_mpa(1000.0, "GPa") == 1.0


def test_from_kn_accepts_string():
    """Reverse force helper should accept string unit tokens."""
    assert from_kn(1.0, "N") == 1000.0


def test_from_knm_accepts_string():
    """Reverse moment helper should accept string unit tokens."""
    assert from_knm(1.0, "N·m") == 1000.0


def test_from_mm_rejects_invalid_string():
    """Reverse length helper should reject unsupported unit strings."""
    with pytest.raises(ValueError):
        from_mm(1.0, "furlongs")


def test_from_kn_rejects_invalid_string():
    """Reverse force helper should reject unsupported unit strings."""
    with pytest.raises(ValueError):
        from_kn(1.0, "bananas")
