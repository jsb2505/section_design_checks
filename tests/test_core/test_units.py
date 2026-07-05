"""
Tests for core.units module.
"""

import pytest
from materials.core.units import (
    LengthUnit,
    StressUnit,
    ForceUnit,
    LENGTH_TO_MM,
    STRESS_TO_MPA,
    FORCE_TO_KN,
)


class TestUnits:
    """Tests for unit enums and conversion factors."""

    def test_length_units_exist(self):
        """Test that length units are defined."""
        assert LengthUnit.MM == "mm"
        assert LengthUnit.M == "m"
        assert LengthUnit.CM == "cm"
        assert LengthUnit.IN == "in"
        assert LengthUnit.FT == "ft"

    def test_stress_units_exist(self):
        """Test that stress units are defined."""
        assert StressUnit.MPA == "MPa"
        assert StressUnit.PA == "Pa"
        assert StressUnit.KPA == "kPa"
        assert StressUnit.GPA == "GPa"
        assert StressUnit.PSI == "psi"
        assert StressUnit.KSI == "ksi"

    def test_force_units_exist(self):
        """Test that force units are defined."""
        assert ForceUnit.N == "N"
        assert ForceUnit.KN == "kN"
        assert ForceUnit.MN == "MN"

    def test_length_conversion_factors(self):
        """Test length conversion factors."""
        assert LENGTH_TO_MM[LengthUnit.MM] == 1.0
        assert LENGTH_TO_MM[LengthUnit.M] == 1000.0
        assert LENGTH_TO_MM[LengthUnit.CM] == 10.0
        assert LENGTH_TO_MM[LengthUnit.IN] == pytest.approx(25.4)
        assert LENGTH_TO_MM[LengthUnit.FT] == pytest.approx(304.8)

    def test_stress_conversion_factors(self):
        """Test stress conversion factors."""
        assert STRESS_TO_MPA[StressUnit.MPA] == 1.0
        assert STRESS_TO_MPA[StressUnit.PA] == 1e-6
        assert STRESS_TO_MPA[StressUnit.KPA] == 1e-3
        assert STRESS_TO_MPA[StressUnit.GPA] == 1e3
        assert STRESS_TO_MPA[StressUnit.PSI] == pytest.approx(0.00689476, rel=1e-5)
        assert STRESS_TO_MPA[StressUnit.KSI] == pytest.approx(6.89476, rel=1e-5)

    def test_force_conversion_factors(self):
        """Test force conversion factors."""
        assert FORCE_TO_KN[ForceUnit.KN] == 1.0
        assert FORCE_TO_KN[ForceUnit.N] == 1e-3
        assert FORCE_TO_KN[ForceUnit.MN] == 1e3

    def test_length_conversion_calculation(self):
        """Test using conversion factors."""
        # Convert 1 meter to mm
        meters = 1.0
        mm = meters * LENGTH_TO_MM[LengthUnit.M]
        assert mm == 1000.0

        # Convert 12 inches to mm
        inches = 12.0
        mm = inches * LENGTH_TO_MM[LengthUnit.IN]
        assert mm == pytest.approx(304.8)

    def test_stress_conversion_calculation(self):
        """Test stress conversion."""
        # Convert 1 GPa to MPa
        gpa = 1.0
        mpa = gpa * STRESS_TO_MPA[StressUnit.GPA]
        assert mpa == 1000.0

        # Convert 1000 psi to MPa
        psi = 1000.0
        mpa = psi * STRESS_TO_MPA[StressUnit.PSI]
        assert mpa == pytest.approx(6.89476, rel=1e-5)
