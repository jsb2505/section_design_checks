"""
Tests for BendingCheck with simplified tension shift rule implementation.

The tension shift rule is now automatically applied when M_cap is provided.
"""

import pytest
from materials.reinforced_concrete.geometry import (
    create_rectangular_section,
    create_linear_rebar_layer,
)
from materials.reinforced_concrete.materials import ConcreteMaterial, Rebar, ShearRebar
from materials.reinforced_concrete.code_checks.ec2.bending_check import BendingCheck


@pytest.fixture
def test_beam():
    """Create a standard test beam section."""
    section = create_rectangular_section(width=300, height=500, section_name="Test Beam")
    rebar = Rebar(grade="B500B", diameter=20)

    # Bottom reinforcement (3×ø20)
    bottom_layer = create_linear_rebar_layer(
        rebar=rebar,
        n_bars=3,
        start_point=(50, 50),
        end_point=(250, 50),
        layer_name="bottom",
    )
    section.add_rebar_group(bottom_layer)

    # Top reinforcement (2×ø20)
    top_layer = create_linear_rebar_layer(
        rebar=rebar,
        n_bars=2,
        start_point=(75, 450),
        end_point=(225, 450),
        layer_name="top",
    )
    section.add_rebar_group(top_layer)

    return section


@pytest.fixture
def concrete_c30():
    """Create C30/37 concrete material."""
    return ConcreteMaterial(grade="C30/37")


class TestBasicBendingCheck:
    """Tests for basic bending check functionality without tension shift."""

    def test_basic_check_safe(self, test_beam, concrete_c30):
        """Test basic bending check for safe load."""
        check = BendingCheck(section=test_beam, concrete=concrete_c30)
        result = check.perform_check(M_Ed=100.0, N_Ed=500.0)

        assert result.status.name == "PASS"
        assert 0 < result.utilization < 1.0
        assert result.details["tension_shift_applied"] is False
        assert result.details["M_add"] is None

    def test_basic_check_failing(self, test_beam, concrete_c30):
        """Test basic bending check for failing load."""
        check = BendingCheck(section=test_beam, concrete=concrete_c30)
        result = check.perform_check(M_Ed=500.0, N_Ed=500.0)

        assert result.status.name == "FAIL"
        assert result.utilization > 1.0
        assert result.details["tension_shift_applied"] is False


class TestTensionShiftRuleSimplified:
    """Tests for simplified EC2 §9.2.1.3 tension shift rule implementation."""

    def test_tension_shift_requires_V_Ed(self, test_beam, concrete_c30):
        """Test that V_Ed is required when M_cap is provided."""
        check = BendingCheck(section=test_beam, concrete=concrete_c30)

        with pytest.raises(ValueError, match="V_Ed must be provided when M_cap is provided"):
            check.perform_check(
                M_Ed=100.0,
                N_Ed=500.0,
                M_cap=200.0,  # Enables tension shift
                # V_Ed not provided
            )

    def test_M_cap_enables_tension_shift(self, test_beam, concrete_c30):
        """Test that providing M_cap automatically enables tension shift."""
        check = BendingCheck(section=test_beam, concrete=concrete_c30)

        result = check.perform_check(
            M_Ed=100.0,
            N_Ed=500.0,
            V_Ed=150.0,
            M_cap=300.0,
        )

        assert result.details["tension_shift_applied"] is True
        assert result.details["M_add"] is not None
        assert result.details["M_add"] > 0

    def test_no_M_cap_no_tension_shift(self, test_beam, concrete_c30):
        """Test that without M_cap, tension shift is not applied."""
        check = BendingCheck(section=test_beam, concrete=concrete_c30)

        result = check.perform_check(
            M_Ed=100.0,
            N_Ed=500.0,
            V_Ed=150.0,  # V_Ed provided but M_cap is not
        )

        assert result.details["tension_shift_applied"] is False
        assert result.details["M_add"] is None

    def test_tension_shift_without_shear_reinforcement(self, test_beam, concrete_c30):
        """Test tension shift uses a_l = d when no shear reinforcement."""
        check = BendingCheck(section=test_beam, concrete=concrete_c30)

        result = check.perform_check(
            M_Ed=100.0,
            N_Ed=500.0,
            V_Ed=150.0,
            M_cap=300.0,
            # No shear_reinforcement provided
        )

        # Should use a_l = d ≈ 0.9 * 500 = 450 mm
        # M_add = V_Ed * a_l = 150 * 0.450 = 67.5 kN·m
        expected_M_add = 150 * 0.9 * 500 / 1000.0

        assert result.details["shear_reinforcement_provided"] is False
        assert result.details["cot_theta"] is None  # Not applicable without shear reinforcement
        assert result.details["M_add"] == pytest.approx(expected_M_add, rel=0.01)

    def test_tension_shift_with_shear_reinforcement(self, test_beam, concrete_c30):
        """Test tension shift calculates cot_theta when shear reinforcement provided."""
        check = BendingCheck(section=test_beam, concrete=concrete_c30)

        # Create shear reinforcement
        shear_rebar = ShearRebar(grade="B500B", diameter=10, spacing=200, n_legs=2)

        result = check.perform_check(
            M_Ed=100.0,
            N_Ed=500.0,
            V_Ed=150.0,
            M_cap=300.0,
            shear_reinforcement=shear_rebar,
        )

        # Should calculate cot_theta from V_Ed
        assert result.details["shear_reinforcement_provided"] is True
        assert result.details["cot_theta"] is not None
        assert 1.0 <= result.details["cot_theta"] <= 2.5  # EC2 limits
        assert result.details["M_add"] is not None
        assert result.details["M_add"] > 0

    def test_tension_shift_M_cap_limits_design_moment(self, test_beam, concrete_c30):
        """Test that M_cap correctly limits the design moment."""
        check = BendingCheck(section=test_beam, concrete=concrete_c30)

        # Use a low M_cap that will limit M_Ed + M_add
        result = check.perform_check(
            M_Ed=100.0,
            N_Ed=500.0,
            V_Ed=150.0,
            M_cap=120.0,  # Low cap
        )

        M_Ed_orig = result.details["M_Ed_original"]
        M_add = result.details["M_add"]
        M_Ed_design = result.details["M_Ed_design"]
        M_cap = result.details["M_cap"]

        # Check that cap was applied
        assert M_Ed_orig + M_add > M_cap  # Would exceed without cap
        assert M_Ed_design == pytest.approx(M_cap, rel=1e-6)

    def test_tension_shift_increases_utilization(self, test_beam, concrete_c30):
        """Test that tension shift increases utilization."""
        check = BendingCheck(section=test_beam, concrete=concrete_c30)

        # Without tension shift
        result_no_shift = check.perform_check(M_Ed=100.0, N_Ed=500.0)

        # With tension shift (high M_cap won't limit)
        result_with_shift = check.perform_check(
            M_Ed=100.0,
            N_Ed=500.0,
            V_Ed=150.0,
            M_cap=300.0,
        )

        # Utilization should increase
        assert result_with_shift.utilization > result_no_shift.utilization

    def test_tension_shift_zero_shear(self, test_beam, concrete_c30):
        """Test tension shift with zero shear force."""
        check = BendingCheck(section=test_beam, concrete=concrete_c30)

        result = check.perform_check(
            M_Ed=100.0,
            N_Ed=500.0,
            V_Ed=0.0,
            M_cap=300.0,
        )

        # With zero shear, M_add should be zero
        assert result.details["M_add"] == pytest.approx(0.0, abs=1e-6)
        assert result.details["M_Ed_design"] == pytest.approx(100.0, rel=1e-6)

    def test_tension_shift_details_populated(self, test_beam, concrete_c30):
        """Test that all tension shift details are properly populated."""
        check = BendingCheck(section=test_beam, concrete=concrete_c30)

        result = check.perform_check(
            M_Ed=100.0,
            N_Ed=500.0,
            V_Ed=150.0,
            M_cap=200.0,
        )

        details = result.details

        # Check all fields are populated
        assert details["tension_shift_applied"] is True
        assert details["M_Ed_original"] == 100.0
        assert details["M_Ed_design"] > 100.0
        assert details["M_add"] is not None
        assert details["M_add"] > 0
        assert details["V_Ed"] == 150.0
        assert details["shift_distance_a_l_mm"] is not None
        assert details["shift_distance_a_l_mm"] > 0
        assert details["z_lever_arm_mm"] is not None
        assert details["M_cap"] == 200.0
        assert details["shear_reinforcement_provided"] is False
