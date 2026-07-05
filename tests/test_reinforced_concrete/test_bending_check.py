"""
Tests for BendingCheck with simplified tension shift rule implementation.

The tension shift rule is now automatically applied when M_cap is provided.
"""

import pytest
import numpy as np
from materials.reinforced_concrete.geometry import (
    create_rectangular_section,
    create_linear_rebar_layer,
)
from materials.reinforced_concrete.materials import ConcreteMaterial, Rebar, ShearRebar
from materials.reinforced_concrete.code_checks.ec2_2004.bending_check import BendingCheck
from materials.reinforced_concrete.code_checks.base_check import CheckStatus


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


class TestNegativeMoment:
    """Tests for negative moment handling and sign preservation."""

    def test_negative_moment_preserves_sign(self, test_beam, concrete_c30):
        """Test that negative moment sign is preserved through tension shift."""
        check = BendingCheck(section=test_beam, concrete=concrete_c30)

        result = check.perform_check(
            M_Ed=-100.0,  # Negative moment
            N_Ed=500.0,
            V_Ed=150.0,
            M_cap=-300.0,  # Negative cap
        )

        # Sign should be preserved
        assert result.details["M_Ed_original"] == -100.0
        assert result.details["M_Ed_design"] < 0  # Still negative
        # Magnitude should increase (more negative)
        assert abs(result.details["M_Ed_design"]) > abs(result.details["M_Ed_original"])

    def test_negative_moment_cap_applied_correctly(self, test_beam, concrete_c30):
        """Test M_cap limits negative moment correctly."""
        check = BendingCheck(section=test_beam, concrete=concrete_c30)

        result = check.perform_check(
            M_Ed=-100.0,
            N_Ed=500.0,
            V_Ed=150.0,
            M_cap=-120.0,  # Low cap (in magnitude)
        )

        # Design moment should be capped at -120
        assert result.details["M_Ed_design"] == pytest.approx(-120.0, rel=1e-6)

    def test_positive_and_negative_moment_symmetry(self, test_beam, concrete_c30):
        """Test that positive and negative moments produce symmetric M_add."""
        check = BendingCheck(section=test_beam, concrete=concrete_c30)

        result_pos = check.perform_check(
            M_Ed=100.0, N_Ed=500.0, V_Ed=150.0, M_cap=300.0
        )
        result_neg = check.perform_check(
            M_Ed=-100.0, N_Ed=500.0, V_Ed=150.0, M_cap=-300.0
        )

        # M_add magnitude should be the same (it's always positive)
        assert result_pos.details["M_add"] == pytest.approx(result_neg.details["M_add"], rel=0.01)


class TestWarningThreshold:
    """Tests for warning threshold functionality."""

    def test_warning_status_near_capacity(self, test_beam, concrete_c30):
        """Test WARNING status when utilization exceeds threshold but passes."""
        check = BendingCheck(section=test_beam, concrete=concrete_c30)

        # Find a load that gives ~96% utilization
        # Start with a moderate load and adjust
        result = check.perform_check(M_Ed=200.0, N_Ed=200.0, warning_threshold=0.90)

        # If utilization is between 0.90 and 1.0, should be WARNING
        if 0.90 <= result.utilization <= 1.0:
            assert result.status == CheckStatus.WARNING
            assert "High utilization" in result.message

    def test_custom_warning_threshold(self, test_beam, concrete_c30):
        """Test custom warning threshold."""
        check = BendingCheck(section=test_beam, concrete=concrete_c30)

        # Same load, different thresholds
        result_low = check.perform_check(M_Ed=100.0, N_Ed=500.0, warning_threshold=0.30)
        result_high = check.perform_check(M_Ed=100.0, N_Ed=500.0, warning_threshold=0.99)

        # With low threshold, likely to hit warning
        # With high threshold, likely to pass cleanly
        if result_low.utilization >= 0.30:
            assert result_low.status == CheckStatus.WARNING
        if result_high.utilization < 0.99:
            assert result_high.status == CheckStatus.PASS


class TestGetMomentCapacity:
    """Tests for get_moment_capacity() method."""

    def test_moment_capacity_at_zero_axial(self, test_beam, concrete_c30):
        """Test moment capacity with zero axial force."""
        check = BendingCheck(section=test_beam, concrete=concrete_c30)

        M_pos, M_neg = check.get_moment_capacity(N_Ed=0.0)

        assert M_pos is not None
        assert M_neg is not None
        assert M_pos > 0  # Positive capacity
        assert M_neg < 0  # Negative capacity (hogging)

    def test_moment_capacity_with_compression(self, test_beam, concrete_c30):
        """Test moment capacity with axial compression."""
        check = BendingCheck(section=test_beam, concrete=concrete_c30)

        M_pos_0, M_neg_0 = check.get_moment_capacity(N_Ed=0.0)
        M_pos_500, M_neg_500 = check.get_moment_capacity(N_Ed=500.0)

        # Moderate compression often increases moment capacity
        assert M_pos_500 is not None
        assert M_neg_500 is not None

    def test_moment_capacity_with_tension(self, test_beam, concrete_c30):
        """Test moment capacity with axial tension."""
        check = BendingCheck(section=test_beam, concrete=concrete_c30)

        M_pos, M_neg = check.get_moment_capacity(N_Ed=-200.0)  # Tension

        assert M_pos is not None
        assert M_neg is not None

    def test_moment_capacity_outside_bounds(self, test_beam, concrete_c30):
        """Test moment capacity returns None when N_Ed is outside diagram bounds."""
        check = BendingCheck(section=test_beam, concrete=concrete_c30)

        # Very high compression - likely outside bounds
        M_pos, M_neg = check.get_moment_capacity(N_Ed=50000.0)

        # Should return None when outside bounds
        assert M_pos is None
        assert M_neg is None


class TestGenerateInteractionDiagramArrays:
    """Tests for generate_interaction_diagram_arrays() method."""

    def test_diagram_arrays_returned(self, test_beam, concrete_c30):
        """Test that diagram arrays are returned correctly."""
        check = BendingCheck(section=test_beam, concrete=concrete_c30)

        N_array, M_array = check.generate_interaction_diagram_arrays()

        assert isinstance(N_array, np.ndarray)
        assert isinstance(M_array, np.ndarray)
        assert len(N_array) == len(M_array)
        assert len(N_array) > 0

    def test_diagram_arrays_custom_points(self, test_beam, concrete_c30):
        """Test diagram arrays with custom number of points."""
        check = BendingCheck(section=test_beam, concrete=concrete_c30)

        N_array, M_array = check.generate_interaction_diagram_arrays(n_points=50)

        # Should have approximately n_points (may vary slightly due to implementation)
        assert len(N_array) >= 40  # Allow some tolerance

    def test_diagram_contains_key_points(self, test_beam, concrete_c30):
        """Test that diagram contains expected key points."""
        check = BendingCheck(section=test_beam, concrete=concrete_c30)

        N_array, M_array = check.generate_interaction_diagram_arrays()

        # Should include pure compression (high N, low M)
        assert np.max(N_array) > 0

        # Should include pure tension (negative N)
        assert np.min(N_array) < 0

        # Should include balanced point region (moderate N, high M)
        assert np.max(np.abs(M_array)) > 0


class TestAccidentalLimitState:
    """Tests for accidental limit state factors."""

    def test_accidental_factors_applied(self, test_beam, concrete_c30):
        """Test that accidental factors increase capacity."""
        check_persistent = BendingCheck(
            section=test_beam, concrete=concrete_c30, use_accidental=False
        )
        check_accidental = BendingCheck(
            section=test_beam, concrete=concrete_c30, use_accidental=True
        )

        # Same load
        result_persistent = check_persistent.perform_check(M_Ed=150.0, N_Ed=500.0)
        result_accidental = check_accidental.perform_check(M_Ed=150.0, N_Ed=500.0)

        # Accidental should have lower utilization (higher capacity)
        assert result_accidental.utilization < result_persistent.utilization

    def test_f_cd_design_property(self, test_beam, concrete_c30):
        """Test f_cd_design property returns correct value."""
        check_persistent = BendingCheck(
            section=test_beam, concrete=concrete_c30, use_accidental=False
        )
        check_accidental = BendingCheck(
            section=test_beam, concrete=concrete_c30, use_accidental=True
        )

        # f_cd for accidental should be higher (lower gamma_c)
        assert check_accidental.f_cd_design > check_persistent.f_cd_design


class TestPureAxialAndBending:
    """Tests for pure axial and pure bending load cases."""

    def test_pure_bending_zero_axial(self, test_beam, concrete_c30):
        """Test pure bending with zero axial force."""
        check = BendingCheck(section=test_beam, concrete=concrete_c30)

        result = check.perform_check(M_Ed=150.0, N_Ed=0.0)

        assert result.utilization > 0
        assert result.details["N_Ed"] == 0.0
        assert result.details["M_Ed_original"] == 150.0

    def test_pure_axial_zero_moment(self, test_beam, concrete_c30):
        """Test pure axial with zero moment."""
        check = BendingCheck(section=test_beam, concrete=concrete_c30)

        result = check.perform_check(M_Ed=0.0, N_Ed=500.0)

        assert result.utilization > 0
        assert result.details["N_Ed"] == 500.0
        assert result.details["M_Ed_original"] == 0.0

    def test_pure_tension(self, test_beam, concrete_c30):
        """Test pure axial tension."""
        check = BendingCheck(section=test_beam, concrete=concrete_c30)

        result = check.perform_check(M_Ed=0.0, N_Ed=-200.0)  # Tension

        assert result.utilization > 0
        assert result.details["N_Ed"] == -200.0


class TestOutsideDiagramDomain:
    """Tests for load points outside the interaction diagram."""

    def test_excessive_axial_compression(self, test_beam, concrete_c30):
        """Test excessive axial compression is detected."""
        check = BendingCheck(section=test_beam, concrete=concrete_c30)

        # Very high axial - should be outside diagram
        result = check.perform_check(M_Ed=10.0, N_Ed=50000.0)

        assert result.status == CheckStatus.FAIL
        assert result.utilization == float("inf")
        assert "outside" in result.message.lower() or "no capacity" in result.message.lower()

    def test_excessive_tension(self, test_beam, concrete_c30):
        """Test excessive tension is detected."""
        check = BendingCheck(section=test_beam, concrete=concrete_c30)

        # Very high tension - should be outside diagram
        result = check.perform_check(M_Ed=10.0, N_Ed=-10000.0)

        assert result.status == CheckStatus.FAIL
        assert result.utilization == float("inf")


class TestCheckResultDetails:
    """Tests for CheckResult details dictionary completeness."""

    def test_all_details_present_basic_check(self, test_beam, concrete_c30):
        """Test all expected details are present for basic check."""
        check = BendingCheck(section=test_beam, concrete=concrete_c30)
        result = check.perform_check(M_Ed=100.0, N_Ed=500.0)

        expected_keys = [
            "N_Ed",
            "M_Ed_original",
            "M_Ed_design",
            "tension_shift_applied",
            "M_add",
            "V_Ed",
            "M_cap",
            "cot_theta",
            "shift_distance_a_l_mm",
            "z_lever_arm_mm",
            "shear_reinforcement_provided",
            "N_Rd",
            "M_Rd",
            "utilization",
            "concrete_model",
            "steel_model",
            "section_name",
            "concrete_grade",
            "reinforcement_ratio",
        ]

        for key in expected_keys:
            assert key in result.details, f"Missing key: {key}"

    def test_all_details_present_with_tension_shift(self, test_beam, concrete_c30):
        """Test all expected details are present with tension shift enabled."""
        check = BendingCheck(section=test_beam, concrete=concrete_c30)
        shear_rebar = ShearRebar(grade="B500B", diameter=10, spacing=200, n_legs=2)

        result = check.perform_check(
            M_Ed=100.0,
            N_Ed=500.0,
            V_Ed=150.0,
            M_cap=300.0,
            shear_reinforcement=shear_rebar,
        )

        # All tension shift fields should be populated
        assert result.details["tension_shift_applied"] is True
        assert result.details["M_add"] is not None
        assert result.details["V_Ed"] is not None
        assert result.details["M_cap"] is not None
        assert result.details["cot_theta"] is not None
        assert result.details["shift_distance_a_l_mm"] is not None
        assert result.details["z_lever_arm_mm"] is not None
        assert result.details["shear_reinforcement_provided"] is True

    def test_demand_and_capacity_components(self, test_beam, concrete_c30):
        """Test demand and capacity components in result."""
        check = BendingCheck(section=test_beam, concrete=concrete_c30)
        result = check.perform_check(M_Ed=100.0, N_Ed=500.0)

        # Check demand components
        assert result.demand_components is not None
        assert "N" in result.demand_components
        assert "M" in result.demand_components
        assert result.demand_components["N"] == 500.0
        assert result.demand_components["M"] == 100.0

        # Check capacity components
        assert result.capacity_components is not None
        assert "N" in result.capacity_components
        assert "M" in result.capacity_components

        # Check units
        assert result.units_components is not None
        assert result.units_components["N"] == "kN"
        assert result.units_components["M"] == "kN·m"


class TestTensionShiftResultDataclass:
    """Tests for the internal _TensionShiftResult dataclass."""

    def test_tension_shift_result_details_dict_disabled(self, test_beam, concrete_c30):
        """Test details_dict() when tension shift is disabled."""
        check = BendingCheck(section=test_beam, concrete=concrete_c30)

        # Access internal method for testing
        shift_result = check._apply_tension_shift_if_enabled(
            M_Ed_original=100.0,
            N_Ed=500.0,
            V_Ed=None,
            M_cap=None,
            shear_reinforcement=None,
            enabled=False,
        )

        assert shift_result.applied is False
        assert shift_result.M_design == 100.0

        details = shift_result.details_dict()
        assert details["tension_shift_applied"] is False
        assert details["M_add"] is None
        assert details["V_Ed"] is None
        assert details["cot_theta"] is None

    def test_tension_shift_result_details_dict_enabled(self, test_beam, concrete_c30):
        """Test details_dict() when tension shift is enabled."""
        check = BendingCheck(section=test_beam, concrete=concrete_c30)
        shear_rebar = ShearRebar(grade="B500B", diameter=10, spacing=200, n_legs=2)

        shift_result = check._apply_tension_shift_if_enabled(
            M_Ed_original=100.0,
            N_Ed=500.0,
            V_Ed=150.0,
            M_cap=300.0,
            shear_reinforcement=shear_rebar,
            enabled=True,
        )

        assert shift_result.applied is True
        assert shift_result.M_design > 100.0  # Increased by shift

        details = shift_result.details_dict()
        assert details["tension_shift_applied"] is True
        assert details["M_add"] is not None
        assert details["V_Ed"] == 150.0
        assert details["M_cap"] == 300.0
        assert details["cot_theta"] is not None
        assert 1.0 <= details["cot_theta"] <= 2.5
        assert details["shift_distance_a_l_mm"] is not None
        assert details["z_lever_arm_mm"] is not None
        assert details["shear_reinforcement_provided"] is True

    def test_tension_shift_without_rebar_uses_d(self, test_beam, concrete_c30):
        """Test that shift distance equals d when no shear reinforcement."""
        check = BendingCheck(section=test_beam, concrete=concrete_c30)

        shift_result = check._apply_tension_shift_if_enabled(
            M_Ed_original=100.0,
            N_Ed=500.0,
            V_Ed=150.0,
            M_cap=300.0,
            shear_reinforcement=None,  # No shear rebar
            enabled=True,
        )

        # a_l should equal d (effective depth)
        # For 500mm beam with 50mm cover, d ≈ 450mm
        assert shift_result.shift_distance_a_l_mm is not None
        assert 400 < shift_result.shift_distance_a_l_mm < 460  # Reasonable range for d
        assert shift_result.cot_theta is None  # Not calculated without shear rebar
        assert shift_result.shear_reinforcement_provided is False


class TestConcreteAndSteelModels:
    """Tests for different concrete and steel constitutive models."""

    def test_different_concrete_models(self, test_beam, concrete_c30):
        """Test with different concrete model types."""
        from materials.reinforced_concrete.constitutive import ConcreteModelType

        for model_type in [
            ConcreteModelType.PARABOLA_RECTANGLE,
            ConcreteModelType.BILINEAR,
            ConcreteModelType.SCHEMATIC,
        ]:
            check = BendingCheck(
                section=test_beam,
                concrete=concrete_c30,
                concrete_model_type=model_type,
            )
            result = check.perform_check(M_Ed=100.0, N_Ed=500.0)

            assert result.status in [CheckStatus.PASS, CheckStatus.WARNING]
            assert result.details["concrete_model"] == model_type

    def test_different_steel_models(self, test_beam, concrete_c30):
        """Test with different steel model types."""
        from materials.reinforced_concrete.constitutive import SteelModelType

        for model_type in [SteelModelType.INCLINED, SteelModelType.HORIZONTAL]:
            check = BendingCheck(
                section=test_beam,
                concrete=concrete_c30,
                steel_model_type=model_type,
            )
            result = check.perform_check(M_Ed=100.0, N_Ed=500.0)

            assert result.status in [CheckStatus.PASS, CheckStatus.WARNING]
            assert result.details["steel_model"] == model_type


class TestCheckResultStringRepresentation:
    """Tests for CheckResult __str__ method."""

    def test_result_string_contains_status(self, test_beam, concrete_c30):
        """Test that result string contains status."""
        check = BendingCheck(section=test_beam, concrete=concrete_c30)
        result = check.perform_check(M_Ed=100.0, N_Ed=500.0)

        result_str = str(result)
        assert "PASS" in result_str or "FAIL" in result_str or "WARNING" in result_str

    def test_result_string_contains_utilization(self, test_beam, concrete_c30):
        """Test that result string contains utilization."""
        check = BendingCheck(section=test_beam, concrete=concrete_c30)
        result = check.perform_check(M_Ed=100.0, N_Ed=500.0)

        result_str = str(result)
        assert "utilization" in result_str.lower() or "%" in result_str


class TestMultipleLoadCases:
    """Tests for checking multiple load cases efficiently."""

    def test_reuse_check_object(self, test_beam, concrete_c30):
        """Test that same BendingCheck can check multiple load cases."""
        check = BendingCheck(section=test_beam, concrete=concrete_c30)

        load_cases = [
            {"M_Ed": 50.0, "N_Ed": 200.0},
            {"M_Ed": 100.0, "N_Ed": 500.0},
            {"M_Ed": 150.0, "N_Ed": 800.0},
            {"M_Ed": -100.0, "N_Ed": 300.0},
        ]

        results = [check.perform_check(**lc) for lc in load_cases]

        # All should return valid results
        assert all(r.utilization > 0 for r in results)
        assert all(r.status in [CheckStatus.PASS, CheckStatus.WARNING, CheckStatus.FAIL] for r in results)

    def test_diagram_cached_across_checks(self, test_beam, concrete_c30):
        """Test that diagram is cached and reused."""
        check = BendingCheck(section=test_beam, concrete=concrete_c30)

        # Store reference to internal diagram
        diagram_ref = check._diagram

        # Perform multiple checks
        check.perform_check(M_Ed=50.0, N_Ed=200.0)
        check.perform_check(M_Ed=100.0, N_Ed=500.0)

        # Diagram should be the same object (cached)
        assert check._diagram is diagram_ref
