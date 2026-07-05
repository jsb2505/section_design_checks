"""
Tests for CircularSectionCheck circular shear web-width behavior.
"""

from math import sqrt

import pytest

from materials.core.geometry import Point2D
from materials.reinforced_concrete.code_checks.ec2_2004.circular_section_check import (
    CircularSectionCheck,
)
from materials.reinforced_concrete.code_checks.ec2_2004.shear_check import ShearLoadCase
from materials.reinforced_concrete.geometry import (
    RebarGroup,
    create_circular_perimeter_rebars,
    create_circular_section,
)
from materials.reinforced_concrete.materials import ConcreteMaterial, Rebar, ShearRebar


def _make_circular_section(diameter: float = 600.0):
    section = create_circular_section(diameter=diameter, hook_ref=0)
    perimeter = create_circular_perimeter_rebars(
        rebar=Rebar(diameter=20, grade="B500B"),
        diameter=diameter,
        cover=40.0,
        n_bars=12,
        origin=(0.0, 0.0),
    )
    section.add_rebar_group(perimeter)
    return section


def _make_asymmetric_circular_section_for_rho_l(diameter: float = 600.0):
    section = create_circular_section(diameter=diameter, hook_ref=0)

    top_group = RebarGroup(
        rebar=Rebar(diameter=28, grade="B500B"),
        positions=(Point2D(x=0.0, y=220.0),),
        layer_name="top",
    )
    bottom_group = RebarGroup(
        rebar=Rebar(diameter=12, grade="B500B"),
        positions=(Point2D(x=0.0, y=-220.0),),
        layer_name="bottom",
    )

    section.add_rebar_group(top_group)
    section.add_rebar_group(bottom_group)
    return section


def _expected_width(d: float, z: float, r: float, r_sv: float) -> tuple[float, float, float]:
    c = max(d - z, 0.0)
    b_wc = 2.0 * sqrt(max(c * (2.0 * r - c), 0.0))

    e = max(r + r_sv - d, 0.0)
    b_wt = 2.0 * sqrt(max(e * (2.0 * r_sv - e), 0.0))

    if b_wc <= 0.0 and b_wt <= 0.0:
        b_w = 0.0
    elif b_wc <= 0.0:
        b_w = b_wt
    elif b_wt <= 0.0:
        b_w = b_wc
    else:
        b_w = min(b_wc, b_wt)

    return b_w, b_wc, b_wt


class TestCircularEquivalentWebWidth:
    """Tests for TestCircularEquivalentWebWidth."""
    def test_unreinforced_width_is_independent_of_cover(self):
        """Test unreinforced width is independent of cover."""
        section = _make_circular_section()
        concrete = ConcreteMaterial(grade="C30/37")

        check_cover_30 = CircularSectionCheck(
            section=section,
            concrete=concrete,
            diameter=600.0,
            cover=30.0,
            shear_reinforcement=None,
        )
        check_cover_80 = CircularSectionCheck(
            section=section,
            concrete=concrete,
            diameter=600.0,
            cover=80.0,
            shear_reinforcement=None,
        )

        d = 510.0
        z = 459.0

        bw_30, bwc_30, bwt_30 = check_cover_30.calculate_equivalent_web_width(d, z)
        bw_80, bwc_80, bwt_80 = check_cover_80.calculate_equivalent_web_width(d, z)

        assert bw_30 == pytest.approx(bw_80, rel=1e-12)
        assert bwc_30 == pytest.approx(bwc_80, rel=1e-12)
        assert bwt_30 == pytest.approx(bwt_80, rel=1e-12)

    def test_reinforced_width_matches_eq10_to_13_formula(self):
        """Test reinforced width matches eq10 to 13 formula."""
        section = _make_circular_section()
        concrete = ConcreteMaterial(grade="C30/37")
        shear_rebar = ShearRebar(diameter=12, link_spacing=200, n_legs=2, grade="B500B")

        check = CircularSectionCheck(
            section=section,
            concrete=concrete,
            diameter=600.0,
            cover=50.0,
            shear_reinforcement=shear_rebar,
        )

        d = 510.0
        z = 459.0

        r = 300.0
        r_sv = r - 50.0 - shear_rebar.diameter / 2.0
        expected_bw, expected_bwc, expected_bwt = _expected_width(d, z, r, r_sv)

        b_w, b_wc, b_wt = check.calculate_equivalent_web_width(d, z)

        assert b_w == pytest.approx(expected_bw, rel=1e-12)
        assert b_wc == pytest.approx(expected_bwc, rel=1e-12)
        assert b_wt == pytest.approx(expected_bwt, rel=1e-12)

    def test_unreinforced_can_still_be_governed_by_tension_chord(self):
        """Test unreinforced can still be governed by tension chord."""
        section = _make_circular_section()
        concrete = ConcreteMaterial(grade="C30/37")

        check = CircularSectionCheck(
            section=section,
            concrete=concrete,
            diameter=600.0,
            cover=80.0,
            shear_reinforcement=None,
        )

        d = 510.0
        z = 400.0
        b_w, b_wc, b_wt = check.calculate_equivalent_web_width(d, z)

        assert b_w == pytest.approx(b_wt, rel=1e-12)
        assert b_w < b_wc


class TestCircularShearCapacityPolicy:
    """Tests for TestCircularShearCapacityPolicy."""
    def test_perform_shear_check_requires_shear_reinforcement(self):
        """Test perform shear check requires shear reinforcement."""
        section = _make_circular_section()
        concrete = ConcreteMaterial(grade="C30/37")

        check = CircularSectionCheck(
            section=section,
            concrete=concrete,
            diameter=600.0,
            cover=30.0,
            shear_reinforcement=None,
        )

        load = ShearLoadCase(V_Ed=200.0, M_Ed=150.0, N_Ed=1000.0)

        with pytest.raises(ValueError, match="requires shear_reinforcement"):
            check.perform_shear_check(load_case=load, suppress_warnings=True)

    def test_reports_both_cracked_and_uncracked_vrdc(self):
        """Test reports both cracked and uncracked vrdc."""
        section = _make_circular_section()
        concrete = ConcreteMaterial(grade="C30/37")
        shear_rebar = ShearRebar(diameter=12, link_spacing=200, n_legs=2, grade="B500B")

        check = CircularSectionCheck(
            section=section,
            concrete=concrete,
            diameter=600.0,
            cover=50.0,
            shear_reinforcement=shear_rebar,
        )

        load = ShearLoadCase(V_Ed=200.0, M_Ed=150.0, N_Ed=1000.0)
        result = check.perform_shear_check(load_case=load, suppress_warnings=True)

        assert result.details["use_uncracked_V_Rd_c"] is False
        assert result.details["V_Rd_c"] == pytest.approx(result.details["V_Rd_c_cracked"], rel=1e-12)
        assert result.details["V_Rd_c_uncracked"] is not None
        assert result.details["V_Rd_s"] is not None
        assert result.details["V_Rd_max"] is not None

    def test_use_uncracked_vrdc_changes_selected_vrdc_not_reinforced_capacity(self):
        """Test use uncracked vrdc changes selected vrdc not reinforced capacity."""
        section = _make_circular_section()
        concrete = ConcreteMaterial(grade="C30/37")
        shear_rebar = ShearRebar(diameter=12, link_spacing=200, n_legs=2, grade="B500B")

        check = CircularSectionCheck(
            section=section,
            concrete=concrete,
            diameter=600.0,
            cover=50.0,
            shear_reinforcement=shear_rebar,
        )

        load = ShearLoadCase(V_Ed=200.0, M_Ed=150.0, N_Ed=1000.0)

        result_default = check.perform_shear_check(
            load_case=load,
            suppress_warnings=True,
        )
        result_uncracked = check.perform_shear_check(
            load_case=load,
            use_uncracked_V_Rd_c=True,
            suppress_warnings=True,
        )

        assert result_default.capacity == pytest.approx(result_uncracked.capacity, rel=1e-12)
        assert result_default.details["V_Rd_c"] == pytest.approx(
            result_default.details["V_Rd_c_cracked"], rel=1e-12
        )
        assert result_uncracked.details["V_Rd_c"] == pytest.approx(
            result_uncracked.details["V_Rd_c_uncracked"], rel=1e-12
        )


class TestCircularRhoLFromStrains:
    """Tests for TestCircularRhoLFromStrains."""
    def test_rho_l_uses_tension_side_not_centroid_side(self):
        """Test rho l uses tension side not centroid side."""
        section = _make_asymmetric_circular_section_for_rho_l()
        concrete = ConcreteMaterial(grade="C30/37")
        check = CircularSectionCheck(
            section=section,
            concrete=concrete,
            diameter=600.0,
            cover=50.0,
            shear_reinforcement=None,
        )

        b_w = 300.0
        d = 500.0

        # Sagging-like strain state: bottom bar in tension
        rho_sagging = check._find_rho_l(
            M_Ed=100.0,
            N_Ed=0.0,
            b_w=b_w,
            d=d,
            eps_top=0.001,
            eps_bottom=-0.001,
        )

        # Hogging-like strain state: top bar in tension (larger bar area)
        rho_hogging = check._find_rho_l(
            M_Ed=-100.0,
            N_Ed=0.0,
            b_w=b_w,
            d=d,
            eps_top=-0.001,
            eps_bottom=0.001,
        )

        assert rho_hogging > rho_sagging

        top_area = Rebar(diameter=28, grade="B500B").area
        bottom_area = Rebar(diameter=12, grade="B500B").area
        assert rho_sagging == pytest.approx(bottom_area / (b_w * d), rel=1e-12)
        assert rho_hogging == pytest.approx(top_area / (b_w * d), rel=1e-12)

    def test_rho_l_is_zero_when_all_bars_are_in_compression(self):
        """Test rho l is zero when all bars are in compression."""
        section = _make_asymmetric_circular_section_for_rho_l()
        concrete = ConcreteMaterial(grade="C30/37")
        check = CircularSectionCheck(
            section=section,
            concrete=concrete,
            diameter=600.0,
            cover=50.0,
            shear_reinforcement=None,
        )

        rho_l = check._find_rho_l(
            M_Ed=0.0,
            N_Ed=1000.0,
            b_w=300.0,
            d=500.0,
            eps_top=0.001,
            eps_bottom=0.001,
        )
        assert rho_l == pytest.approx(0.0, abs=1e-15)


class TestCircularStressLimitsWrapper:
    """Tests for TestCircularStressLimitsWrapper."""
    def test_wrapper_matches_internal_stress_limits_check(self):
        """Test wrapper matches internal stress limits check."""
        section = _make_circular_section()
        concrete = ConcreteMaterial(grade="C30/37")
        check = CircularSectionCheck(
            section=section,
            concrete=concrete,
            diameter=600.0,
            cover=50.0,
            shear_reinforcement=None,
        )

        M_Ed = 80.0
        N_Ed = 300.0

        wrapped = check.perform_stress_limits_check(
            M_Ed=M_Ed,
            N_Ed=N_Ed,
            warning_threshold=0.9,
            ignore_compression_steel=True,
            suppress_warnings=True,
        )
        direct = check.stress_limits.perform_check(
            M_Ed=M_Ed,
            N_Ed=N_Ed,
            warning_threshold=0.9,
            ignore_compression_steel=True,
            suppress_warnings=True,
        )

        assert wrapped.utilization == pytest.approx(direct.utilization, rel=1e-12)
        assert wrapped.details["governing_check"] == direct.details["governing_check"]
