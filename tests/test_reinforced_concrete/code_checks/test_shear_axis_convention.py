"""
Regression tests for the shear-axis (Vy/Vz) convention.

Convention under test:
- ``Vz_Ed`` = major-axis shear, acting along the vertical (z) axis.
- ``Vy_Ed`` = minor-axis shear, acting along the horizontal (y) axis.
- ``V_Ed``  = direction-agnostic resultant, a first-class input that maps to the
  major axis (``Vz_Ed``) with no DeprecationWarning.
- ``My_Ed`` (major) / ``Mz_Ed`` (minor) moments are unchanged.

These would FAIL under the previous (inverted) convention where ``Vy_Ed`` was the
major/vertical shear: a pure ``Vz_Ed`` would have produced the horizontal
direction ``(1, 0)`` and the minor-axis breadth.
"""

import warnings
from math import hypot

import pytest

from section_design_checks.core.geometry import Point2D
from section_design_checks.reinforced_concrete.code_checks.ec2_2004.flexure_utils import LoadCase
from section_design_checks.reinforced_concrete.code_checks.ec2_2004.shear_check import ShearCheck
from section_design_checks.reinforced_concrete.geometry import RebarGroup, create_rectangular_section
from section_design_checks.reinforced_concrete.materials import ConcreteMaterial, Rebar

# 300 (width) × 500 (height): deliberately non-square so the major-axis (vertical)
# breadth (≈ width 300) differs from the minor-axis (horizontal) breadth (≈ height 500).
WIDTH = 300.0
HEIGHT = 500.0


def _asymmetric_check():
    section = create_rectangular_section(width=WIDTH, height=HEIGHT)
    rebar = Rebar(diameter=20, grade="B500B")
    section.add_rebar_group(
        RebarGroup(rebar=rebar, positions=[Point2D(x=100, y=50), Point2D(x=200, y=50)])
    )
    return ShearCheck(section=section, concrete=ConcreteMaterial(grade="C30/37"))


class TestShearAxisConvention:
    def test_major_shear_Vz_maps_to_vertical_axis(self):
        """Pure Vz_Ed → vertical direction (0, 1) and the major-axis breadth (≈ width)."""
        check = _asymmetric_check()
        details = check.perform_check(load_case=LoadCase(Vz_Ed=150.0)).details

        vx, vy = details["shear_direction"]
        assert (vx, vy) == pytest.approx((0.0, 1.0), abs=1e-9)  # vertical
        # Breadth for vertical shear is measured horizontally → the 300 mm width.
        assert details["b_w"] == pytest.approx(WIDTH, rel=0.02)

    def test_minor_shear_Vy_maps_to_horizontal_axis(self):
        """Pure Vy_Ed → horizontal direction (1, 0) and the minor-axis breadth (≈ height)."""
        check = _asymmetric_check()
        details = check.perform_check(load_case=LoadCase(Vy_Ed=150.0)).details

        vx, vy = details["shear_direction"]
        assert (vx, vy) == pytest.approx((1.0, 0.0), abs=1e-9)  # horizontal
        # Breadth for horizontal shear is measured vertically → the 500 mm height.
        assert details["b_w"] == pytest.approx(HEIGHT, rel=0.02)

    def test_major_and_minor_breadths_differ(self):
        """Sanity: the two axes really do give different breadths on this section."""
        check = _asymmetric_check()
        b_major = check.perform_check(load_case=LoadCase(Vz_Ed=150.0)).details["b_w"]
        b_minor = check.perform_check(load_case=LoadCase(Vy_Ed=150.0)).details["b_w"]
        assert b_major < b_minor

    def test_resultant_V_Ed_maps_to_major_axis(self):
        """The agnostic V_Ed input behaves identically to the explicit major Vz_Ed."""
        check = _asymmetric_check()
        d_ved = check.perform_check(load_case=LoadCase(V_Ed=150.0)).details
        d_vz = check.perform_check(load_case=LoadCase(Vz_Ed=150.0)).details

        assert d_ved["shear_direction"] == pytest.approx(d_vz["shear_direction"], abs=1e-9)
        assert d_ved["b_w"] == pytest.approx(d_vz["b_w"], rel=1e-9)
        assert d_ved["V_Rd"] == pytest.approx(d_vz["V_Rd"], rel=1e-9)


class TestLoadCaseConvention:
    def test_V_Ed_input_is_first_class_no_warning(self):
        """V_Ed= is a first-class input (→ Vz_Ed major) and must NOT warn."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            lc = LoadCase(V_Ed=150.0, M_Ed=50.0, N_Ed=100.0)
        assert not [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert lc.Vz_Ed == 150.0  # mapped to major axis
        assert lc.Vy_Ed == 0.0
        assert lc.My_Ed == 50.0  # M_Ed → My_Ed (major)

    def test_explicit_components_take_precedence_over_V_Ed(self):
        """Explicit shear components win; V_Ed is only a fallback."""
        lc = LoadCase(V_Ed=999.0, Vy_Ed=30.0, Vz_Ed=40.0)
        assert lc.Vy_Ed == 30.0
        assert lc.Vz_Ed == 40.0

    def test_V_Ed_computed_resultant_is_hypot(self):
        """The V_Ed read-back is the resultant of both components."""
        lc = LoadCase(Vy_Ed=30.0, Vz_Ed=40.0)
        assert lc.V_Ed == pytest.approx(hypot(30.0, 40.0))  # 50.0

    def test_M_Ed_alias_no_warning(self):
        """M_Ed= maps to My_Ed (major) with no DeprecationWarning."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            lc = LoadCase(M_Ed=75.0)
        assert not [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert lc.My_Ed == 75.0
        assert lc.M_Ed == 75.0
