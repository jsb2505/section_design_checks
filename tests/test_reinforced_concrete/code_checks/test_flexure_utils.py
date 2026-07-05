"""
Tests for shared flexure utility functions.
"""

from __future__ import annotations

import warnings

import pytest

from materials.reinforced_concrete.code_checks.ec2_2004 import flexure_utils
from materials.reinforced_concrete.constitutive import SteelModelType
from materials.reinforced_concrete.geometry import (
    create_linear_rebar_layer,
    create_rectangular_section,
)
from materials.reinforced_concrete.materials import Rebar
from materials.reinforced_concrete.ndp import ndp_override


def _make_section(*, include_top: bool = True, include_bottom: bool = True):
    section = create_rectangular_section(width=300.0, height=500.0)
    if include_bottom:
        bottom = create_linear_rebar_layer(
            rebar=Rebar(diameter=20, grade="B500B"),
            n_bars=3,
            start_point=(50.0, 50.0),
            end_point=(250.0, 50.0),
            layer_name="bottom",
        )
        section.add_rebar_group(bottom)
    if include_top:
        top = create_linear_rebar_layer(
            rebar=Rebar(diameter=16, grade="B500B"),
            n_bars=2,
            start_point=(70.0, 450.0),
            end_point=(230.0, 450.0),
            layer_name="top",
        )
        section.add_rebar_group(top)
    return section


class _FailingDiagram:
    def find_strains_for_MN(self, M_Ed: float, N_Ed: float):
        raise RuntimeError("solver failed")


class TestBasicGeometryAndStrainHelpers:
    """Tests for TestBasicGeometryAndStrainHelpers."""
    def test_section_height_breadth_and_modular_ratio(self):
        """Test section height breadth and modular ratio."""
        section = _make_section()
        assert flexure_utils.calculate_section_height(section) == pytest.approx(500.0, rel=1e-12)
        assert flexure_utils.calculate_section_breadth(section) == pytest.approx(300.0, rel=1e-12)
        assert flexure_utils.calculate_modular_ratio(E_s=200000.0, E_cm=30000.0) == pytest.approx(
            200000.0 / 30000.0,
            rel=1e-12,
        )
        with pytest.raises(ValueError, match="E_cm must be > 0"):
            flexure_utils.calculate_modular_ratio(E_s=200000.0, E_cm=0.0)

    def test_neutral_axis_depth_from_strains(self):
        """Test neutral axis depth from strains."""
        assert flexure_utils.calculate_neutral_axis_depth_from_strains(0.001, -0.001, 500.0) == pytest.approx(
            250.0,
            rel=1e-12,
        )
        assert flexure_utils.calculate_neutral_axis_depth_from_strains(0.001, 0.001, 500.0) is None
        assert flexure_utils.calculate_neutral_axis_depth_from_strains(0.001, 0.0005, 500.0) is None
        assert flexure_utils.calculate_neutral_axis_depth_from_strains(-0.001, -0.0005, 500.0) is None
        # Negative section height triggers the x<0 sanity branch.
        assert flexure_utils.calculate_neutral_axis_depth_from_strains(0.001, -0.001, -500.0) is None

    def test_compression_face_from_strains(self):
        """Test compression face from strains."""
        assert flexure_utils.calculate_compression_face_from_strains(0.001, -0.001) == "top"
        assert flexure_utils.calculate_compression_face_from_strains(-0.001, 0.001) == "bottom"
        assert flexure_utils.calculate_compression_face_from_strains(-0.001, -0.0001) is None


class TestEffectiveDepthHelpers:
    """Tests for TestEffectiveDepthHelpers."""
    def test_find_effective_depth_no_rebar_uses_fallback(self):
        """No rebar section uses ratio_of_h fallback (default 0.9h)."""
        section = create_rectangular_section(width=300.0, height=500.0)
        d = flexure_utils.find_effective_depth_for_flexure(
            section=section,
            diagram=None,
            M_Ed=100.0,
            N_Ed=0.0,
        )
        assert d == pytest.approx(0.9 * 500.0)

    def test_find_effective_depth_pure_moment_zero_returns_fallback(self):
        """M_Ed=0 → fallback policy (default 0.9h)."""
        section = _make_section(include_top=True, include_bottom=True)
        h = flexure_utils.calculate_section_height(section)
        d = flexure_utils.find_effective_depth_for_flexure(
            section=section,
            diagram=None,
            M_Ed=0.0,
            N_Ed=0.0,
        )
        assert d == pytest.approx(0.9 * h, rel=1e-12)

    def test_find_effective_depth_uses_strain_based_compression_face(self):
        """Test find effective depth uses strain based compression face."""
        section = _make_section(include_top=True, include_bottom=True)
        d_top = section.get_effective_depth(compression_face="top")
        d_bottom = section.get_effective_depth(compression_face="bottom")

        d1 = flexure_utils.find_effective_depth_for_flexure(
            section=section,
            diagram=None,
            M_Ed=100.0,
            N_Ed=0.0,
            eps_top=0.001,
            eps_bottom=-0.001,
        )
        d2 = flexure_utils.find_effective_depth_for_flexure(
            section=section,
            diagram=None,
            M_Ed=100.0,
            N_Ed=0.0,
            eps_top=-0.001,
            eps_bottom=0.001,
        )

        assert d1 == pytest.approx(d_top, rel=1e-12)
        assert d2 == pytest.approx(d_bottom, rel=1e-12)

    def test_find_effective_depth_fallback_branches(self):
        """Test find effective depth fallback branches."""
        section = _make_section(include_top=False, include_bottom=True)
        h = flexure_utils.calculate_section_height(section)
        expected_fallback = 0.9 * h

        # Missing strain state from failing solver -> fallback + warning.
        with pytest.warns(UserWarning, match="strain state unavailable"):
            d_fail_solver = flexure_utils.find_effective_depth_for_flexure(
                section=section,
                diagram=_FailingDiagram(),
                M_Ed=100.0,
                N_Ed=0.0,
            )
        assert d_fail_solver == pytest.approx(expected_fallback, rel=1e-12)

        # Both faces in tension -> fallback warning.
        with pytest.warns(UserWarning, match="no compression/tension split"):
            d_tension = flexure_utils.find_effective_depth_for_flexure(
                section=section,
                diagram=None,
                M_Ed=100.0,
                N_Ed=0.0,
                eps_top=-0.001,
                eps_bottom=-0.002,
            )
        assert d_tension == pytest.approx(expected_fallback, rel=1e-12)

        # Compression face selected as bottom but no top bars => fallback warning.
        with pytest.warns(UserWarning, match="no rebar in tension zone"):
            d_missing_face = flexure_utils.find_effective_depth_for_flexure(
                section=section,
                diagram=None,
                M_Ed=100.0,
                N_Ed=0.0,
                eps_top=-0.001,
                eps_bottom=0.001,
            )
        assert d_missing_face == pytest.approx(expected_fallback, rel=1e-12)

    def test_find_effective_depth_top_compression_fallback_when_d_top_missing(self):
        """Compression at top but no bottom rebar → fallback (0.9h)."""
        section = _make_section(include_top=True, include_bottom=False)
        h = flexure_utils.calculate_section_height(section)

        with pytest.warns(UserWarning, match="no rebar in tension zone"):
            d = flexure_utils.find_effective_depth_for_flexure(
                section=section,
                diagram=None,
                M_Ed=100.0,
                N_Ed=0.0,
                eps_top=0.001,
                eps_bottom=-0.001,
            )
        assert d == pytest.approx(0.9 * h, rel=1e-12)

    def test_find_effective_depth_can_suppress_fallback_warning(self):
        """Test find effective depth can suppress fallback warning."""
        section = _make_section(include_top=False, include_bottom=True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _ = flexure_utils.find_effective_depth_for_flexure(
                section=section,
                diagram=None,
                M_Ed=100.0,
                N_Ed=0.0,
                eps_top=-0.001,
                eps_bottom=-0.002,
                warn_on_fallback=False,
            )
        assert len(caught) == 0

    def test_find_mean_effective_depth(self):
        """Test find mean effective depth."""
        section = _make_section(include_top=True, include_bottom=True)
        d_bottom_tension = flexure_utils.find_mean_effective_depth(section, tension_face="bottom", zone_fraction=0.5)
        d_top_tension = flexure_utils.find_mean_effective_depth(section, tension_face="top", zone_fraction=0.5)
        assert d_bottom_tension == pytest.approx(section.get_effective_depth(compression_face="top", zone_fraction=0.5))
        assert d_top_tension == pytest.approx(section.get_effective_depth(compression_face="bottom", zone_fraction=0.5))


class TestRebarAndSteelFormulas:
    """Tests for TestRebarAndSteelFormulas."""
    def test_find_equivalent_diameter(self):
        """Test find equivalent diameter."""
        with pytest.raises(ValueError, match="No bars provided"):
            flexure_utils.find_equivalent_diameter([])

        assert flexure_utils.find_equivalent_diameter([(16.0, 3), (16.0, 2)]) == pytest.approx(16.0, rel=1e-12)
        assert flexure_utils.find_equivalent_diameter([(16.0, 4), (20.0, 2)]) == pytest.approx(
            (4 * 16.0**2 + 2 * 20.0**2) / (4 * 16.0 + 2 * 20.0),
            rel=1e-12,
        )
        with pytest.raises(ValueError, match="Total bar count is zero"):
            flexure_utils.find_equivalent_diameter([(16.0, 0), (20.0, 0)])

    def test_get_tension_rebars_from_strain_state(self):
        """Test get tension rebars from strain state."""
        section = _make_section(include_top=True, include_bottom=True)
        rebars = flexure_utils.get_tension_rebars_from_strain_state(
            section=section,
            eps_top=0.001,
            eps_bottom=-0.001,
        )
        # Bottom layer has 3 bars of 20 mm in this helper section.
        assert len(rebars) == 3
        assert all(r[0] == pytest.approx(20.0, rel=1e-12) for r in rebars)

    def test_rebar_characteristic_stress_from_strain(self):
        # Elastic region
        """Test rebar characteristic stress from strain."""
        s_el = flexure_utils.calculate_rebar_characteristic_stress_from_strain(
            strain=0.001,
            steel_model_type=SteelModelType.INCLINED,
            E_s=200000.0,
            f_yk=500.0,
            k=1.08,
            epsilon_uk=0.05,
        )
        assert s_el == pytest.approx(200.0, rel=1e-12)

        # Horizontal post-yield
        s_h = flexure_utils.calculate_rebar_characteristic_stress_from_strain(
            strain=-0.01,
            steel_model_type=SteelModelType.HORIZONTAL,
            E_s=200000.0,
            f_yk=500.0,
        )
        assert s_h == pytest.approx(-500.0, rel=1e-12)

        # Inclined hardening within ultimate
        s_i = flexure_utils.calculate_rebar_characteristic_stress_from_strain(
            strain=0.01,
            steel_model_type=SteelModelType.INCLINED,
            E_s=200000.0,
            f_yk=500.0,
            k=1.08,
            epsilon_uk=0.05,
        )
        assert 500.0 < s_i < 540.0

        # Beyond ultimate caps at k*f_yk
        s_cap = flexure_utils.calculate_rebar_characteristic_stress_from_strain(
            strain=0.1,
            steel_model_type=SteelModelType.INCLINED,
            E_s=200000.0,
            f_yk=500.0,
            k=1.08,
            epsilon_uk=0.05,
        )
        assert s_cap == pytest.approx(540.0, rel=1e-12)

        with pytest.raises(ValueError, match="Unsupported steel model type"):
            flexure_utils.calculate_rebar_characteristic_stress_from_strain(
                strain=0.01,  # force post-yield branch where model type is checked
                steel_model_type="bad-model",  # type: ignore[arg-type]
            )

    def test_find_area_of_steel_minimum_and_maximum(self):
        """Test find area of steel minimum and maximum."""
        with ndp_override(as_min_flexural_ratio=lambda f_ctm, f_yk: 0.002):
            A_min = flexure_utils.find_area_of_steel_minimum(b=300.0, d=450.0, f_ctm=2.9, f_yk=500.0)
        assert A_min == pytest.approx(0.002 * 300.0 * 450.0, rel=1e-12)

        with ndp_override(as_max_flexural_ratio=lambda section_area: 0.04):
            A_max = flexure_utils.find_area_of_steel_maximum(section_area=150000.0)
        assert A_max == pytest.approx(0.04 * 150000.0, rel=1e-12)

        with pytest.raises(ValueError, match="b must be > 0"):
            flexure_utils.find_area_of_steel_minimum(b=0.0, d=450.0, f_ctm=2.9, f_yk=500.0)
        with pytest.raises(ValueError, match="d must be > 0"):
            flexure_utils.find_area_of_steel_minimum(b=300.0, d=0.0, f_ctm=2.9, f_yk=500.0)
        with pytest.raises(ValueError, match="f_ctm must be >= 0"):
            flexure_utils.find_area_of_steel_minimum(b=300.0, d=450.0, f_ctm=-1.0, f_yk=500.0)
        with pytest.raises(ValueError, match="f_yk must be > 0"):
            flexure_utils.find_area_of_steel_minimum(b=300.0, d=450.0, f_ctm=2.9, f_yk=0.0)
        with pytest.raises(ValueError, match="section_area must be >= 0"):
            flexure_utils.find_area_of_steel_maximum(section_area=-1.0)
