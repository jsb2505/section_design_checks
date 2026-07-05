"""
Unit tests for EC2 shear utility helpers.
"""

from __future__ import annotations

from math import sqrt
from types import SimpleNamespace

import pytest
from shapely.geometry import Point, Polygon

from section_design_checks.reinforced_concrete.code_checks.ec2_2004 import shear_utils
from section_design_checks.reinforced_concrete.geometry import (
    create_i_beam_section,
    create_linear_rebar_layer,
    create_rectangular_section,
)
from section_design_checks.reinforced_concrete.materials import Rebar, ShearRebar
from section_design_checks.reinforced_concrete.ndp import ndp_override


def _make_shear_rebar(angle: float = 90.0) -> ShearRebar:
    return ShearRebar(
        diameter=10.0,
        grade="B500B",
        link_spacing=200.0,
        n_legs=2,
        angle=angle,
    )


class TestCalculateTensionShift:
    """Tests for TestCalculateTensionShift."""
    def test_without_shear_reinforcement_uses_al_equals_d(self):
        """Test without shear reinforcement uses al equals d."""
        out = shear_utils.calculate_tension_shift(
            M_Ed=100.0,
            V_Ed=-50.0,
            z=450.0,
            d=500.0,
        )

        assert out.cot_theta is None
        assert out.shift_distance_a_l == pytest.approx(500.0, rel=1e-12)
        assert out.M_add == pytest.approx(25.0, rel=1e-12)
        assert out.M_design == pytest.approx(125.0, rel=1e-12)
        assert out.capped_by_M_cap is False

    def test_cap_applies_to_magnitude_and_restores_sign(self):
        """Test cap applies to magnitude and restores sign."""
        out = shear_utils.calculate_tension_shift(
            M_Ed=-120.0,
            V_Ed=100.0,
            z=450.0,
            d=500.0,
            M_cap=150.0,
        )

        assert out.M_add == pytest.approx(50.0, rel=1e-12)
        assert out.M_design == pytest.approx(-150.0, rel=1e-12)
        assert out.capped_by_M_cap is True

    def test_with_shear_reinforcement_requires_parameters_unless_override(self):
        """Test with shear reinforcement requires parameters unless override."""
        links = _make_shear_rebar()
        with pytest.raises(ValueError, match="required"):
            shear_utils.calculate_tension_shift(
                M_Ed=80.0,
                V_Ed=60.0,
                z=450.0,
                d=500.0,
                shear_reinforcement=links,
            )

    def test_with_override_clamps_cot_theta_and_uses_stirrup_angle(self):
        """Test with override clamps cot theta and uses stirrup angle."""
        links = _make_shear_rebar(angle=90.0)
        with ndp_override(cot_theta_lower_lim=1.0, cot_theta_upper_lim=2.5):
            out = shear_utils.calculate_tension_shift(
                M_Ed=90.0,
                V_Ed=100.0,
                z=450.0,
                d=500.0,
                shear_reinforcement=links,
                cot_theta_override=3.0,  # clamped to 2.5
            )

        assert out.cot_theta == pytest.approx(2.5, rel=1e-12)
        assert out.shift_distance_a_l == pytest.approx(562.5, rel=1e-12)
        assert out.M_add == pytest.approx(56.25, rel=1e-12)

    def test_with_v_rd_s_path_uses_vrds_solver(self, monkeypatch):
        """Test with v rd s path uses vrds solver."""
        links = _make_shear_rebar(angle=90.0)
        calls = {"count": 0}

        def _fake_vrds(**kwargs):
            calls["count"] += 1
            return 1.6

        monkeypatch.setattr(shear_utils, "find_cot_theta_for_V_Ed_from_V_Rd_s", _fake_vrds)

        out = shear_utils.calculate_tension_shift(
            M_Ed=100.0,
            V_Ed=100.0,
            z=450.0,
            d=500.0,
            b_w=300.0,
            f_cd=20.0,
            f_ck=30.0,
            shear_reinforcement=links,
            use_v_rd_s_for_cot_theta=True,
        )

        assert calls["count"] == 1
        assert out.cot_theta == pytest.approx(1.6, rel=1e-12)
        assert out.shift_distance_a_l == pytest.approx(360.0, rel=1e-12)

    def test_with_v_rd_max_path_uses_alpha_cw_and_nu_factor(self, monkeypatch):
        """Test with v rd max path uses alpha cw and nu factor."""
        links = _make_shear_rebar(angle=90.0)
        seen = {"sigma": None, "K": None}

        def _fake_alpha_cw(*, f_cd, sigma_cp, use_sigma_cp_for_alpha_cw):
            seen["sigma"] = sigma_cp
            return 1.2

        def _fake_nu(*, f_ck):
            return 0.6

        def _fake_cot(*, V_Ed, K, link_angle_degrees):
            seen["K"] = K
            return 1.5

        monkeypatch.setattr(shear_utils, "find_alpha_cw", _fake_alpha_cw)
        monkeypatch.setattr(shear_utils, "find_nu_factor", _fake_nu)
        monkeypatch.setattr(shear_utils, "find_cot_theta_for_V_Ed_from_V_Rd_max", _fake_cot)

        out = shear_utils.calculate_tension_shift(
            M_Ed=80.0,
            V_Ed=70.0,
            z=450.0,
            d=500.0,
            b_w=300.0,
            f_cd=20.0,
            f_ck=30.0,
            sigma_cp=2.0,
            use_sigma_cp_for_alpha_cw=True,
            shear_reinforcement=links,
        )

        assert seen["sigma"] == pytest.approx(2.0, rel=1e-12)
        assert seen["K"] == pytest.approx(1.2 * 300.0 * 450.0 * 0.6 * 20.0, rel=1e-12)
        assert out.cot_theta == pytest.approx(1.5, rel=1e-12)


class TestSectionBreadthAndRhoL:
    """Tests for TestSectionBreadthAndRhoL."""
    def test_calculate_section_breadth_rectangular_section(self):
        """Test calculate section breadth rectangular section."""
        section = create_rectangular_section(width=300.0, height=500.0)
        b_w = shear_utils.calculate_section_breadth(section=section, n_slices=20)
        assert b_w == pytest.approx(300.0, rel=1e-12)

    def test_calculate_section_breadth_degenerate_height(self):
        """Test calculate section breadth degenerate height."""
        outline = SimpleNamespace(bounds=(0.0, 1.0, 5.0, 1.0))
        section = SimpleNamespace(outline=outline)
        assert shear_utils.calculate_section_breadth(section=section) == pytest.approx(5.0, rel=1e-12)

    def test_calculate_section_breadth_hollow_section_hits_multiline_branch(self):
        """Test calculate section breadth hollow section hits multiline branch."""
        outline = Polygon(
            shell=[(0, 0), (10, 0), (10, 10), (0, 10)],
            holes=[[(4, 2), (6, 2), (6, 8), (4, 8)]],
        )
        section = SimpleNamespace(outline=outline)

        b_w = shear_utils.calculate_section_breadth(section=section, n_slices=10)
        assert b_w == pytest.approx(8.0, rel=1e-12)

    def test_calculate_section_breadth_skips_empty_intersections(self):
        """Test calculate section breadth skips empty intersections."""
        class _EmptyGeom:
            is_empty = True

        class _Outline:
            bounds = (0.0, 0.0, 10.0, 10.0)

            def intersection(self, _line):
                return _EmptyGeom()

        section = SimpleNamespace(outline=_Outline())
        out = shear_utils.calculate_section_breadth(section=section, n_slices=2)
        assert out == pytest.approx(10.0, rel=1e-12)

    def test_calculate_section_breadth_skips_point_tangency_intersections(self):
        """Test calculate section breadth skips point tangency intersections."""
        class _Outline:
            bounds = (0.0, 0.0, 10.0, 10.0)

            def intersection(self, _line):
                return Point(5.0, 5.0)

        section = SimpleNamespace(outline=_Outline())
        out = shear_utils.calculate_section_breadth(section=section, n_slices=2)
        assert out == pytest.approx(10.0, rel=1e-12)

    def test_calculate_section_breadth_average_policy_for_i_beam(self):
        """Average policy can target the web region around mid-depth."""
        section = create_i_beam_section(
            b_f_top=500.0,
            h_f_top=100.0,
            b_f_bot=500.0,
            h_f_bot=100.0,
            b_w=200.0,
            h_w=400.0,
        )

        b_min = shear_utils.calculate_section_breadth(
            section=section,
            n_slices=401,
            policy="minimum",
        )
        b_avg_mid = shear_utils.calculate_section_breadth(
            section=section,
            n_slices=401,
            policy="average",
            average_height_ratio=0.5,
        )
        b_avg_full = shear_utils.calculate_section_breadth(
            section=section,
            n_slices=401,
            policy="average",
            average_height_ratio=1.0,
        )

        assert b_min == pytest.approx(200.0, rel=1e-12)
        assert b_avg_mid == pytest.approx(200.0, abs=1.0)
        assert b_avg_full == pytest.approx(300.0, abs=3.0)
        assert b_avg_full > b_avg_mid

    def test_calculate_section_breadth_supports_non_vertical_shear_direction(self):
        """Horizontal shear direction slices vertically and returns section height."""
        section = create_rectangular_section(width=300.0, height=500.0)
        b_w = shear_utils.calculate_section_breadth(
            section=section,
            n_slices=40,
            policy="minimum",
            shear_direction=(1.0, 0.0),
        )
        assert b_w == pytest.approx(500.0, rel=1e-12)

    def test_calculate_section_breadth_policy_input_validation(self):
        """Invalid breadth policy inputs raise clear errors."""
        section = create_rectangular_section(width=300.0, height=500.0)

        with pytest.raises(ValueError, match="policy must be one of"):
            shear_utils.calculate_section_breadth(section=section, policy="median")  # type: ignore[arg-type]

        with pytest.raises(ValueError, match="average_height_ratio"):
            shear_utils.calculate_section_breadth(
                section=section,
                policy="average",
                average_height_ratio=0.0,
            )

        with pytest.raises(ValueError, match="non-zero vector"):
            shear_utils.calculate_section_breadth(
                section=section,
                shear_direction=(0.0, 0.0),
            )

    def test_find_rho_l_invalid_geometry_returns_zero(self):
        """Test find rho l invalid geometry returns zero."""
        outline = SimpleNamespace(bounds=(0.0, 0.0, 10.0, 0.0))
        section = SimpleNamespace(outline=outline, rebar_groups=[])

        assert shear_utils.find_rho_l_from_strains(
            section=section,
            b_w=300.0,
            d=500.0,
            eps_top=0.001,
            eps_bottom=-0.001,
        ) == pytest.approx(0.0, rel=1e-12)

        assert shear_utils.find_rho_l_from_strains(
            section=section,
            b_w=0.0,
            d=500.0,
            eps_top=0.001,
            eps_bottom=-0.001,
        ) == pytest.approx(0.0, rel=1e-12)

    def test_find_rho_l_counts_only_tension_bars_and_applies_cap(self):
        """Test find rho l counts only tension bars and applies cap."""
        section = create_rectangular_section(width=300.0, height=500.0)
        rebar = Rebar(diameter=20.0, grade="B500B")
        bottom = create_linear_rebar_layer(
            rebar=rebar,
            n_bars=2,
            start_point=(60.0, 50.0),
            end_point=(240.0, 50.0),
            layer_name="bottom",
        )
        top = create_linear_rebar_layer(
            rebar=rebar,
            n_bars=2,
            start_point=(60.0, 450.0),
            end_point=(240.0, 450.0),
            layer_name="top",
        )
        section.add_rebar_group(bottom)
        section.add_rebar_group(top)

        rho = shear_utils.find_rho_l_from_strains(
            section=section,
            b_w=300.0,
            d=450.0,
            eps_top=0.001,
            eps_bottom=-0.001,
        )
        expected = (2.0 * rebar.area) / (300.0 * 450.0)
        assert rho == pytest.approx(expected, rel=1e-12)

        rho_capped = shear_utils.find_rho_l_from_strains(
            section=section,
            b_w=20.0,
            d=20.0,
            eps_top=0.001,
            eps_bottom=-0.001,
            rho_l_max=0.02,
        )
        assert rho_capped == pytest.approx(0.02, rel=1e-12)

    def test_find_rho_l_returns_zero_when_no_bars_are_in_tension(self):
        """Test find rho l returns zero when no bars are in tension."""
        section = create_rectangular_section(width=300.0, height=500.0)
        top = create_linear_rebar_layer(
            rebar=Rebar(diameter=20.0, grade="B500B"),
            n_bars=2,
            start_point=(60.0, 450.0),
            end_point=(240.0, 450.0),
            layer_name="top",
        )
        section.add_rebar_group(top)

        rho = shear_utils.find_rho_l_from_strains(
            section=section,
            b_w=300.0,
            d=450.0,
            eps_top=0.001,
            eps_bottom=0.0005,
        )
        assert rho == pytest.approx(0.0, rel=1e-12)


class TestSpacingDelegation:
    """Tests for TestSpacingDelegation."""
    def test_find_max_allowable_link_spacing_delegates_to_ndp_callable(self, monkeypatch):
        """Test find max allowable link spacing delegates to ndp callable."""
        captured = {}

        def _fake_get_ndp_callable(name):
            assert name == "max_link_spacing"

            def _fn(**kwargs):
                captured.update(kwargs)
                return 123.4

            return _fn

        monkeypatch.setattr(shear_utils, "get_ndp_callable", _fake_get_ndp_callable)

        out = shear_utils.find_max_allowable_link_spacing(
            effective_depth=500,
            section_depth=600,
            f_ck=30,
            V_Ed=120,
            V_Rd_max=300,
            V_Rd_c=None,
            link_angle_degrees=90,
        )

        assert out == pytest.approx(123.4, rel=1e-12)
        assert captured["V_Rd_c"] is None
        assert captured["effective_depth"] == pytest.approx(500.0, rel=1e-12)

    def test_find_max_allowable_leg_spacing_delegates_to_ndp_callable(self, monkeypatch):
        """Test find max allowable leg spacing delegates to ndp callable."""
        captured = {}

        def _fake_get_ndp_callable(name):
            assert name == "max_leg_spacing"

            def _fn(**kwargs):
                captured.update(kwargs)
                return 222.0

            return _fn

        monkeypatch.setattr(shear_utils, "get_ndp_callable", _fake_get_ndp_callable)

        out = shear_utils.find_max_allowable_leg_spacing(
            effective_depth=500,
            section_depth=600,
            f_ck=30,
            V_Ed=120,
            V_Rd_max=300,
            V_Rd_c=80,
            link_angle_degrees=60,
        )

        assert out == pytest.approx(222.0, rel=1e-12)
        assert captured["V_Rd_c"] == pytest.approx(80.0, rel=1e-12)
        assert captured["link_angle_degrees"] == pytest.approx(60.0, rel=1e-12)


class TestCotThetaHelpers:
    """Tests for TestCotThetaHelpers."""
    def test_find_cot_theta_from_vrdmax_invalid_inputs_return_cot_min(self):
        """Test find cot theta from vrdmax invalid inputs return cot min."""
        assert shear_utils.find_cot_theta_for_V_Ed_from_V_Rd_max(
            V_Ed=0.0,
            K=10.0,
            cot_min=1.1,
            cot_max=2.2,
        ) == pytest.approx(1.1, rel=1e-12)

        assert shear_utils.find_cot_theta_for_V_Ed_from_V_Rd_max(
            V_Ed=100.0,
            K=0.0,
            cot_min=1.1,
            cot_max=2.2,
        ) == pytest.approx(1.1, rel=1e-12)

    def test_find_cot_theta_from_vrdmax_negative_discriminant_returns_cot_min(self):
        """Test find cot theta from vrdmax negative discriminant returns cot min."""
        out = shear_utils.find_cot_theta_for_V_Ed_from_V_Rd_max(
            V_Ed=100.0,
            K=10.0,
            link_angle_degrees=90.0,
            cot_min=1.0,
            cot_max=2.5,
        )
        assert out == pytest.approx(1.0, rel=1e-12)

    def test_find_cot_theta_from_vrdmax_clamps_to_upper_bound(self):
        """Test find cot theta from vrdmax clamps to upper bound."""
        out = shear_utils.find_cot_theta_for_V_Ed_from_V_Rd_max(
            V_Ed=100.0,
            K=500_000.0,
            link_angle_degrees=90.0,
            cot_min=1.0,
            cot_max=2.5,
        )
        assert out == pytest.approx(2.5, rel=1e-12)

    def test_find_cot_theta_from_vrdmax_non_vertical_links_uses_cot_alpha(self):
        """Test find cot theta from vrdmax non vertical links uses cot alpha."""
        out = shear_utils.find_cot_theta_for_V_Ed_from_V_Rd_max(
            V_Ed=100.0,
            K=500_000.0,
            link_angle_degrees=45.0,
            cot_min=1.0,
            cot_max=2.5,
        )
        assert 1.0 <= out <= 2.5

    def test_find_cot_theta_from_vrdmax_tiny_negative_discriminant_is_clipped_to_zero(self):
        """Test find cot theta from vrdmax tiny negative discriminant is clipped to zero."""
        out = shear_utils.find_cot_theta_for_V_Ed_from_V_Rd_max(
            V_Ed=1e-6,  # 0.001 N after conversion
            K=0.002 * (1.0 - 1e-16),
            link_angle_degrees=90.0,
            cot_min=1.0,
            cot_max=2.5,
        )
        assert out == pytest.approx(1.0, rel=1e-12)

    def test_find_cot_theta_from_vrdmax_returns_cot_min_when_roots_not_finite(self, monkeypatch):
        """Test find cot theta from vrdmax returns cot min when roots not finite."""
        monkeypatch.setattr(shear_utils, "sqrt", lambda _x: float("nan"))
        out = shear_utils.find_cot_theta_for_V_Ed_from_V_Rd_max(
            V_Ed=100.0,
            K=500_000.0,
            link_angle_degrees=90.0,
            cot_min=1.2,
            cot_max=2.5,
        )
        assert out == pytest.approx(1.2, rel=1e-12)

    def test_find_cot_theta_from_vrds_guards_and_clamps(self):
        """Test find cot theta from vrds guards and clamps."""
        assert shear_utils.find_cot_theta_for_V_Ed_from_V_Rd_s(
            V_Ed=100.0,
            A_sw_over_s=0.0,
            z=450.0,
            f_ywd=435.0,
            cot_min=1.2,
            cot_max=2.2,
        ) == pytest.approx(1.2, rel=1e-12)

        assert shear_utils.find_cot_theta_for_V_Ed_from_V_Rd_s(
            V_Ed=100.0,
            A_sw_over_s=float("inf"),
            z=450.0,
            f_ywd=435.0,
            cot_min=1.2,
            cot_max=2.2,
        ) == pytest.approx(1.2, rel=1e-12)

        high = shear_utils.find_cot_theta_for_V_Ed_from_V_Rd_s(
            V_Ed=200.0,
            A_sw_over_s=0.2,
            z=450.0,
            f_ywd=435.0,
            link_angle_degrees=90.0,
            cot_min=1.0,
            cot_max=2.5,
        )
        assert high == pytest.approx(2.5, rel=1e-12)

        low = shear_utils.find_cot_theta_for_V_Ed_from_V_Rd_s(
            V_Ed=10.0,
            A_sw_over_s=1.0,
            z=500.0,
            f_ywd=500.0,
            link_angle_degrees=45.0,
            cot_min=1.0,
            cot_max=3.0,
        )
        assert low == pytest.approx(1.0, rel=1e-12)

    def test_find_cot_theta_from_vrds_returns_cot_min_when_result_not_finite(self, monkeypatch):
        """Test find cot theta from vrds returns cot min when result not finite."""
        monkeypatch.setattr(shear_utils, "cot", lambda _x: float("inf"))
        out = shear_utils.find_cot_theta_for_V_Ed_from_V_Rd_s(
            V_Ed=100.0,
            A_sw_over_s=0.5,
            z=450.0,
            f_ywd=435.0,
            link_angle_degrees=45.0,
            cot_min=1.1,
            cot_max=2.5,
        )
        assert out == pytest.approx(1.1, rel=1e-12)

    def test_clamp_cot_theta_with_explicit_and_ndp_bounds(self):
        """Test clamp cot theta with explicit and ndp bounds."""
        assert shear_utils.clamp_cot_theta(0.8, cot_min=1.0, cot_max=2.0) == pytest.approx(1.0, rel=1e-12)
        assert shear_utils.clamp_cot_theta(2.5, cot_min=1.0, cot_max=2.0) == pytest.approx(2.0, rel=1e-12)

        with ndp_override(cot_theta_lower_lim=1.2, cot_theta_upper_lim=2.2):
            assert shear_utils.clamp_cot_theta(0.5) == pytest.approx(1.2, rel=1e-12)
            assert shear_utils.clamp_cot_theta(3.0) == pytest.approx(2.2, rel=1e-12)


class TestNdpWrappersAndDerivedValues:
    """Tests for TestNdpWrappersAndDerivedValues."""
    def test_alpha_cw_uses_sigma_flag(self, monkeypatch):
        """Test alpha cw uses sigma flag."""
        def _fake_get_ndp_callable(name):
            assert name == "alpha_cw"
            return lambda f_cd, sigma_cp: (10.0 * f_cd) + sigma_cp

        monkeypatch.setattr(shear_utils, "get_ndp_callable", _fake_get_ndp_callable)

        assert shear_utils.find_alpha_cw(20.0, 3.0, use_sigma_cp_for_alpha_cw=False) == pytest.approx(200.0, rel=1e-12)
        assert shear_utils.find_alpha_cw(20.0, 3.0, use_sigma_cp_for_alpha_cw=True) == pytest.approx(203.0, rel=1e-12)

    def test_nu_factor_wrappers_delegate_to_ndp(self, monkeypatch):
        """Test nu factor wrappers delegate to ndp."""
        def _fake_get_ndp_callable(name):
            if name == "nu_shear":
                return lambda f_ck: 0.5 + f_ck / 1000.0
            if name == "nu_1":
                return lambda f_ck, angle: 0.6 + f_ck / 2000.0 + angle / 1000.0
            if name == "nu_1_note_2":
                return lambda f_ck, angle: 0.7 + f_ck / 3000.0 + angle / 2000.0
            if name == "nu_torsion":
                return lambda f_ck: 0.4 + f_ck / 1500.0
            raise AssertionError(f"Unexpected NDP key: {name}")

        monkeypatch.setattr(shear_utils, "get_ndp_callable", _fake_get_ndp_callable)

        assert shear_utils.find_nu_factor(30.0) == pytest.approx(0.53, rel=1e-12)
        assert shear_utils.find_nu_1_factor(30.0, 60.0) == pytest.approx(0.675, rel=1e-12)
        assert shear_utils.find_nu_1_factor_note_2(30.0, 60.0) == pytest.approx(0.74, rel=1e-12)
        assert shear_utils.find_nu_factor_torsion(30.0) == pytest.approx(0.42, rel=1e-12)

    def test_find_k_factor_and_find_v_min(self, monkeypatch):
        """Test find k factor and find v min."""
        with pytest.raises(ValueError, match="must be > 0"):
            shear_utils.find_k_factor(0.0)

        assert shear_utils.find_k_factor(50.0) == pytest.approx(2.0, rel=1e-12)
        assert shear_utils.find_k_factor(800.0) == pytest.approx(1.5, rel=1e-12)

        def _fake_get_ndp_callable(name):
            assert name == "v_min_coefficient"
            return lambda d, gamma_c: 0.031 + d / 1_000_000.0 + gamma_c / 1000.0

        monkeypatch.setattr(shear_utils, "get_ndp_callable", _fake_get_ndp_callable)
        v_min = shear_utils.find_v_min(f_ck=30.0, k_factor=1.5, d=500.0, gamma_c=1.5)
        coeff = 0.031 + 500.0 / 1_000_000.0 + 1.5 / 1000.0
        assert v_min == pytest.approx(coeff * (1.5 ** 1.5) * sqrt(30.0), rel=1e-12)

    def test_sigma_cp_and_cap(self):
        """Test sigma cp and cap."""
        sigma = shear_utils.sigma_cp_from_N_and_area(N_Ed=1000.0, area=100_000.0)
        assert sigma == pytest.approx(10.0, rel=1e-12)

        assert shear_utils.cap_sigma_cp_upper(8.0, f_cd=20.0) == pytest.approx(4.0, rel=1e-12)
        assert shear_utils.cap_sigma_cp_upper(-2.0, f_cd=20.0) == pytest.approx(-2.0, rel=1e-12)

    def test_find_vrdc_cracked_uses_max_of_main_and_min_formula(self, monkeypatch):
        """Test find vrdc cracked uses max of main and min formula."""
        monkeypatch.setattr(
            shear_utils,
            "get_ndp",
            lambda name: {"c_rd_c_coefficient": 0.18, "k_1_shear": 0.15}[name],
        )
        monkeypatch.setattr(shear_utils, "find_k_factor", lambda d: 1.5)
        monkeypatch.setattr(shear_utils, "find_v_min", lambda f_ck, k_factor, d, gamma_c: 0.5)

        out = shear_utils.find_V_Rd_c_cracked(
            b_w=300.0,
            d=500.0,
            rho_l=0.01,
            sigma_cp=1.0,
            f_ck=30.0,
            gamma_c=1.5,
        )

        c_rd_c = 0.18 / 1.5
        v_main = (c_rd_c * 1.5 * ((100.0 * 0.01 * 30.0) ** (1.0 / 3.0)) + 0.15 * 1.0) * 300.0 * 500.0
        v_min = (0.5 + 0.15 * 1.0) * 300.0 * 500.0
        expected_kn = max(v_main, v_min) / 1000.0
        assert out == pytest.approx(expected_kn, rel=1e-12)

    def test_find_vrdc_cracked_floors_at_zero(self, monkeypatch):
        """Test find vrdc cracked floors at zero."""
        monkeypatch.setattr(
            shear_utils,
            "get_ndp",
            lambda name: {"c_rd_c_coefficient": 0.18, "k_1_shear": 0.15}[name],
        )
        monkeypatch.setattr(shear_utils, "find_k_factor", lambda d: 1.5)
        monkeypatch.setattr(shear_utils, "find_v_min", lambda f_ck, k_factor, d, gamma_c: 0.0)

        out = shear_utils.find_V_Rd_c_cracked(
            b_w=300.0,
            d=500.0,
            rho_l=0.0,
            sigma_cp=-10.0,
            f_ck=30.0,
            gamma_c=1.5,
        )
        assert out == pytest.approx(0.0, rel=1e-12)

    def test_find_vrdc_max_unreinforced(self, monkeypatch):
        """Test find vrdc max unreinforced."""
        monkeypatch.setattr(shear_utils, "find_nu_factor", lambda f_ck: 0.6)
        out = shear_utils.find_V_Rd_c_max_unreinforced(
            b_w=300.0,
            d=500.0,
            f_ck=30.0,
            f_cd=20.0,
        )
        assert out == pytest.approx(900.0, rel=1e-12)

    def test_find_minimum_ratio_of_shear_reinforcement_delegates_to_ndp(self, monkeypatch):
        """Test find minimum ratio of shear reinforcement delegates to ndp."""
        def _fake_get_ndp_callable(name):
            assert name == "rho_w_min"
            return lambda f_ck, f_yk, f_ctm: (f_ck + f_yk + f_ctm) / 10_000.0

        monkeypatch.setattr(shear_utils, "get_ndp_callable", _fake_get_ndp_callable)
        out = shear_utils.find_minimum_ratio_of_shear_reinforcement(
            f_ck=30.0,
            f_yk=500.0,
            f_ctm=2.9,
        )
        assert out == pytest.approx((30.0 + 500.0 + 2.9) / 10_000.0, rel=1e-12)
