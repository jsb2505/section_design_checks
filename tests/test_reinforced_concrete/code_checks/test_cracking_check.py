"""
Tests for CrackingCheck helper methods and branch behavior.
"""

from __future__ import annotations

import math
import sys
import types

import numpy as np
import pytest

from materials.core.units import ForceUnit, from_kn
from materials.reinforced_concrete.code_checks.base_check import CheckStatus
from materials.reinforced_concrete.code_checks.ec2_2004.cracking_check import (
    CrackingCheck,
    CrackingResult,
    LoadDuration,
)
from materials.reinforced_concrete.code_checks.ec2_2004.stress_limits_check import (
    compute_nonlinear_creep_coefficient,
)
from materials.reinforced_concrete.geometry import (
    create_linear_rebar_layer,
    create_rectangular_section,
)
from materials.reinforced_concrete.materials import Rebar
from materials.reinforced_concrete.ndp import ndp_override


def _make_section(
    *,
    width: float = 300.0,
    height: float = 500.0,
    bottom_E_s: float = 200000.0,
    top_E_s: float = 200000.0,
    include_bottom: bool = True,
    include_top: bool = True,
):
    section = create_rectangular_section(width=width, height=height)

    if include_bottom:
        bottom = create_linear_rebar_layer(
            rebar=Rebar(diameter=20, grade="B500B", E_s=bottom_E_s),
            n_bars=3,
            start_point=(50.0, 50.0),
            end_point=(250.0, 50.0),
            layer_name="bottom",
        )
        section.add_rebar_group(bottom)

    if include_top:
        top = create_linear_rebar_layer(
            rebar=Rebar(diameter=16, grade="B500B", E_s=top_E_s),
            n_bars=2,
            start_point=(70.0, 450.0),
            end_point=(230.0, 450.0),
            layer_name="top",
        )
        section.add_rebar_group(top)

    return section


class _FakeDiagram:
    def __init__(
        self,
        *,
        eps_top: float,
        eps_bottom: float,
        raise_on_find: bool = False,
        fibre_mat: np.ndarray | None = None,
        forces: np.ndarray | None = None,
        areas: np.ndarray | None = None,
    ):
        self.eps_top = eps_top
        self.eps_bottom = eps_bottom
        self.raise_on_find = raise_on_find
        self._fibre_mat = fibre_mat if fibre_mat is not None else np.array(["concrete", "concrete"])
        self._forces = forces if forces is not None else np.array([20.0, 10.0])
        self._areas = areas if areas is not None else np.array([2.0, 2.0])

    def find_strains_for_MN(self, *args, **kwargs):
        if self.raise_on_find:
            raise ValueError("outside diagram")
        return self.eps_top, self.eps_bottom

    def find_strain_state_for_MN(self, *args, **kwargs):
        return None

    def get_fibre_forces_from_end_strains(self, eps_top: float, eps_bottom: float):
        y = np.zeros_like(self._forces, dtype=float)
        return self._forces, y, self._areas


class TestCrackingHelpers:
    """Tests for TestCrackingHelpers."""
    def test_load_duration_kt_values(self):
        """Test load duration kt values."""
        assert LoadDuration.SHORT_TERM.k_t == pytest.approx(0.6, rel=1e-12)
        assert LoadDuration.LONG_TERM.k_t == pytest.approx(0.4, rel=1e-12)

    def test_basic_properties(self, concrete_c30):
        """Test basic properties."""
        check = CrackingCheck(section=_make_section(), concrete=concrete_c30)
        assert check.height == pytest.approx(500.0, rel=1e-12)
        assert check.breadth == pytest.approx(300.0, rel=1e-12)
        assert check.k_t == pytest.approx(check.load_duration.k_t, rel=1e-12)
        assert check.effective_modulus_ratio == pytest.approx(1.0 + check.creep_coefficient, rel=1e-12)

    def test_find_h_c_ef_variants(self, concrete_c30):
        """Test find h c ef variants."""
        check = CrackingCheck(section=_make_section(), concrete=concrete_c30)
        h_c_no_x = check.find_h_c_ef(d=450.0, x=None)
        h_c_with_x = check.find_h_c_ef(d=450.0, x=200.0)
        assert h_c_no_x == pytest.approx(125.0, rel=1e-12)
        assert h_c_with_x == pytest.approx(100.0, rel=1e-12)

    def test_find_h_c_ef_tension_member(self, concrete_c30):
        """Test find h c ef tension member."""
        check = CrackingCheck(section=_make_section(), concrete=concrete_c30)
        top, bottom = check.find_h_c_ef_tension_member(d_top=30.0, d_bottom=40.0)
        assert top == pytest.approx(75.0, rel=1e-12)
        assert bottom == pytest.approx(100.0, rel=1e-12)

    def test_find_rho_p_eff_and_invalid_area(self, concrete_c30):
        """Test find rho p eff and invalid area."""
        check = CrackingCheck(section=_make_section(), concrete=concrete_c30)
        rho = check.find_rho_p_eff(A_s_tension=1000.0, h_c_ef=100.0)
        assert rho == pytest.approx(1000.0 / (100.0 * 300.0), rel=1e-12)

        with pytest.raises(ValueError, match="A_c,eff must be > 0"):
            check.find_rho_p_eff(A_s_tension=1000.0, h_c_ef=0.0)

    def test_find_k_2_branches(self, concrete_c30):
        """Test find k 2 branches."""
        check = CrackingCheck(section=_make_section(), concrete=concrete_c30)
        assert check.find_k_2(0.001, -0.001) == pytest.approx(0.5, rel=1e-12)
        assert check.find_k_2(-0.001, -0.001) == pytest.approx(1.0, rel=1e-12)
        assert check.find_k_2(-0.001, -0.0002) == pytest.approx(0.6, rel=1e-12)
        assert check.find_k_2(-1e-13, -5e-14) == pytest.approx(0.5, rel=1e-12)

    def test_find_maximum_crack_spacing_standard_and_eq714(self, concrete_c30):
        """Test find maximum crack spacing standard and eq714."""
        check = CrackingCheck(section=_make_section(), concrete=concrete_c30, is_high_bond_bar=True)

        cover = 35.0
        phi_eq = 16.0
        rho_p_eff = 0.01
        k_2 = 0.5

        k_1 = check.find_k_1(k_2)
        expected_standard = check.k_3 * cover + (k_1 * k_2 * check.k_4 * phi_eq / rho_p_eff)
        actual_standard = check.find_maximum_crack_spacing(
            cover=cover,
            phi_eq=phi_eq,
            rho_p_eff=rho_p_eff,
            k_2=k_2,
        )
        assert actual_standard == pytest.approx(expected_standard, rel=1e-12)

        # Eq. 7.14 trigger by spacing
        x = 200.0
        expected_714 = 1.3 * (check.height - x)
        actual_triggered = check.find_maximum_crack_spacing(
            cover=cover,
            phi_eq=phi_eq,
            rho_p_eff=rho_p_eff,
            k_2=k_2,
            x=x,
            has_tension_reinforcement=True,
            bar_spacing=1000.0,
        )
        assert actual_triggered == pytest.approx(max(expected_standard, expected_714), rel=1e-12)

        # rho <= 0 with no Eq. 7.14 trigger returns inf
        no_rho = check.find_maximum_crack_spacing(
            cover=cover,
            phi_eq=phi_eq,
            rho_p_eff=0.0,
            k_2=k_2,
            has_tension_reinforcement=True,
            bar_spacing=0.0,
        )
        assert math.isinf(no_rho)

    def test_find_maximum_crack_spacing_with_ndp_cap(self, concrete_c30):
        """Test find maximum crack spacing with ndp cap."""
        check = CrackingCheck(section=_make_section(), concrete=concrete_c30)

        with ndp_override(s_r_max_lim=lambda sigma_s, diameter, f_ct_eff: 100.0):
            s_r = check.find_maximum_crack_spacing(
                cover=40.0,
                phi_eq=20.0,
                rho_p_eff=0.01,
                k_2=0.5,
                sigma_s=300.0,
            )
        assert s_r <= 100.0

    def test_find_strain_difference_branches(self, concrete_c30):
        """Test find strain difference branches."""
        check = CrackingCheck(section=_make_section(), concrete=concrete_c30)

        assert check.find_strain_difference(sigma_s=0.0, rho_p_eff=0.01, E_s=200000.0) == 0.0

        eps = check.find_strain_difference(sigma_s=200.0, rho_p_eff=0.01, E_s=200000.0)
        assert eps > 0.0

        eps_no_rho = check.find_strain_difference(sigma_s=200.0, rho_p_eff=0.0, E_s=200000.0)
        assert eps_no_rho == pytest.approx(200.0 / 200000.0, rel=1e-12)

        eps_min_bound = check.find_strain_difference(sigma_s=20.0, rho_p_eff=0.02, E_s=200000.0)
        assert eps_min_bound == pytest.approx(0.6 * 20.0 / 200000.0, rel=1e-12)

    def test_calculate_crack_width(self, concrete_c30):
        """Test calculate crack width."""
        check = CrackingCheck(section=_make_section(), concrete=concrete_c30)
        w_k = check.calculate_crack_width(s_r_max=150.0, eps_sm_minus_eps_cm=0.0012)
        assert w_k == pytest.approx(0.18, rel=1e-12)

    def test_find_k_c_branches(self, concrete_c30):
        """Test find k c branches."""
        check = CrackingCheck(section=_make_section(), concrete=concrete_c30)

        assert check.find_k_c(is_in_bending=False) == pytest.approx(1.0, rel=1e-12)

        for N_Ed in (300.0, -300.0):
            h = check.height
            h_star = min(h, 1000.0)
            k_1 = 1.5 if N_Ed >= 0 else (2.0 * h_star) / (3.0 * h)
            A_eff = check.section.get_transformed_area(check.concrete.E_cm)
            sigma_c = from_kn(N_Ed, ForceUnit.N) / A_eff
            expected = 0.4 * (1.0 - sigma_c / (k_1 * (h / h_star) * check.concrete.f_ctm))
            expected = min(1.0, max(0.0, expected))
            assert check.find_k_c(N_Ed=N_Ed, is_in_bending=True) == pytest.approx(expected, rel=1e-12)

    def test_find_minimum_crack_reinforcement_and_ndp_min(self, concrete_c30):
        """Test find minimum crack reinforcement and ndp min."""
        check = CrackingCheck(section=_make_section(width=300.0, height=500.0), concrete=concrete_c30)
        k_c = 0.4
        steel_stress = 500.0

        base = check.find_minimum_crack_reinforcement(steel_stress=steel_stress, k_c=k_c, N_Ed=0.0)
        A_ct = 0.5 * check.height * check.breadth
        expected_base = k_c * 1.0 * check.concrete.f_ctm * A_ct / steel_stress
        assert base == pytest.approx(expected_base, rel=1e-12)

        with ndp_override(f_ct_eff_min=10.0):
            boosted = check.find_minimum_crack_reinforcement(
                steel_stress=steel_stress,
                k_c=k_c,
                N_Ed=0.0,
            )
        expected_boosted = k_c * 1.0 * 10.0 * A_ct / steel_stress
        assert boosted == pytest.approx(expected_boosted, rel=1e-12)
        assert boosted > base

    def test_tension_rebar_info_and_filters(self, concrete_c30):
        """Test tension rebar info and filters."""
        check = CrackingCheck(section=_make_section(), concrete=concrete_c30)

        # Bending: top compression, bottom tension -> only bottom bars
        A_s_bottom, cover_bottom, bars_bottom = check._get_tension_rebar_info(
            eps_top=0.001,
            eps_bottom=-0.001,
        )
        expected_area_bottom = 3.0 * Rebar(diameter=20, grade="B500B").area
        assert A_s_bottom == pytest.approx(expected_area_bottom, rel=1e-12)
        assert cover_bottom == pytest.approx(40.0, rel=1e-12)
        assert bars_bottom == [(20.0, 3)]

        # Net tension + face filter: pick top bars only
        A_s_top, cover_top, bars_top = check._get_tension_rebar_info(
            eps_top=-0.0005,
            eps_bottom=-0.0010,
            face="top",
        )
        expected_area_top = 2.0 * Rebar(diameter=16, grade="B500B").area
        assert A_s_top == pytest.approx(expected_area_top, rel=1e-12)
        assert cover_top == pytest.approx(42.0, rel=1e-12)
        assert bars_top == [(16.0, 2)]

        # h_c,ef limit excludes top bars (50 mm from top face)
        A_s_limited, _, _ = check._get_tension_rebar_info(
            eps_top=-0.0005,
            eps_bottom=-0.0010,
            face="top",
            h_c_ef_limit=20.0,
        )
        assert A_s_limited == pytest.approx(0.0, abs=1e-12)

    def test_get_steel_stress_and_hcef_filter(self, concrete_c30):
        """Test get steel stress and hcef filter."""
        check = CrackingCheck(section=_make_section(), concrete=concrete_c30)
        sigma = check._get_steel_stress(eps_top=0.001, eps_bottom=-0.001)
        assert sigma > 0.0

        sigma_limited = check._get_steel_stress(
            eps_top=0.001,
            eps_bottom=-0.001,
            h_c_ef_limit=20.0,
        )
        assert sigma_limited == pytest.approx(0.0, abs=1e-12)

    def test_get_tension_zone_E_s_default_same_and_outermost(self, concrete_c30):
        """Test get tension zone E s default same and outermost."""
        empty = CrackingCheck(
            section=create_rectangular_section(width=300.0, height=500.0),
            concrete=concrete_c30,
        )
        assert empty._get_tension_zone_E_s(0.001, -0.001) == pytest.approx(200000.0, rel=1e-12)

        same_es = CrackingCheck(
            section=_make_section(bottom_E_s=210000.0, top_E_s=210000.0),
            concrete=concrete_c30,
        )
        assert same_es._get_tension_zone_E_s(0.001, -0.001) == pytest.approx(210000.0, rel=1e-12)

        diff_es = CrackingCheck(
            section=_make_section(bottom_E_s=190000.0, top_E_s=210000.0),
            concrete=concrete_c30,
        )
        # Compression at top -> bottom is tension face, should pick bottom E_s
        assert diff_es._get_tension_zone_E_s(0.001, -0.001) == pytest.approx(190000.0, rel=1e-12)

    def test_compute_max_bar_spacing_branches(self, concrete_c30):
        """Test compute max bar spacing branches."""
        check = CrackingCheck(section=_make_section(), concrete=concrete_c30)

        spacing_bottom = check._compute_max_bar_spacing(eps_top=0.001, eps_bottom=-0.001)
        assert spacing_bottom == pytest.approx(100.0, rel=1e-12)

        spacing_none = check._compute_max_bar_spacing(
            eps_top=0.001,
            eps_bottom=-0.001,
            h_c_ef_limit=20.0,
        )
        assert spacing_none == pytest.approx(0.0, abs=1e-12)

        spacing_top_net_tension = check._compute_max_bar_spacing(
            eps_top=-0.0005,
            eps_bottom=-0.001,
            face="top",
        )
        assert spacing_top_net_tension == pytest.approx(160.0, rel=1e-12)

    def test_peak_stress_creep_wrapper_and_f_yk_max(self, concrete_c30):
        """Test peak stress creep wrapper and f yk max."""
        check = CrackingCheck(section=_make_section(), concrete=concrete_c30)
        diag = _FakeDiagram(
            eps_top=0.001,
            eps_bottom=-0.001,
            fibre_mat=np.array(["concrete", "concrete", "steel"]),
            forces=np.array([30.0, -5.0, 100.0]),
            areas=np.array([2.0, 1.0, 1.0]),
        )
        peak = check._get_peak_concrete_stress(0.001, -0.001, diag)
        assert peak == pytest.approx(15.0, rel=1e-12)

        phi_nl = check._compute_nonlinear_creep_coefficient(sigma_c=18.0)
        expected = compute_nonlinear_creep_coefficient(18.0, check.concrete.f_ck, check.creep_coefficient)
        assert phi_nl == pytest.approx(expected, rel=1e-12)

        assert check._get_f_yk_max() == pytest.approx(500.0, abs=1e-12)


class TestCrackingFaceAndMainFlow:
    """Tests for TestCrackingFaceAndMainFlow."""
    def test_calculate_face_crack_width_no_tension_rebar_returns_zero(self, monkeypatch, concrete_c30):
        """Test calculate face crack width no tension rebar returns zero."""
        check = CrackingCheck(section=_make_section(), concrete=concrete_c30)
        monkeypatch.setattr(
            CrackingCheck,
            "_get_tension_rebar_info",
            lambda self, eps_top, eps_bottom, face=None, h_c_ef_limit=None, **kw: (0.0, 0.0, []),
        )

        result = check._calculate_face_crack_width(
            eps_top=0.001,
            eps_bottom=-0.001,
            face="bottom",
            x=200.0,
            is_net_tension=False,
            suppress_warnings=True,
        )
        assert result.w_k == pytest.approx(0.0, abs=1e-12)
        assert result.s_r_max == pytest.approx(0.0, abs=1e-12)
        assert result.rho_p_eff == pytest.approx(0.0, abs=1e-12)

    def test_calculate_face_crack_width_cover_fallback_and_yield_flag(self, monkeypatch, concrete_c30):
        """Test calculate face crack width cover fallback and yield flag."""
        check = CrackingCheck(section=_make_section(), concrete=concrete_c30)

        monkeypatch.setattr(
            CrackingCheck,
            "_get_tension_rebar_info",
            lambda self, eps_top, eps_bottom, face=None, h_c_ef_limit=None, **kw: (1000.0, 45.0, [(20.0, 3)]),
        )
        monkeypatch.setattr(CrackingCheck, "find_rho_p_eff", lambda self, A_s_tension, h_c_ef, xi_1=0.0, A_p=0.0, A_c_eff=None: 0.02)
        monkeypatch.setattr(CrackingCheck, "_get_steel_stress", lambda self, eps_top, eps_bottom, face=None, h_c_ef_limit=None, **kw: 300.0)
        monkeypatch.setattr(CrackingCheck, "find_k_2", lambda self, eps_top, eps_bottom: 0.5)
        monkeypatch.setattr(CrackingCheck, "_compute_max_bar_spacing", lambda self, eps_top, eps_bottom, face=None, h_c_ef_limit=None, **kw: 0.0)
        monkeypatch.setattr(
            CrackingCheck,
            "find_maximum_crack_spacing",
            lambda self, cover, phi_eq, rho_p_eff, k_2, x=None, has_tension_reinforcement=True, sigma_s=0.0, bar_spacing=0.0: 120.0,
        )
        monkeypatch.setattr(CrackingCheck, "_get_tension_zone_E_s", lambda self, eps_top, eps_bottom, **kw: 200000.0)
        monkeypatch.setattr(CrackingCheck, "find_strain_difference", lambda self, sigma_s, rho_p_eff, E_s: 0.001)
        monkeypatch.setattr(CrackingCheck, "_get_f_yk_max", lambda self: 250.0)

        def _raise_cover(self, reference: str):
            raise ValueError("invalid reference")

        monkeypatch.setattr(type(check.section), "get_concrete_cover", _raise_cover)

        result = check._calculate_face_crack_width(
            eps_top=0.001,
            eps_bottom=-0.001,
            face="bottom",
            x=200.0,
            is_net_tension=False,
            suppress_warnings=True,
        )
        assert result.w_k == pytest.approx(0.12, rel=1e-12)
        assert result.cover == pytest.approx(45.0, rel=1e-12)
        assert result.steel_yielded is True

    def test_perform_check_delegates_to_single_case(self, monkeypatch, concrete_c30):
        """Test perform check delegates to single case."""
        check = CrackingCheck(section=_make_section(), concrete=concrete_c30)

        captured = {}
        sentinel = check._create_result(
            check_name="Cracking check (EC2 §7.3)",
            code_reference="EC2 §7.3",
            warning_threshold=0.95,
            utilization=0.2,
            message="sentinel",
            details={},
        )

        def _fake_single(self, **kwargs):
            captured.update(kwargs)
            return sentinel

        monkeypatch.setattr(CrackingCheck, "_check_single_case", _fake_single)
        result = check.perform_check(M_Ed=80.0, N_Ed=120.0, force_cracked=True, suppress_warnings=True)

        assert result is sentinel
        assert captured["My_Ed"] == pytest.approx(80.0, rel=1e-12)
        assert captured["N_Ed"] == pytest.approx(120.0, rel=1e-12)
        assert captured["force_cracked"] is True
        assert captured["suppress_warnings"] is True

    def test_check_single_case_unreinforced_raises(self, concrete_c30):
        """Test check single case unreinforced raises."""
        check = CrackingCheck(
            section=create_rectangular_section(width=300.0, height=500.0),
            concrete=concrete_c30,
        )
        with pytest.raises(ValueError, match="invalid for unreinforced sections"):
            check.perform_check(M_Ed=20.0, N_Ed=0.0)

    def test_check_single_case_uncracked_returns_zero_util(self, monkeypatch, concrete_c30):
        """Test check single case uncracked returns zero util."""
        check = CrackingCheck(section=_make_section(), concrete=concrete_c30)
        monkeypatch.setattr(CrackingCheck, "find_cracking_moment", lambda self, N_Ed=0.0: 200.0)

        result = check.perform_check(M_Ed=20.0, N_Ed=0.0)
        assert result.status == CheckStatus.PASS
        assert result.utilization == pytest.approx(0.0, abs=1e-12)
        assert result.details["is_cracked"] is False

    def test_check_single_case_solver_error_returns_inf(self, monkeypatch, concrete_c30):
        """Test check single case solver error returns inf."""
        check = CrackingCheck(section=_make_section(), concrete=concrete_c30)
        monkeypatch.setattr(CrackingCheck, "find_cracking_moment", lambda self, N_Ed=0.0: 0.0)
        monkeypatch.setattr(
            CrackingCheck,
            "_get_diagram",
            lambda self, ignore_compression_steel=False: _FakeDiagram(
                eps_top=0.001,
                eps_bottom=-0.001,
                raise_on_find=True,
            ),
        )

        result = check.perform_check(M_Ed=100.0, N_Ed=0.0, force_cracked=True)
        assert result.status == CheckStatus.FAIL
        assert result.utilization == float("inf")
        assert "outside diagram" in result.details["error"]

    def test_check_single_case_net_compression_branch(self, monkeypatch, concrete_c30):
        """Test check single case net compression branch."""
        check = CrackingCheck(section=_make_section(), concrete=concrete_c30)
        monkeypatch.setattr(CrackingCheck, "find_cracking_moment", lambda self, N_Ed=0.0: 0.0)
        monkeypatch.setattr(
            CrackingCheck,
            "_get_diagram",
            lambda self, ignore_compression_steel=False: _FakeDiagram(
                eps_top=0.0008,
                eps_bottom=0.0002,
            ),
        )
        monkeypatch.setattr(CrackingCheck, "_get_peak_concrete_stress", lambda self, eps_top, eps_bottom, diagram=None, **kw: 10.0)

        result = check.perform_check(M_Ed=120.0, N_Ed=200.0, force_cracked=True)
        assert result.status == CheckStatus.PASS
        assert result.utilization == pytest.approx(0.0, abs=1e-12)
        assert result.details["is_cracked"] is False
        assert result.details["w_k"] == pytest.approx(0.0, abs=1e-12)

    def test_check_single_case_face_selection_net_tension(self, monkeypatch, concrete_c30):
        """Test check single case face selection net tension."""
        check = CrackingCheck(section=_make_section(), concrete=concrete_c30)
        monkeypatch.setattr(CrackingCheck, "find_cracking_moment", lambda self, N_Ed=0.0: 0.0)
        monkeypatch.setattr(
            CrackingCheck,
            "_get_diagram",
            lambda self, ignore_compression_steel=False: _FakeDiagram(
                eps_top=-0.0005,
                eps_bottom=-0.0010,
            ),
        )
        monkeypatch.setattr(CrackingCheck, "_get_peak_concrete_stress", lambda self, eps_top, eps_bottom, diagram=None, **kw: 5.0)

        captured_faces = []

        def _fake_face(self, eps_top, eps_bottom, face, x, is_net_tension, suppress_warnings=False, **kwargs):
            captured_faces.append({"face": face, "is_net_tension": is_net_tension})
            # Bottom face returns larger w_k so it should govern
            w_k = 0.15 if face == "bottom" else 0.08
            return CrackingResult(
                w_k=w_k,
                w_k_limit=self.w_k_limit,
                s_r_max=120.0,
                eps_sm_minus_eps_cm=0.001,
                sigma_s=200.0,
                rho_p_eff=0.01,
                h_c_ef=120.0,
                x=x,
                is_cracked=True,
                phi_eq=20.0,
                cover=40.0,
            )

        monkeypatch.setattr(CrackingCheck, "_calculate_face_crack_width", _fake_face)
        result = check.perform_check(M_Ed=120.0, N_Ed=0.0, force_cracked=True, suppress_warnings=True)

        # Both faces checked in net tension (default policy)
        assert len(captured_faces) == 2
        assert all(c["is_net_tension"] for c in captured_faces)
        faces_checked = {c["face"] for c in captured_faces}
        assert faces_checked == {"top", "bottom"}
        # Bottom face governed (larger w_k)
        assert result.utilization == pytest.approx(0.15 / check.w_k_limit, rel=1e-12)
        assert result.details["governing_face"] == "bottom"
        assert result.details["is_net_tension"] is True

    def test_check_single_case_face_selection_bending(self, monkeypatch, concrete_c30):
        """Test check single case face selection bending."""
        check = CrackingCheck(section=_make_section(), concrete=concrete_c30)
        monkeypatch.setattr(CrackingCheck, "find_cracking_moment", lambda self, N_Ed=0.0: 0.0)
        monkeypatch.setattr(
            CrackingCheck,
            "_get_diagram",
            lambda self, ignore_compression_steel=False: _FakeDiagram(
                eps_top=0.001,
                eps_bottom=-0.001,
            ),
        )
        monkeypatch.setattr(CrackingCheck, "_get_peak_concrete_stress", lambda self, eps_top, eps_bottom, diagram=None, **kw: 5.0)

        captured = {}

        def _fake_face(self, eps_top, eps_bottom, face, x, is_net_tension, suppress_warnings=False, **kwargs):
            captured["face"] = face
            captured["is_net_tension"] = is_net_tension
            return CrackingResult(
                w_k=0.10,
                w_k_limit=self.w_k_limit,
                s_r_max=100.0,
                eps_sm_minus_eps_cm=0.001,
                sigma_s=180.0,
                rho_p_eff=0.01,
                h_c_ef=100.0,
                x=x,
                is_cracked=True,
                phi_eq=20.0,
                cover=40.0,
            )

        monkeypatch.setattr(CrackingCheck, "_calculate_face_crack_width", _fake_face)
        result = check.perform_check(M_Ed=120.0, N_Ed=0.0, force_cracked=True, suppress_warnings=True)

        assert captured["face"] == "bottom"
        assert captured["is_net_tension"] is False
        assert result.utilization == pytest.approx(0.10 / check.w_k_limit, rel=1e-12)

    def test_calculate_detailed_uncracked_and_net_compression(self, monkeypatch, concrete_c30):
        """Test calculate detailed uncracked and net compression."""
        check = CrackingCheck(section=_make_section(), concrete=concrete_c30)
        monkeypatch.setattr(CrackingCheck, "find_cracking_moment", lambda self, N_Ed=0.0: 200.0)
        uncracked = check.calculate_detailed(M_Ed=20.0, N_Ed=0.0)
        assert uncracked.is_cracked is False
        assert uncracked.w_k == pytest.approx(0.0, abs=1e-12)

        monkeypatch.setattr(CrackingCheck, "find_cracking_moment", lambda self, N_Ed=0.0: 0.0)
        monkeypatch.setattr(
            CrackingCheck,
            "_get_diagram",
            lambda self, ignore_compression_steel=False: _FakeDiagram(
                eps_top=0.0008,
                eps_bottom=0.0002,
            ),
        )
        monkeypatch.setattr(CrackingCheck, "_get_peak_concrete_stress", lambda self, eps_top, eps_bottom, diagram=None, **kw: 9.0)

        net_comp = check.calculate_detailed(M_Ed=100.0, N_Ed=100.0, force_cracked=True)
        assert net_comp.is_cracked is False
        assert net_comp.w_k == pytest.approx(0.0, abs=1e-12)
        assert net_comp.sigma_c_peak == pytest.approx(9.0, rel=1e-12)

    def test_plot_wrappers_delegate_to_viewer(self, concrete_c30, monkeypatch):
        """Test plot wrappers delegate to viewer."""
        check = CrackingCheck(section=_make_section(), concrete=concrete_c30)

        class _FakeViewer:
            def __init__(self, c):
                self.check = c

            def plot_load_cases(self, load_cases, **kwargs):
                return ("load_cases", load_cases, kwargs)

            def plot_contours(self, **kwargs):
                return ("contours", kwargs)

        fake_module = types.SimpleNamespace(CrackWidthViewer=_FakeViewer)
        monkeypatch.setitem(
            sys.modules,
            "materials.reinforced_concrete.analysis.crack_width_viewer",
            fake_module,
        )

        fig1 = check.plot_load_cases([{"M_Ed": 10.0, "N_Ed": 0.0}], show=False)
        fig2 = check.plot_crack_width_contours(show=False, n_grid=10)

        assert fig1[0] == "load_cases"
        assert fig2[0] == "contours"
