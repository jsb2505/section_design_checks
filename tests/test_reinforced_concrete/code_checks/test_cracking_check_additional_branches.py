"""
Additional branch coverage tests for CrackingCheck.
"""

from __future__ import annotations

import numpy as np
import pytest

import section_design_checks.reinforced_concrete.code_checks.ec2_2004.cracking_check as cc_mod
from section_design_checks.reinforced_concrete.code_checks.ec2_2004.cracking_check import (
    CrackingCheck,
    CrackingResult,
)
from section_design_checks.reinforced_concrete.geometry import (
    create_linear_rebar_layer,
    create_rectangular_section,
)
from section_design_checks.reinforced_concrete.materials import Rebar


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
        fibre_mat: np.ndarray | None = None,
        forces: np.ndarray | None = None,
        areas: np.ndarray | None = None,
    ):
        self.eps_top = eps_top
        self.eps_bottom = eps_bottom
        self._fibre_mat = fibre_mat if fibre_mat is not None else np.array(["concrete", "concrete"])
        self._forces = forces if forces is not None else np.array([20.0, 10.0])
        self._areas = areas if areas is not None else np.array([2.0, 2.0])

    def find_strains_for_MN(self, *args, **kwargs):
        return self.eps_top, self.eps_bottom

    def find_strain_state_for_MN(self, *args, **kwargs):
        return None

    def get_fibre_forces_from_end_strains(self, eps_top: float, eps_bottom: float):
        y = np.zeros_like(self._forces, dtype=float)
        return self._forces, y, self._areas


class TestCrackingAdditionalBranches:
    """Tests for TestCrackingAdditionalBranches."""
    def test_snapshot_and_diagram_cache_paths(self, monkeypatch, concrete_c30):
        """Test snapshot and diagram cache paths."""
        check = CrackingCheck(section=_make_section(), concrete=concrete_c30)

        snapshot = check._take_snapshot()
        assert snapshot["section"]
        assert snapshot["concrete"]
        assert snapshot["E_c_eff"] == pytest.approx(check.E_c_eff, rel=1e-12)

        calls = {"n": 0, "ignore": []}

        class _DummyDiagram:
            def __init__(self, **kwargs):
                calls["n"] += 1
                calls["ignore"].append(kwargs["ignore_compression_steel"])

            def find_strains_for_MN(self, *args, **kwargs):
                return 0.0, 0.0

            def find_strain_state_for_MN(self, *args, **kwargs):
                return None

        monkeypatch.setattr(
            cc_mod,
            "create_interaction_diagram",
            lambda **kwargs: _DummyDiagram(**kwargs),
        )

        snap = {"k": 1}
        monkeypatch.setattr(CrackingCheck, "_take_snapshot", lambda self: dict(snap))

        main_1 = check._get_diagram(False)
        assert isinstance(main_1, _DummyDiagram)
        assert calls["n"] == 1
        assert calls["ignore"][-1] is False

        main_2 = check._get_diagram(False)
        assert main_2 is main_1
        assert calls["n"] == 1

        snap["k"] = 2
        rebuilt_main = check._get_diagram(False)
        assert isinstance(rebuilt_main, _DummyDiagram)
        assert rebuilt_main is not main_1
        assert calls["n"] == 2
        assert calls["ignore"][-1] is False

        no_comp_1 = check._get_diagram(True)
        assert isinstance(no_comp_1, _DummyDiagram)
        assert calls["n"] == 3
        assert calls["ignore"][-1] is True

        no_comp_2 = check._get_diagram(True)
        assert no_comp_2 is no_comp_1
        assert calls["n"] == 3

        snap["k"] = 3
        rebuilt_no_comp = check._get_diagram(True)
        assert isinstance(rebuilt_no_comp, _DummyDiagram)
        assert rebuilt_no_comp is not no_comp_1
        assert calls["n"] == 4
        assert calls["ignore"][-1] is True

    def test_find_cracking_moment_branches(self, monkeypatch, concrete_c30):
        """Test find cracking moment branches."""
        check = CrackingCheck(section=_make_section(), concrete=concrete_c30)

        monkeypatch.setattr(type(check.concrete), "find_mean_flexural_tensile_strength", lambda self, h: 3.0)
        monkeypatch.setattr(type(check.section), "get_transformed_second_moment_area", lambda self, E: (2.0e9, 0.0, 0.0))
        monkeypatch.setattr(type(check.section), "get_transformed_centroid", lambda self, E: (0.0, 200.0, 0.0))
        monkeypatch.setattr(type(check.section), "get_transformed_area", lambda self, E: 100000.0)

        with_axial = check.find_cracking_moment(N_Ed=100.0, use_f_ctm_fl=True)
        assert with_axial == pytest.approx(40.0, rel=1e-12)

        monkeypatch.setattr(type(check.section), "get_transformed_centroid", lambda self, E: (0.0, 0.0, 0.0))
        no_y_tension = check.find_cracking_moment(N_Ed=0.0, use_f_ctm_fl=True)
        assert no_y_tension == pytest.approx(24.0, rel=1e-12)

    def test_maximum_spacing_eq714_without_x(self, concrete_c30):
        """Test maximum spacing eq714 without x."""
        check = CrackingCheck(section=_make_section(), concrete=concrete_c30)
        s_r_max = check.find_maximum_crack_spacing(
            cover=40.0,
            phi_eq=20.0,
            rho_p_eff=1.0,
            k_2=0.5,
            x=None,
            has_tension_reinforcement=True,
            bar_spacing=10000.0,
        )
        assert s_r_max == pytest.approx(1.3 * check.height, rel=1e-12)

    def test_minimum_reinforcement_k_interpolation_and_tension_area(self, monkeypatch, concrete_c30):
        """Test minimum reinforcement k interpolation and tension area."""
        check = CrackingCheck(section=_make_section(width=500.0, height=600.0), concrete=concrete_c30)

        monkeypatch.setattr(CrackingCheck, "find_k_c", lambda self, N_Ed=0.0, is_in_bending=True: 0.4)
        monkeypatch.setattr(type(check.section), "get_transformed_centroid", lambda self, E: (0.0, 360.0, 0.0))

        as_min = check.find_minimum_crack_reinforcement(
            steel_stress=500.0,
            k_c=None,
            N_Ed=-100.0,
            is_in_bending=True,
        )

        k = 1.0 - 0.35 * (500.0 - 300.0) / 500.0
        expected = 0.4 * k * check.concrete.f_ctm * (360.0 * 500.0) / 500.0
        assert as_min == pytest.approx(expected, rel=1e-12)

    def test_minimum_reinforcement_large_section_uses_k_065(self, concrete_c30):
        """Test minimum reinforcement large section uses k 065."""
        check = CrackingCheck(section=_make_section(width=900.0, height=900.0), concrete=concrete_c30)
        as_min = check.find_minimum_crack_reinforcement(
            steel_stress=500.0,
            k_c=0.4,
            N_Ed=0.0,
            is_in_bending=True,
        )
        expected = 0.4 * 0.65 * check.concrete.f_ctm * (0.5 * 900.0 * 900.0) / 500.0
        assert as_min == pytest.approx(expected, rel=1e-12)

    def test_net_tension_face_filters_for_rebar_stress_and_spacing(self, monkeypatch, concrete_c30):
        """Test net tension face filters for rebar stress and spacing."""
        check = CrackingCheck(section=_make_section(), concrete=concrete_c30)

        area, _, bars = check._get_tension_rebar_info(
            eps_top=-0.0004,
            eps_bottom=-0.0012,
            face="bottom",
            h_c_ef_limit=60.0,
        )
        expected_bottom_area = 3.0 * Rebar(diameter=20, grade="B500B").area
        assert area == pytest.approx(expected_bottom_area, rel=1e-12)
        assert bars == [(20.0, 3)]

        monkeypatch.setattr(
            cc_mod.flexure_utils,
            "calculate_rebar_characteristic_stress_from_strain",
            lambda **kwargs: -123.0,
        )
        sigma_top = check._get_steel_stress(
            eps_top=-0.0012,
            eps_bottom=-0.0004,
            face="top",
            h_c_ef_limit=80.0,
        )
        assert sigma_top == pytest.approx(123.0, rel=1e-12)

        spacing_bottom = check._compute_max_bar_spacing(
            eps_top=-0.0004,
            eps_bottom=-0.0012,
            face="bottom",
            h_c_ef_limit=80.0,
        )
        spacing_top = check._compute_max_bar_spacing(
            eps_top=-0.0012,
            eps_bottom=-0.0004,
            face="top",
            h_c_ef_limit=80.0,
        )
        assert spacing_bottom > 0.0
        assert spacing_top > 0.0

    def test_get_steel_stress_bottom_face_filter_skips_top_bars(self, monkeypatch, concrete_c30):
        """Test get steel stress bottom face filter skips top bars."""
        check = CrackingCheck(section=_make_section(), concrete=concrete_c30)
        calls = {"n": 0}

        def _stress_from_strain(**kwargs):
            calls["n"] += 1
            return -100.0

        monkeypatch.setattr(
            cc_mod.flexure_utils,
            "calculate_rebar_characteristic_stress_from_strain",
            _stress_from_strain,
        )
        sigma = check._get_steel_stress(
            eps_top=-0.0004,
            eps_bottom=-0.0012,
            face="bottom",
        )
        assert sigma == pytest.approx(100.0, rel=1e-12)
        assert calls["n"] == 3

    def test_tension_zone_es_top_tension_branch(self, concrete_c30):
        """Test tension zone es top tension branch."""
        check = CrackingCheck(
            section=_make_section(bottom_E_s=190000.0, top_E_s=210000.0),
            concrete=concrete_c30,
        )
        assert check._get_tension_zone_E_s(-0.001, 0.001) == pytest.approx(210000.0, rel=1e-12)

    def test_peak_stress_zero_area_build_diagram_and_helpers(self, monkeypatch, concrete_c30):
        """Test peak stress zero area build diagram and helpers."""
        check = CrackingCheck(section=_make_section(), concrete=concrete_c30)

        zero_area_diag = _FakeDiagram(
            eps_top=0.001,
            eps_bottom=-0.001,
            fibre_mat=np.array(["concrete", "concrete"]),
            forces=np.array([10.0, 20.0]),
            areas=np.array([0.0, 0.0]),
        )
        assert check._get_peak_concrete_stress(0.001, -0.001, zero_area_diag) == pytest.approx(0.0, abs=1e-12)

        monkeypatch.setattr(cc_mod, "create_interaction_diagram", lambda **kwargs: kwargs)
        built = check._build_diagram_with_E_c_eff(12345.0, ignore_compression_steel=True)
        assert built["elastic_modulus"] == pytest.approx(12345.0, rel=1e-12)
        assert built["ignore_compression_steel"] is True

        unchanged, factor0 = CrackingCheck._compute_bar_diameter_correction(
            s_r_max=200.0,
            phi_eq=0.0,
            actual_bar_diameter=16.0,
            cover=35.0,
            k_3=3.4,
        )
        assert unchanged == pytest.approx(200.0, rel=1e-12)
        assert factor0 == pytest.approx(1.0, rel=1e-12)

        corrected, factor = CrackingCheck._compute_bar_diameter_correction(
            s_r_max=200.0,
            phi_eq=20.0,
            actual_bar_diameter=16.0,
            cover=35.0,
            k_3=3.4,
        )
        assert corrected == pytest.approx(183.8, rel=1e-12)
        assert factor == pytest.approx(0.8, rel=1e-12)

        empty = CrackingCheck(
            section=create_rectangular_section(width=300.0, height=500.0),
            concrete=concrete_c30,
        )
        assert empty._get_f_yk_max() == pytest.approx(500.0, abs=1e-12)

    def test_calculate_face_width_warning_net_tension_and_diameter_correction(self, monkeypatch, concrete_c30):
        """Test calculate face width warning net tension and diameter correction."""
        check = CrackingCheck(section=_make_section(), concrete=concrete_c30)

        monkeypatch.setattr(
            CrackingCheck,
            "_get_tension_rebar_info",
            lambda self, eps_top, eps_bottom, face=None, h_c_ef_limit=None, **kw: (0.0, 0.0, []),
        )
        with pytest.warns(UserWarning, match="No tension reinforcement found"):
            zero = check._calculate_face_crack_width(
                eps_top=0.001,
                eps_bottom=-0.001,
                face="bottom",
                x=200.0,
                is_net_tension=False,
                suppress_warnings=False,
            )
        assert zero.w_k == pytest.approx(0.0, abs=1e-12)

        check2 = CrackingCheck(section=_make_section(), concrete=concrete_c30)
        captured = {}

        def _get_tension(self, eps_top, eps_bottom, face=None, h_c_ef_limit=None, **kw):
            return (1000.0, 40.0, [(20.0, 2)])

        def _get_d(self, compression_face: str, zone_fraction: float | None = None):
            captured["compression_face"] = compression_face
            captured["zone_fraction"] = zone_fraction
            return 460.0

        monkeypatch.setattr(CrackingCheck, "_get_tension_rebar_info", _get_tension)
        monkeypatch.setattr(type(check2.section), "get_effective_depth", _get_d)
        monkeypatch.setattr(CrackingCheck, "find_rho_p_eff", lambda self, A_s_tension, h_c_ef, xi_1=0.0, A_p=0.0, A_c_eff=None: 0.02)
        monkeypatch.setattr(CrackingCheck, "_get_steel_stress", lambda self, eps_top, eps_bottom, face=None, h_c_ef_limit=None, **kw: 250.0)
        monkeypatch.setattr(CrackingCheck, "find_k_2", lambda self, eps_top, eps_bottom: 0.5)
        monkeypatch.setattr(CrackingCheck, "_compute_max_bar_spacing", lambda self, eps_top, eps_bottom, face=None, h_c_ef_limit=None, **kw: 100.0)
        monkeypatch.setattr(CrackingCheck, "find_maximum_crack_spacing", lambda self, **kwargs: 200.0)
        monkeypatch.setattr(CrackingCheck, "_get_tension_zone_E_s", lambda self, eps_top, eps_bottom, **kw: 200000.0)
        monkeypatch.setattr(CrackingCheck, "find_strain_difference", lambda self, sigma_s, rho_p_eff, E_s: 0.001)
        monkeypatch.setattr(CrackingCheck, "_get_f_yk_max", lambda self: 500.0)
        monkeypatch.setattr(type(check2.section), "get_concrete_cover", lambda self, reference: 35.0)

        result = check2._calculate_face_crack_width(
            eps_top=-0.0010,
            eps_bottom=-0.0005,
            face="top",
            x=120.0,
            is_net_tension=True,
            suppress_warnings=True,
            actual_bar_diameter=16.0,
        )
        assert captured["compression_face"] == "bottom"
        assert captured["zone_fraction"] == pytest.approx(0.5, rel=1e-12)
        assert result.h_c_ef == pytest.approx(100.0, rel=1e-12)
        assert result.s_r_max_uncorrected == pytest.approx(200.0, rel=1e-12)
        assert result.phi_correction_factor == pytest.approx(0.8, rel=1e-12)

    def test_calculate_face_width_relaxed_hcef_and_update_branch(self, monkeypatch, concrete_c30):
        """Test calculate face width relaxed hcef and update branch."""
        check = CrackingCheck(section=_make_section(), concrete=concrete_c30)
        calls = {"n": 0}

        def _get_tension(self, eps_top, eps_bottom, face=None, h_c_ef_limit=None, **kw):
            calls["n"] += 1
            if calls["n"] == 1:
                return (1000.0, 40.0, [(20.0, 3)])
            if calls["n"] == 2:
                return (0.0, 0.0, [])
            return (1000.0, 40.0, [(20.0, 2)])

        monkeypatch.setattr(CrackingCheck, "_get_tension_rebar_info", _get_tension)
        monkeypatch.setattr(type(check.section), "get_effective_depth", lambda self, compression_face, zone_fraction=None: 450.0)
        monkeypatch.setattr(CrackingCheck, "find_h_c_ef", lambda self, d, x=None, **kw: 10.0)
        monkeypatch.setattr(CrackingCheck, "find_rho_p_eff", lambda self, A_s_tension, h_c_ef, xi_1=0.0, A_p=0.0, A_c_eff=None: 0.02)
        monkeypatch.setattr(CrackingCheck, "_get_steel_stress", lambda self, eps_top, eps_bottom, face=None, h_c_ef_limit=None, **kw: 200.0)
        monkeypatch.setattr(CrackingCheck, "find_k_2", lambda self, eps_top, eps_bottom: 0.5)
        monkeypatch.setattr(CrackingCheck, "_compute_max_bar_spacing", lambda self, eps_top, eps_bottom, face=None, h_c_ef_limit=None, **kw: 100.0)
        monkeypatch.setattr(CrackingCheck, "find_maximum_crack_spacing", lambda self, **kwargs: 140.0)
        monkeypatch.setattr(CrackingCheck, "_get_tension_zone_E_s", lambda self, eps_top, eps_bottom, **kw: 200000.0)
        monkeypatch.setattr(CrackingCheck, "find_strain_difference", lambda self, sigma_s, rho_p_eff, E_s: 0.001)
        monkeypatch.setattr(CrackingCheck, "_get_f_yk_max", lambda self: 500.0)
        monkeypatch.setattr(type(check.section), "get_concrete_cover", lambda self, reference: 35.0)

        with pytest.warns(UserWarning, match="Relaxing to"):
            result = check._calculate_face_crack_width(
                eps_top=0.001,
                eps_bottom=-0.001,
                face="bottom",
                x=470.0,
                is_net_tension=False,
                suppress_warnings=False,
            )
        assert result.w_k > 0.0

    def test_calculate_face_width_relaxed_hcef_still_no_bars_breaks(self, monkeypatch, concrete_c30):
        """Test calculate face width relaxed hcef still no bars breaks."""
        check = CrackingCheck(section=_make_section(), concrete=concrete_c30)
        calls = {"n": 0}

        def _get_tension(self, eps_top, eps_bottom, face=None, h_c_ef_limit=None, **kw):
            calls["n"] += 1
            if calls["n"] == 1:
                return (1000.0, 40.0, [(20.0, 3)])
            return (0.0, 0.0, [])

        monkeypatch.setattr(CrackingCheck, "_get_tension_rebar_info", _get_tension)
        monkeypatch.setattr(type(check.section), "get_effective_depth", lambda self, compression_face, zone_fraction=None: 450.0)
        monkeypatch.setattr(CrackingCheck, "find_h_c_ef", lambda self, d, x=None, **kw: 10.0)
        monkeypatch.setattr(CrackingCheck, "find_rho_p_eff", lambda self, A_s_tension, h_c_ef, xi_1=0.0, A_p=0.0, A_c_eff=None: 0.02)
        monkeypatch.setattr(CrackingCheck, "_get_steel_stress", lambda self, eps_top, eps_bottom, face=None, h_c_ef_limit=None, **kw: 200.0)
        monkeypatch.setattr(CrackingCheck, "find_k_2", lambda self, eps_top, eps_bottom: 0.5)
        monkeypatch.setattr(CrackingCheck, "_compute_max_bar_spacing", lambda self, eps_top, eps_bottom, face=None, h_c_ef_limit=None, **kw: 100.0)
        monkeypatch.setattr(CrackingCheck, "find_maximum_crack_spacing", lambda self, **kwargs: 140.0)
        monkeypatch.setattr(CrackingCheck, "_get_tension_zone_E_s", lambda self, eps_top, eps_bottom, **kw: 200000.0)
        monkeypatch.setattr(CrackingCheck, "find_strain_difference", lambda self, sigma_s, rho_p_eff, E_s: 0.001)
        monkeypatch.setattr(CrackingCheck, "_get_f_yk_max", lambda self: 500.0)
        monkeypatch.setattr(type(check.section), "get_concrete_cover", lambda self, reference: 35.0)

        with pytest.warns(UserWarning, match="Relaxing to"):
            result = check._calculate_face_crack_width(
                eps_top=0.001,
                eps_bottom=-0.001,
                face="bottom",
                x=470.0,
                is_net_tension=False,
                suppress_warnings=False,
            )
        assert result.w_k > 0.0

    def test_calculate_face_width_early_zero_return_after_filter_update(self, monkeypatch, concrete_c30):
        """Test calculate face width early zero return after filter update."""
        check = CrackingCheck(section=_make_section(), concrete=concrete_c30)
        calls = {"n": 0}

        def _get_tension(self, eps_top, eps_bottom, face=None, h_c_ef_limit=None, **kw):
            calls["n"] += 1
            if calls["n"] == 1:
                return (1000.0, 40.0, [(20.0, 2)])
            return (0.0, 40.0, [(20.0, 1)])

        monkeypatch.setattr(CrackingCheck, "_get_tension_rebar_info", _get_tension)
        monkeypatch.setattr(type(check.section), "get_effective_depth", lambda self, compression_face, zone_fraction=None: 450.0)
        monkeypatch.setattr(CrackingCheck, "find_h_c_ef", lambda self, d, x=None, **kw: 120.0)

        result = check._calculate_face_crack_width(
            eps_top=0.001,
            eps_bottom=-0.001,
            face="bottom",
            x=200.0,
            is_net_tension=False,
            suppress_warnings=True,
        )
        assert result.w_k == pytest.approx(0.0, abs=1e-12)
        assert result.rho_p_eff == pytest.approx(0.0, abs=1e-12)

    def test_check_single_case_stress_limits_nonlinear_creep_and_details(self, monkeypatch, concrete_c30):
        """Test check single case stress limits nonlinear creep and details."""
        check = CrackingCheck(
            section=_make_section(),
            concrete=concrete_c30,
            check_k1_stress=True,
            check_k2_stress=True,
            check_k3_stress=True,
            check_yielding=True,
            check_k4_stress=True,
            apply_nonlinear_creep=True,
            iterate_nonlinear_creep=True,
        )

        monkeypatch.setattr(CrackingCheck, "find_cracking_moment", lambda self, N_Ed=0.0: 0.0)
        monkeypatch.setattr(
            CrackingCheck,
            "_get_diagram",
            lambda self, ignore_compression_steel=False: _FakeDiagram(eps_top=-0.0010, eps_bottom=-0.0003),
        )
        monkeypatch.setattr(
            CrackingCheck,
            "_get_peak_concrete_stress",
            lambda self, eps_top, eps_bottom, diagram=None, **kw: 25.0 if diagram is None else 22.0,
        )
        monkeypatch.setattr(cc_mod, "check_characteristic_concrete_stress", lambda sigma_c, f_ck: (True, "k1"))
        monkeypatch.setattr(cc_mod, "check_quasi_permanent_concrete_stress", lambda sigma_c, f_ck: (True, "k2"))
        monkeypatch.setattr(cc_mod, "check_characteristic_reinforcement_stress", lambda sigma_s, f_yk: (True, "k3"))
        monkeypatch.setattr(cc_mod, "check_reinforcement_yielding", lambda sigma_s, f_yk: (True, "yield"))
        monkeypatch.setattr(cc_mod, "check_imposed_deformation_stress", lambda sigma_s, f_yk: (True, "k4"))
        monkeypatch.setattr(CrackingCheck, "_compute_nonlinear_creep_coefficient", lambda self, sigma_c: 3.0)
        monkeypatch.setattr(
            CrackingCheck,
            "_build_diagram_with_E_c_eff",
            lambda self, E_c_eff, ignore_compression_steel=False: _FakeDiagram(
                eps_top=-0.0011, eps_bottom=-0.0002
            ),
        )

        captured = {}

        def _fake_face(
            self,
            eps_top,
            eps_bottom,
            face,
            x,
            is_net_tension,
            suppress_warnings=False,
            actual_bar_diameter=None,
            **kw,
        ):
            captured["face"] = face
            captured["is_net_tension"] = is_net_tension
            return CrackingResult(
                w_k=0.2,
                w_k_limit=self.w_k_limit,
                s_r_max=140.0,
                eps_sm_minus_eps_cm=0.001,
                sigma_s=600.0,
                rho_p_eff=0.02,
                h_c_ef=120.0,
                x=x,
                is_cracked=True,
                phi_eq=20.0,
                cover=35.0,
                actual_bar_diameter=16.0,
                s_r_max_uncorrected=150.0,
                phi_correction_factor=0.8,
            )

        monkeypatch.setattr(CrackingCheck, "_calculate_face_crack_width", _fake_face)

        with pytest.warns(UserWarning):
            result = check.perform_check(
                M_Ed=120.0,
                N_Ed=0.0,
                force_cracked=True,
                suppress_warnings=False,
                actual_bar_diameter=16.0,
            )

        assert captured["is_net_tension"] is True
        assert captured["face"] == "top"
        assert result.details["steel_yielded"] is True
        assert result.details["nonlinear_creep_applied"] is True
        assert result.details["actual_bar_diameter"] == pytest.approx(16.0, rel=1e-12)
        assert result.details["phi_correction_factor"] == pytest.approx(0.8, rel=1e-12)
        assert result.details["s_r_max_uncorrected"] == pytest.approx(150.0, rel=1e-12)

    def test_calculate_detailed_nonlinear_creep_and_face_selection(self, monkeypatch, concrete_c30):
        """Test calculate detailed nonlinear creep and face selection."""
        check = CrackingCheck(
            section=_make_section(),
            concrete=concrete_c30,
            check_k2_stress=True,
            apply_nonlinear_creep=True,
            iterate_nonlinear_creep=True,
        )

        monkeypatch.setattr(CrackingCheck, "find_cracking_moment", lambda self, N_Ed=0.0: 0.0)
        monkeypatch.setattr(
            CrackingCheck,
            "_get_diagram",
            lambda self, ignore_compression_steel=False: _FakeDiagram(eps_top=-0.0012, eps_bottom=-0.0003),
        )
        monkeypatch.setattr(
            CrackingCheck,
            "_get_peak_concrete_stress",
            lambda self, eps_top, eps_bottom, diagram=None, **kw: 24.0 if diagram is None else 21.0,
        )
        monkeypatch.setattr(cc_mod, "check_quasi_permanent_concrete_stress", lambda sigma_c, f_ck: (True, "k2"))
        monkeypatch.setattr(CrackingCheck, "_compute_nonlinear_creep_coefficient", lambda self, sigma_c: 3.0)
        monkeypatch.setattr(
            CrackingCheck,
            "_build_diagram_with_E_c_eff",
            lambda self, E_c_eff, ignore_compression_steel=False: _FakeDiagram(
                eps_top=-0.0011, eps_bottom=-0.0002
            ),
        )

        captured_faces = []

        def _fake_face(
            self,
            eps_top,
            eps_bottom,
            face,
            x,
            is_net_tension,
            suppress_warnings=False,
            actual_bar_diameter=None,
            **kw,
        ):
            captured_faces.append({"face": face, "is_net_tension": is_net_tension})
            # Top face returns larger w_k so it governs in the first net tension case
            w_k = 0.20 if face == "top" else 0.12
            return CrackingResult(
                w_k=w_k,
                w_k_limit=self.w_k_limit,
                s_r_max=130.0,
                eps_sm_minus_eps_cm=0.001,
                sigma_s=200.0,
                rho_p_eff=0.02,
                h_c_ef=110.0,
                x=x,
                is_cracked=True,
                phi_eq=20.0,
                cover=35.0,
            )

        monkeypatch.setattr(CrackingCheck, "_calculate_face_crack_width", _fake_face)
        out_net_tension = check.calculate_detailed(
            M_Ed=100.0,
            N_Ed=0.0,
            force_cracked=True,
            suppress_warnings=True,
            actual_bar_diameter=16.0,
        )

        # Both faces checked in net tension
        assert len(captured_faces) == 2
        assert all(c["is_net_tension"] for c in captured_faces)
        assert {c["face"] for c in captured_faces} == {"top", "bottom"}
        # Top face governed (larger w_k)
        assert out_net_tension.governing_face == "top"
        assert out_net_tension.nonlinear_creep_applied is True
        assert out_net_tension.creep_coefficient_used == pytest.approx(3.0, rel=1e-12)

        # --- Bending case ---
        captured_faces.clear()
        monkeypatch.setattr(
            CrackingCheck,
            "_get_diagram",
            lambda self, ignore_compression_steel=False: _FakeDiagram(eps_top=0.0010, eps_bottom=-0.0010),
        )
        monkeypatch.setattr(cc_mod, "check_quasi_permanent_concrete_stress", lambda sigma_c, f_ck: (False, "k2"))

        out_bending = check.calculate_detailed(
            M_Ed=100.0,
            N_Ed=0.0,
            force_cracked=True,
            suppress_warnings=True,
        )
        assert out_bending.is_cracked is True
        assert len(captured_faces) == 1
        assert captured_faces[0]["is_net_tension"] is False
        assert captured_faces[0]["face"] == "bottom"
        assert out_bending.governing_face == "bottom"

        # --- Net tension, bottom governs ---
        captured_faces.clear()
        monkeypatch.setattr(
            CrackingCheck,
            "_get_diagram",
            lambda self, ignore_compression_steel=False: _FakeDiagram(eps_top=-0.0004, eps_bottom=-0.0010),
        )
        monkeypatch.setattr(cc_mod, "check_quasi_permanent_concrete_stress", lambda sigma_c, f_ck: (False, "k2"))

        # Make bottom face return larger w_k for this sub-case
        def _fake_face_bottom_governs(
            self, eps_top, eps_bottom, face, x, is_net_tension,
            suppress_warnings=False, actual_bar_diameter=None, **kw,
        ):
            captured_faces.append({"face": face, "is_net_tension": is_net_tension})
            w_k = 0.22 if face == "bottom" else 0.10
            return CrackingResult(
                w_k=w_k, w_k_limit=self.w_k_limit, s_r_max=130.0,
                eps_sm_minus_eps_cm=0.001, sigma_s=200.0, rho_p_eff=0.02,
                h_c_ef=110.0, x=x, is_cracked=True, phi_eq=20.0, cover=35.0,
            )

        monkeypatch.setattr(CrackingCheck, "_calculate_face_crack_width", _fake_face_bottom_governs)
        out_bottom = check.calculate_detailed(
            M_Ed=100.0,
            N_Ed=0.0,
            force_cracked=True,
            suppress_warnings=True,
        )
        assert len(captured_faces) == 2
        assert all(c["is_net_tension"] for c in captured_faces)
        assert out_bottom.governing_face == "bottom"

    def test_perform_check_reports_unsolved_when_nonlinear_creep_resolve_fails(
        self, monkeypatch, concrete_c30
    ):
        """Non-linear creep re-solve failure should return an unsolved fail result."""
        check = CrackingCheck(
            section=_make_section(),
            concrete=concrete_c30,
            check_k2_stress=True,
            apply_nonlinear_creep=True,
            iterate_nonlinear_creep=True,
        )

        monkeypatch.setattr(
            CrackingCheck,
            "_get_diagram",
            lambda self, ignore_compression_steel=False: _FakeDiagram(
                eps_top=0.0010, eps_bottom=-0.0010
            ),
        )
        monkeypatch.setattr(
            CrackingCheck,
            "_get_peak_concrete_stress",
            lambda self, eps_top, eps_bottom, diagram=None, **kw: 24.0,
        )
        monkeypatch.setattr(
            cc_mod,
            "check_quasi_permanent_concrete_stress",
            lambda sigma_c, f_ck: (True, "k2"),
        )
        monkeypatch.setattr(
            CrackingCheck,
            "_compute_nonlinear_creep_coefficient",
            lambda self, sigma_c: 3.0,
        )

        class _FailingDiagram(_FakeDiagram):
            def find_strains_for_MN(self, *args, **kwargs):
                raise ValueError("outside capacity envelope")

        monkeypatch.setattr(
            CrackingCheck,
            "_build_diagram_with_E_c_eff",
            lambda self, E_c_eff, ignore_compression_steel=False: _FailingDiagram(
                eps_top=0.0, eps_bottom=0.0
            ),
        )

        with pytest.warns(UserWarning, match="k2"):
            out = check.perform_check(
                M_Ed=100.0,
                N_Ed=0.0,
                force_cracked=True,
                suppress_warnings=True,
            )

        assert out.utilization == float("inf")
        assert out.details["solved"] is False
        assert out.details["solver_stage"] == "nonlinear_creep"
        assert out.details["solver_error"] is not None
        assert "outside capacity envelope" in out.details["solver_error"]
        assert out.details["solver_residual_N"] is None
        assert out.details["solver_residual_M"] is None
        assert out.details["eps_top_pre_nl"] == pytest.approx(0.0010, rel=1e-12)
        assert out.details["eps_bottom_pre_nl"] == pytest.approx(-0.0010, rel=1e-12)

    def test_calculate_detailed_returns_unsolved_when_nonlinear_creep_resolve_fails(
        self, monkeypatch, concrete_c30
    ):
        """Detailed API should return an unsolved payload when NL creep re-solve fails."""
        check = CrackingCheck(
            section=_make_section(),
            concrete=concrete_c30,
            check_k2_stress=True,
            apply_nonlinear_creep=True,
            iterate_nonlinear_creep=True,
        )

        monkeypatch.setattr(
            CrackingCheck,
            "_get_diagram",
            lambda self, ignore_compression_steel=False: _FakeDiagram(
                eps_top=0.0010, eps_bottom=-0.0010
            ),
        )
        monkeypatch.setattr(
            CrackingCheck,
            "_get_peak_concrete_stress",
            lambda self, eps_top, eps_bottom, diagram=None, **kw: 24.0,
        )
        monkeypatch.setattr(
            cc_mod,
            "check_quasi_permanent_concrete_stress",
            lambda sigma_c, f_ck: (True, "k2"),
        )
        monkeypatch.setattr(
            CrackingCheck,
            "_compute_nonlinear_creep_coefficient",
            lambda self, sigma_c: 3.0,
        )

        class _FailingDiagram(_FakeDiagram):
            def find_strains_for_MN(self, *args, **kwargs):
                raise ValueError("outside capacity envelope")

        monkeypatch.setattr(
            CrackingCheck,
            "_build_diagram_with_E_c_eff",
            lambda self, E_c_eff, ignore_compression_steel=False: _FailingDiagram(
                eps_top=0.0, eps_bottom=0.0
            ),
        )

        out = check.calculate_detailed(
            M_Ed=100.0,
            N_Ed=0.0,
            force_cracked=True,
            suppress_warnings=True,
        )

        assert out.solved is False
        assert out.solver_stage == "nonlinear_creep"
        assert out.solver_error is not None
        assert "outside capacity envelope" in out.solver_error
        assert out.solver_residual_N is None
        assert out.solver_residual_M is None
        assert out.w_k is None
        assert out.s_r_max is None
        assert out.sigma_s is None
        assert out.creep_coefficient_used == pytest.approx(check.creep_coefficient, rel=1e-12)

    def test_calculate_detailed_unsolved_includes_solver_residuals(
        self, monkeypatch, concrete_c30
    ):
        """Residual dN/dM should be parsed into detailed result when available."""
        check = CrackingCheck(
            section=_make_section(),
            concrete=concrete_c30,
            apply_nonlinear_creep=False,
        )

        class _FailingDiagram(_FakeDiagram):
            def find_strains_for_MN(self, *args, **kwargs):
                raise ValueError(
                    "Inverse solver could not match M=500.00 kN.m, N=2000.00 kN within tolerance. "
                    "Best residuals: dN=-3.487 kN, dM=-26.751 kN.m. "
                    "Target may be outside section capacity envelope."
                )

        monkeypatch.setattr(
            CrackingCheck,
            "_get_diagram",
            lambda self, ignore_compression_steel=False: _FailingDiagram(
                eps_top=0.0, eps_bottom=0.0
            ),
        )

        out = check.calculate_detailed(
            M_Ed=500.0,
            N_Ed=2000.0,
            force_cracked=True,
            suppress_warnings=True,
        )

        assert out.solved is False
        assert out.solver_stage == "cracked_state"
        assert out.solver_residual_N == pytest.approx(-3.487, rel=1e-12)
        assert out.solver_residual_M == pytest.approx(-26.751, rel=1e-12)

    def test_net_tension_face_override_in_perform_and_detailed(self, monkeypatch, concrete_c30):
        """Test net tension face override in perform and detailed."""
        check = CrackingCheck(
            section=_make_section(),
            concrete=concrete_c30,
            net_tension_face="top",
            apply_nonlinear_creep=False,
        )

        monkeypatch.setattr(CrackingCheck, "find_cracking_moment", lambda self, N_Ed=0.0: 0.0)
        monkeypatch.setattr(
            CrackingCheck,
            "_get_diagram",
            lambda self, ignore_compression_steel=False: _FakeDiagram(
                eps_top=-0.0010, eps_bottom=-0.0004
            ),
        )
        monkeypatch.setattr(
            CrackingCheck,
            "_get_peak_concrete_stress",
            lambda self, eps_top, eps_bottom, diagram=None, **kw: 6.0,
        )

        captured_faces = []

        def _fake_face(
            self,
            eps_top,
            eps_bottom,
            face,
            x,
            is_net_tension,
            suppress_warnings=False,
            actual_bar_diameter=None,
            **kw,
        ):
            captured_faces.append({"face": face, "is_net_tension": is_net_tension})
            return CrackingResult(
                w_k=0.12,
                w_k_limit=self.w_k_limit,
                s_r_max=120.0,
                eps_sm_minus_eps_cm=0.001,
                sigma_s=200.0,
                rho_p_eff=0.02,
                h_c_ef=100.0,
                x=x,
                is_cracked=True,
                phi_eq=20.0,
                cover=35.0,
            )

        monkeypatch.setattr(CrackingCheck, "_calculate_face_crack_width", _fake_face)

        out = check.perform_check(
            M_Ed=100.0,
            N_Ed=0.0,
            force_cracked=True,
            suppress_warnings=True,
        )
        detailed = check.calculate_detailed(
            M_Ed=100.0,
            N_Ed=0.0,
            force_cracked=True,
            suppress_warnings=True,
        )

        assert len(captured_faces) == 2
        assert all(c["face"] == "top" and c["is_net_tension"] for c in captured_faces)
        assert out.details["governing_face"] == "top"
        assert detailed.governing_face == "top"

    def test_get_diagram_sets_internal_crack_detection_kwargs(self, monkeypatch, concrete_c30):
        """Cracked-analysis diagram should enforce internal crack-detection settings."""
        check = CrackingCheck(
            section=_make_section(),
            concrete=concrete_c30,
        )

        captured: dict = {}

        def _fake_create(**kwargs):
            captured.update(kwargs)
            return _FakeDiagram(eps_top=0.0, eps_bottom=0.0)

        monkeypatch.setattr(cc_mod, "create_interaction_diagram", _fake_create)
        _ = check._get_diagram(ignore_compression_steel=False)

        assert captured["include_tension"] is True
        assert captured["crack_to_neutral_axis_on_first_tension_failure"] is True

    def test_perform_check_uses_solver_crack_decision_not_mcr(self, monkeypatch, concrete_c30):
        """Crack-state branch should follow solver result, independent of M_cr helper."""
        check = CrackingCheck(section=_make_section(), concrete=concrete_c30)

        # Deliberately huge helper cracking moment to prove branch is not M_cr-driven.
        monkeypatch.setattr(CrackingCheck, "find_cracking_moment", lambda self, N_Ed=0.0: 1e9)
        monkeypatch.setattr(
            CrackingCheck,
            "_is_cracked_by_solver",
            lambda self, M_Ed, N_Ed, ignore_compression_steel=False, _mz_kw=None: (
                True, 0.0008, -0.0008, -0.00012, -0.00010
            ),
        )
        monkeypatch.setattr(
            CrackingCheck,
            "_get_diagram",
            lambda self, ignore_compression_steel=False: _FakeDiagram(
                eps_top=0.0008,
                eps_bottom=-0.0008,
            ),
        )
        monkeypatch.setattr(
            CrackingCheck,
            "_get_peak_concrete_stress",
            lambda self, eps_top, eps_bottom, diagram=None, **kw: 5.0,
        )

        monkeypatch.setattr(
            CrackingCheck,
            "_calculate_face_crack_width",
            lambda self, eps_top, eps_bottom, face, x, is_net_tension, suppress_warnings=False, actual_bar_diameter=None, **kw: CrackingResult(
                w_k=0.10,
                w_k_limit=self.w_k_limit,
                s_r_max=100.0,
                eps_sm_minus_eps_cm=0.001,
                sigma_s=200.0,
                rho_p_eff=0.01,
                h_c_ef=100.0,
                x=x,
                is_cracked=True,
                phi_eq=20.0,
                cover=35.0,
            ),
        )

        out = check.perform_check(
            M_Ed=20.0,
            N_Ed=0.0,
            force_cracked=False,
            suppress_warnings=True,
        )

        assert out.details["is_cracked"] is True
        assert out.details["crack_detection_method"] == "solver_uncracked_tension_threshold"


class TestBiaxialEntryPointConsistency:
    """perform_check (via _check_single_case) and calculate_detailed must agree on the
    crack width for biaxial loads. They diverged because calculate_detailed omitted
    the biaxial effective-depth (d_override) that _check_single_case applies, so the
    detailed path fell back to a face-based 1D effective depth."""

    @staticmethod
    def _asymmetric_section():
        # More/larger steel bottom-left than bottom-right => I_xy != 0, so the free
        # neutral axis is skewed even under pure My => the strain state is biaxial.
        section = create_rectangular_section(width=300.0, height=500.0)
        section.add_rebar_group(create_linear_rebar_layer(
            rebar=Rebar(diameter=25, grade="B500B"), n_bars=2,
            start_point=(50.0, 50.0), end_point=(120.0, 50.0), layer_name="bl"))
        section.add_rebar_group(create_linear_rebar_layer(
            rebar=Rebar(diameter=16, grade="B500B"), n_bars=1,
            start_point=(250.0, 50.0), end_point=(250.0, 50.0), layer_name="br"))
        section.add_rebar_group(create_linear_rebar_layer(
            rebar=Rebar(diameter=16, grade="B500B"), n_bars=2,
            start_point=(50.0, 450.0), end_point=(250.0, 450.0), layer_name="top"))
        return section

    def test_perform_and_detailed_agree_for_biaxial(self, concrete_c30):
        check = CrackingCheck(
            section=self._asymmetric_section(), concrete=concrete_c30, free_neutral_axis=True,
        )
        # Confirm the biaxial (free-NA) diagram is actually in use, else the
        # d_override path is not exercised and the test would prove nothing.
        assert hasattr(check._get_diagram(), "get_capacity_biaxial")

        # Load chosen to crack this heavily-reinforced asymmetric section so the
        # biaxial crack-width (and d_override) path is genuinely exercised.
        kw = dict(My_Ed=80.0, N_Ed=0.0)
        pc = check.perform_check(suppress_warnings=True, **kw)
        cd = check.calculate_detailed(suppress_warnings=True, **kw)

        assert cd.w_k is not None and cd.w_k > 0.0
        assert pc.details["w_k"] == pytest.approx(cd.w_k, rel=1e-6)


class TestBiaxialNetCompression:
    """The net-compression short-circuit must use the minimum fibre strain over the
    section for biaxial states, not just the vertical faces — otherwise a skewed-NA
    state with both faces in compression but a corner in tension is wrongly reported
    as uncracked (non-conservative)."""

    def test_corner_tension_is_not_net_compression(self, concrete_c30):
        from section_design_checks.reinforced_concrete.analysis.strain_state import StrainState

        check = CrackingCheck(section=_make_section(width=300.0, height=500.0), concrete=concrete_c30)
        # Uniform vertical strain (both faces in compression, +100e-6) but a horizontal
        # gradient (plane_a < 0) drives the right edge (x_rel = +150 mm) into tension:
        # strain(150, y) = -1e-6*150 + 100e-6 = -50e-6 < 0.
        ss = StrainState(
            eps_top=100e-6, eps_bottom=100e-6,
            plane_a=-1e-6, plane_b=0.0, plane_c=100e-6, is_biaxial=True,
        )
        # Vertical-only logic would call this net compression; the fix must not.
        assert check._is_net_compression(100e-6, 100e-6, ss) is False

    def test_1d_fallback_unchanged(self, concrete_c30):
        check = CrackingCheck(section=_make_section(), concrete=concrete_c30)
        assert check._is_net_compression(100e-6, 100e-6, None) is True
        assert check._is_net_compression(100e-6, -50e-6, None) is False
