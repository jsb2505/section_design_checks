"""
Deterministic unit tests for ShearCheck helper logic (solver-independent).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from shapely.geometry import Polygon

import materials.reinforced_concrete.code_checks.ec2_2004.shear_check as sc_mod
from materials.reinforced_concrete.code_checks.base_check import CheckResult, CheckStatus
from materials.reinforced_concrete.code_checks.ec2_2004.shear_check import (
    ShearCheck,
    ShearLoadCase,
)


def _make_stub_shear_check() -> ShearCheck:
    check = object.__new__(ShearCheck)

    section = SimpleNamespace(
        section_name="S",
        model_dump=lambda: {"section": 1},
        get_transformed_area=lambda E: 120_000.0,
        get_area=lambda: 100_000.0,
        get_centroid=lambda: (150.0, 250.0),
        get_bounding_box=lambda: (0.0, 0.0, 300.0, 500.0),
        get_effective_depth=lambda compression_face="top": 450.0 if compression_face == "top" else 430.0,
        get_second_moment_area=lambda: (1.0e9, 0.0, 0.0),
        get_compression_rebar_depth=lambda compression_face: 30.0,
        outline=Polygon([(0.0, 0.0), (300.0, 0.0), (300.0, 500.0), (0.0, 500.0)]),
        rebar_groups=[],
    )
    concrete = SimpleNamespace(
        model_dump=lambda: {"concrete": 1},
        f_ck=30.0,
        f_ctm=2.9,
        f_ctd=1.3,
        f_ctd_accidental=1.5,
        f_cd_shear=20.0,
        f_cd_shear_accidental=25.0,
        gamma_c=1.5,
        gamma_c_accidental=1.2,
        E_cm=33_000.0,
    )
    shear_reinf = SimpleNamespace(
        area_per_unit_length=1.0,
        angle=90.0,
        f_yk=500.0,
        f_yd=435.0,
        f_yd_accidental=500.0,
        link_spacing=200.0,
        leg_spacing=None,
    )

    object.__setattr__(check, "section", section)
    object.__setattr__(check, "concrete", concrete)
    object.__setattr__(check, "shear_reinforcement", shear_reinf)
    object.__setattr__(check, "use_accidental", False)
    object.__setattr__(check, "use_rigorous", True)
    object.__setattr__(check, "allow_negative_sigma_cp", True)
    object.__setattr__(check, "use_transformed_area_for_sigma_cp", True)
    object.__setattr__(check, "use_sigma_cp_for_alpha_cw", False)
    object.__setattr__(check, "cap_lever_arm", True)
    object.__setattr__(check, "breadth_override", None)
    object.__setattr__(check, "use_increased_nu_1", False)
    object.__setattr__(check, "concrete_model_type", "parabola")
    object.__setattr__(check, "steel_model_type", "inclined")
    object.__setattr__(check, "concrete_model_override", None)
    object.__setattr__(check, "steel_models_override", None)
    object.__setattr__(check, "_diagram", None)
    object.__setattr__(check, "_diagram_no_comp_steel", None)
    object.__setattr__(check, "_diagram_snapshot", None)
    object.__setattr__(check, "_diagram_no_comp_snapshot", None)
    return check


def _ok_result(name: str) -> CheckResult:
    return CheckResult(
        check_name=name,
        status=CheckStatus.PASS,
        utilization=0.5,
        demand=1.0,
        capacity=2.0,
        units="kN",
        message="ok",
        details={},
        code_reference="ref",
    )


class TestSnapshotAndDiagramCache:
    def test_take_snapshot_includes_override_keys(self):
        check = _make_stub_shear_check()
        object.__setattr__(check, "concrete_model_override", SimpleNamespace(cache_key="cc"))
        object.__setattr__(
            check,
            "steel_models_override",
            [SimpleNamespace(cache_key="s1"), SimpleNamespace(cache_key="s2")],
        )
        snap = check._take_snapshot()
        assert snap["section"] == {"section": 1}
        assert snap["concrete"] == {"concrete": 1}
        assert snap["concrete_override_key"] == "cc"
        assert snap["steel_override_keys"] == ["s1", "s2"]

    def test_get_diagram_cache_reuse_and_rebuild(self, monkeypatch):
        check = _make_stub_shear_check()
        calls = {"n": 0}

        class _DummyDiagram:
            def __init__(self, **kwargs):
                calls["n"] += 1
                self.kwargs = kwargs

            def get_lever_arm(self, **kwargs):
                return (400.0, 390.0)

            def find_strains_for_MN(self, M_Ed, N_Ed):
                return (0.001, -0.001)

        monkeypatch.setattr(sc_mod, "MNInteractionDiagram", _DummyDiagram)
        snap = {"k": 1}
        monkeypatch.setattr(ShearCheck, "_take_snapshot", lambda self: dict(snap))

        # Seed cache manually (stub bypasses full Pydantic init for private attrs).
        cached_main = object()
        object.__setattr__(check, "_diagram", cached_main)
        object.__setattr__(check, "_diagram_snapshot", {"k": 1})
        d1 = check._get_diagram(False)
        assert d1 is cached_main
        assert calls["n"] == 0

        # Snapshot changed -> rebuild.
        snap["k"] = 2
        check._get_diagram(False)
        assert calls["n"] == 1

        # Separate cache for ignore compression steel.
        cached_no_comp = object()
        object.__setattr__(check, "_diagram_no_comp_steel", cached_no_comp)
        object.__setattr__(check, "_diagram_no_comp_snapshot", {"k": 2})
        d3 = check._get_diagram(True)
        assert d3 is cached_no_comp
        assert calls["n"] == 1

        snap["k"] = 3
        check._get_diagram(True)
        assert calls["n"] == 2


class TestPropertiesAndDepths:
    def test_design_properties_and_breadth_override(self, monkeypatch):
        check = _make_stub_shear_check()
        monkeypatch.setattr(sc_mod, "calculate_section_breadth", lambda section: 275.0)
        assert check.breadth == pytest.approx(275.0, rel=1e-12)
        object.__setattr__(check, "breadth_override", 300.0)
        assert check.breadth == pytest.approx(300.0, rel=1e-12)

        assert check.f_cd_design == pytest.approx(20.0, rel=1e-12)
        assert check.f_ctd_design == pytest.approx(1.3, rel=1e-12)
        assert check.gamma_c_design == pytest.approx(1.5, rel=1e-12)
        assert check.f_ywd_design == pytest.approx(435.0, rel=1e-12)

        object.__setattr__(check, "use_accidental", True)
        assert check.f_cd_design == pytest.approx(25.0, rel=1e-12)
        assert check.f_ctd_design == pytest.approx(1.5, rel=1e-12)
        assert check.gamma_c_design == pytest.approx(1.2, rel=1e-12)
        assert check.f_ywd_design == pytest.approx(500.0, rel=1e-12)

        object.__setattr__(check, "shear_reinforcement", None)
        assert check.f_ywd_design == pytest.approx(0.0, rel=1e-12)

    def test_find_effective_depth_and_lever_arm_branches(self):
        check = _make_stub_shear_check()

        # Pure shear -> conservative min(d_top,d_bottom).
        d = check.find_effective_depth(M_Ed=0.0, N_Ed=100.0)
        assert d == pytest.approx(430.0, rel=1e-12)

        # Strain-defined compression face.
        d2 = check.find_effective_depth(M_Ed=10.0, N_Ed=0.0, eps_top=0.001, eps_bottom=-0.001)
        assert d2 == pytest.approx(450.0, rel=1e-12)

        # Both faces in tension -> fallback.
        d3 = check.find_effective_depth(M_Ed=10.0, N_Ed=0.0, eps_top=-1e-4, eps_bottom=-2e-4, warn_on_fallback=False)
        assert d3 == pytest.approx(430.0, rel=1e-12)

        # Non-rigorous lever arm.
        object.__setattr__(check, "use_rigorous", False)
        assert check.find_lever_arm(M_Ed=10.0, N_Ed=0.0, d=400.0) == (360.0, None)

        # Rigorous lever arm path delegates to diagram.
        object.__setattr__(check, "use_rigorous", True)
        object.__setattr__(
            check,
            "_get_diagram",
            lambda ignore_compression_steel=False: SimpleNamespace(
                get_lever_arm=lambda **kwargs: (355.0, 340.0)
            ),
        )
        assert check.find_lever_arm(M_Ed=10.0, N_Ed=0.0, d=400.0) == (355.0, 340.0)

    def test_find_rho_l_branches(self, monkeypatch):
        check = _make_stub_shear_check()

        # Approximate mode with strain input delegates to strain helper method.
        object.__setattr__(check, "use_rigorous", False)
        monkeypatch.setattr(ShearCheck, "_compute_rho_l_from_strains", lambda self, eps_top, eps_bottom, d: 0.011)
        assert check._find_rho_l(M_Ed=10.0, N_Ed=0.0, d=400.0, eps_top=0.001, eps_bottom=-0.001) == pytest.approx(0.011, rel=1e-12)

        # Approximate centroid fallback with no tension bars.
        check.section.rebar_groups = [SimpleNamespace(rebar=SimpleNamespace(area=100.0), positions=[SimpleNamespace(y=300.0)])]
        assert check._find_rho_l(M_Ed=10.0, N_Ed=0.0, d=400.0) == pytest.approx(0.0, rel=1e-12)

        # Rigorous mode pulls strains from diagram when missing.
        object.__setattr__(check, "use_rigorous", True)
        object.__setattr__(
            check,
            "_get_diagram",
            lambda ignore_compression_steel=False: SimpleNamespace(find_strains_for_MN=lambda M, N: (0.002, -0.001)),
        )
        monkeypatch.setattr(ShearCheck, "_compute_rho_l_from_strains", lambda self, eps_top, eps_bottom, d: 0.015)
        assert check._find_rho_l(M_Ed=10.0, N_Ed=0.0, d=400.0) == pytest.approx(0.015, rel=1e-12)


class TestSigmaAndCapacityHelpers:
    def test_sigma_cp_policies(self, monkeypatch):
        check = _make_stub_shear_check()
        monkeypatch.setattr(sc_mod, "sigma_cp_from_N_and_area", lambda N_Ed, area: N_Ed / area * 1000.0)
        monkeypatch.setattr(sc_mod, "cap_sigma_cp_upper", lambda sigma_cp, f_cd: min(sigma_cp, 0.2 * f_cd))

        # Negative axial disallowed.
        object.__setattr__(check, "allow_negative_sigma_cp", False)
        assert check._find_sigma_cp(-100.0) == pytest.approx(0.0, rel=1e-12)

        # Transformed area path.
        object.__setattr__(check, "allow_negative_sigma_cp", True)
        object.__setattr__(check, "use_transformed_area_for_sigma_cp", True)
        s1 = check._find_sigma_cp(240.0)
        assert s1 == pytest.approx(min(2.0, 0.2 * 20.0), rel=1e-12)

        # Gross area path.
        object.__setattr__(check, "use_transformed_area_for_sigma_cp", False)
        s2 = check._find_sigma_cp(240.0)
        assert s2 == pytest.approx(min(2.4, 0.2 * 20.0), rel=1e-12)

    def test_vrd_helpers_and_k(self, monkeypatch):
        check = _make_stub_shear_check()
        monkeypatch.setattr(sc_mod, "find_V_Rd_c_cracked", lambda **kwargs: 111.0)
        assert check.find_V_Rd_c(d=400.0, rho_l=0.01, sigma_cp=1.0) == pytest.approx(111.0, rel=1e-12)

        monkeypatch.setattr(sc_mod, "find_V_Rd_c_max_unreinforced", lambda **kwargs: 222.0)
        assert check.find_V_Rd_c_max_unreinforced(d=400.0) == pytest.approx(222.0, rel=1e-12)

        # V_Rd_c_uncracked positive path.
        v_un = check.find_V_Rd_c_uncracked(sigma_cp=1.0)
        assert v_un > 0.0

        # V_Rd_s / V_Rd_max with and without Note 2.
        v_s_n1 = check.find_V_Rd_s(cot_theta=2.0, z=400.0, use_note_2=False)
        v_s_n2 = check.find_V_Rd_s(cot_theta=2.0, z=400.0, use_note_2=True)
        assert v_s_n2 < v_s_n1

        monkeypatch.setattr(sc_mod, "find_alpha_cw", lambda f_cd, sigma_cp, use_sigma_cp_for_alpha_cw=False: 1.0)
        monkeypatch.setattr(sc_mod, "find_nu_1_factor", lambda f_ck, angle: 0.6)
        monkeypatch.setattr(sc_mod, "find_nu_1_factor_note_2", lambda f_ck, angle: 0.8)
        v_max_n1 = check.find_V_Rd_max(cot_theta=2.0, z=400.0, sigma_cp=1.0, use_note_2=False)
        v_max_n2 = check.find_V_Rd_max(cot_theta=2.0, z=400.0, sigma_cp=1.0, use_note_2=True)
        assert v_max_n2 > v_max_n1

        k_n1 = check._calculate_K(z=400.0, sigma_cp=1.0, use_note_2=False)
        k_n2 = check._calculate_K(z=400.0, sigma_cp=1.0, use_note_2=True)
        assert k_n2 > k_n1

        object.__setattr__(check, "shear_reinforcement", None)
        with pytest.raises(ValueError, match=r"without .*shear reinforcement"):
            check.find_V_Rd_s(cot_theta=2.0, z=400.0)
        with pytest.raises(ValueError, match=r"without .*shear reinforcement"):
            check.find_V_Rd_max(cot_theta=2.0, z=400.0, sigma_cp=1.0)
        with pytest.raises(ValueError, match=r"without .*shear reinforcement"):
            check._calculate_K(z=400.0, sigma_cp=1.0)


class TestCotThetaAndNote2:
    def test_cot_theta_limits_and_solver_routing(self, monkeypatch):
        check = _make_stub_shear_check()

        # Constant limits.
        monkeypatch.setattr(sc_mod, "get_ndp", lambda key: 1.0 if key == "cot_theta_lower_lim" else 2.5)
        cot_min, cot_max = check._find_cot_theta_limits(sigma_cp=1.0, z=400.0, V_Ed=100.0)
        assert cot_min == pytest.approx(1.0, rel=1e-12)
        assert cot_max == pytest.approx(2.5, rel=1e-12)

        # Callable max limit.
        def _ndp_callable(key):
            if key == "cot_theta_lower_lim":
                return 1.0
            return lambda **kwargs: 2.2

        monkeypatch.setattr(sc_mod, "get_ndp", _ndp_callable)
        cot_min2, cot_max2 = check._find_cot_theta_limits(sigma_cp=1.0, z=400.0, V_Ed=100.0)
        assert cot_min2 == pytest.approx(1.0, rel=1e-12)
        assert cot_max2 == pytest.approx(2.2, rel=1e-12)

        monkeypatch.setattr(sc_mod, "find_cot_theta_for_V_Ed_from_V_Rd_s", lambda **kwargs: 1.4)
        monkeypatch.setattr(sc_mod, "find_cot_theta_for_V_Ed_from_V_Rd_max", lambda **kwargs: 1.9)
        monkeypatch.setattr(ShearCheck, "_calculate_K", lambda self, z, sigma_cp, use_note_2=False: 123.0)

        out_s = check._find_cot_theta_for_V_Ed(
            V_Ed=100.0,
            z=400.0,
            sigma_cp=1.0,
            cot_min=1.0,
            cot_max=2.5,
            use_note_2=True,
            use_v_rd_s_for_cot_theta=True,
        )
        assert out_s == pytest.approx(1.4, rel=1e-12)

        out_m = check._find_cot_theta_for_V_Ed(
            V_Ed=100.0,
            z=400.0,
            sigma_cp=1.0,
            cot_min=1.0,
            cot_max=2.5,
            use_note_2=False,
            use_v_rd_s_for_cot_theta=False,
        )
        assert out_m == pytest.approx(1.9, rel=1e-12)

        object.__setattr__(check, "shear_reinforcement", None)
        with pytest.raises(ValueError, match="without shear reinforcement"):
            check._find_cot_theta_for_V_Ed(
                V_Ed=100.0,
                z=400.0,
                sigma_cp=1.0,
                cot_min=1.0,
                cot_max=2.5,
            )

    def test_note_2_iteration_branches(self, monkeypatch):
        check = _make_stub_shear_check()
        monkeypatch.setattr(ShearCheck, "_find_cot_theta_limits", lambda self, sigma_cp, z, V_Ed: (1.0, 2.5))

        # A: Note 2 not applicable.
        monkeypatch.setattr(ShearCheck, "_find_cot_theta_for_V_Ed", lambda self, **kwargs: 1.5)
        monkeypatch.setattr(ShearCheck, "find_V_Rd_max", lambda self, cot_theta, z, sigma_cp, use_note_2=False: 500.0 if not use_note_2 else 600.0)
        monkeypatch.setattr(
            ShearCheck,
            "find_V_Rd_s",
            lambda self, cot_theta, z, use_note_2=False: 1.0 if not use_note_2 else 2.0,
        )
        out_a = check._find_V_Rd_max_with_note_2_iteration(V_Ed=100.0, z=400.0, sigma_cp=1.0)
        assert out_a == (500.0, False)

        # B: converges with Note 2.
        monkeypatch.setattr(
            ShearCheck,
            "find_V_Rd_s",
            lambda self, cot_theta, z, use_note_2=False: 500.0 if not use_note_2 else 450.0,
        )
        out_b = check._find_V_Rd_max_with_note_2_iteration(V_Ed=100.0, z=400.0, sigma_cp=1.0)
        assert out_b == (600.0, True)

        # C: oscillation reverts to Note 1.
        monkeypatch.setattr(
            ShearCheck,
            "find_V_Rd_s",
            lambda self, cot_theta, z, use_note_2=False: 500.0 if not use_note_2 else 1.0,
        )
        with pytest.warns(UserWarning, match="Oscillation detected"):
            out_c = check._find_V_Rd_max_with_note_2_iteration(
                V_Ed=100.0,
                z=400.0,
                sigma_cp=1.0,
                suppress_warnings=False,
            )
        assert out_c == (500.0, False)


class TestRemainingPublicHelpers:
    def test_perform_check_wrapper_and_required_reinf(self, monkeypatch):
        check = _make_stub_shear_check()
        monkeypatch.setattr(
            ShearCheck,
            "_check_single_case",
            lambda self, **kwargs: _ok_result("shear-wrapper"),
        )
        out = check.perform_check(load_case=ShearLoadCase(V_Ed=10.0, M_Ed=2.0, N_Ed=1.0))
        assert out.check_name == "shear-wrapper"

        # Required reinforcement: no need branch.
        monkeypatch.setattr(ShearCheck, "find_effective_depth", lambda self, M_Ed, N_Ed: 400.0)
        monkeypatch.setattr(ShearCheck, "_find_sigma_cp", lambda self, N_Ed: 1.0)
        monkeypatch.setattr(ShearCheck, "_find_rho_l", lambda self, M_Ed, N_Ed, d: 0.01)
        monkeypatch.setattr(ShearCheck, "find_V_Rd_c", lambda self, d, rho_l, sigma_cp: 200.0)
        assert check.get_required_shear_reinforcement(V_Ed=150.0, M_Ed=10.0, N_Ed=0.0) == pytest.approx(0.0, rel=1e-12)

        # Required reinforcement: needed branch with clamping + min reinforcement.
        monkeypatch.setattr(ShearCheck, "find_V_Rd_c", lambda self, d, rho_l, sigma_cp: 50.0)
        monkeypatch.setattr(ShearCheck, "find_lever_arm", lambda self, M_Ed, N_Ed, d: (360.0, None))
        monkeypatch.setattr(ShearCheck, "_find_cot_theta_limits", lambda self, sigma_cp, z, V_Ed: (1.0, 2.0))
        monkeypatch.setattr(sc_mod, "clamp_cot_theta", lambda cot_theta, cot_min, cot_max: 2.0)
        monkeypatch.setattr(ShearCheck, "_find_min_a_sw_over_s", lambda self, use_defaults=False: 0.2)
        req = check.get_required_shear_reinforcement(V_Ed=100.0, M_Ed=10.0, N_Ed=0.0, cot_theta=3.0)
        assert req >= 0.2

    def test_min_shear_reinforcement_helper(self, monkeypatch):
        check = _make_stub_shear_check()
        monkeypatch.setattr(sc_mod, "find_minimum_ratio_of_shear_reinforcement", lambda f_ck, f_yk, f_ctm: 0.001)
        object.__setattr__(check, "breadth_override", 300.0)

        # Defaults branch.
        out_default = check._find_min_a_sw_over_s(use_defaults=True)
        assert out_default > 0.0

        # No reinforcement and no defaults -> error.
        object.__setattr__(check, "shear_reinforcement", None)
        with pytest.raises(ValueError, match="must be provided"):
            check._find_min_a_sw_over_s(use_defaults=False)
