"""
Deterministic unit tests for ShearCheck helper logic (solver-independent).
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest
from shapely.geometry import Polygon

import materials.reinforced_concrete.code_checks.ec2_2004.shear_check as sc_mod
from materials.reinforced_concrete.code_checks.base_check import CheckResult, CheckStatus
from materials.reinforced_concrete.code_checks.ec2_2004.shear_check import (
    ShearCheck,
    ShearLoadCase,
)
from materials.reinforced_concrete.constitutive import ConcreteModelType
from materials.reinforced_concrete.materials import ShearRebar


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
    object.__setattr__(check, "apply_tension_cot_theta_limit", True)
    object.__setattr__(check, "d_fallback", "ratio_of_h")
    object.__setattr__(check, "d_ratio", 0.9)
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
    """Tests for TestSnapshotAndDiagramCache."""
    def test_take_snapshot_includes_override_keys(self):
        """Test take snapshot includes override keys."""
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
        """Test get diagram cache reuse and rebuild."""
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


class TestValidation:
    """Tests for TestValidation."""
    def test_linear_elastic_allowed_with_override(self, rectangular_beam_with_rebars, concrete_c30):
        """Test linear elastic allowed with override."""
        check = ShearCheck(
            section=rectangular_beam_with_rebars,
            concrete=concrete_c30,
            concrete_model_type=ConcreteModelType.LINEAR_ELASTIC,
            concrete_model_override=SimpleNamespace(cache_key="override"),
        )
        assert check.concrete_model_type == ConcreteModelType.LINEAR_ELASTIC

    def test_linear_elastic_without_override_raises(self, rectangular_beam_with_rebars, concrete_c30):
        """Test linear elastic without override raises."""
        with pytest.raises(ValueError, match="LINEAR_ELASTIC concrete model is only valid for SLS checks"):
            ShearCheck(
                section=rectangular_beam_with_rebars,
                concrete=concrete_c30,
                concrete_model_type=ConcreteModelType.LINEAR_ELASTIC,
            )


class TestPropertiesAndDepths:
    """Tests for TestPropertiesAndDepths."""
    def test_design_properties_and_breadth_override(self, monkeypatch):
        """Test design properties and breadth override."""
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
        """Test find effective depth and lever arm branches."""
        check = _make_stub_shear_check()

        # Pure shear -> fallback (0.9 * h = 0.9 * 500 = 450).
        d = check.find_effective_depth(M_Ed=0.0, N_Ed=100.0)
        assert d == pytest.approx(450.0, rel=1e-12)

        # Clear compression/tension split → compression at top → d_top=450.
        d2 = check.find_effective_depth(M_Ed=10.0, N_Ed=0.0, eps_top=0.001, eps_bottom=-0.001)
        assert d2 == pytest.approx(450.0, rel=1e-12)

        # Both faces in tension -> fallback (0.9 * h = 450).
        d3 = check.find_effective_depth(M_Ed=10.0, N_Ed=0.0, eps_top=-1e-4, eps_bottom=-2e-4, warn_on_fallback=False)
        assert d3 == pytest.approx(450.0, rel=1e-12)

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

    def test_find_effective_depth_fallback_and_warning_branches(self):
        """Test find effective depth fallback and warning branches."""
        check = _make_stub_shear_check()
        _bbox = lambda: (0.0, 0.0, 300.0, 500.0)  # h=500, 0.9h=450

        # Neither face has rebar → with ratio_of_h policy, still returns 0.9h (no error).
        object.__setattr__(
            check,
            "section",
            SimpleNamespace(
                get_effective_depth=lambda compression_face="top": (_ for _ in ()).throw(ValueError("none")),
                get_bounding_box=_bbox,
            ),
        )
        d_no_rebar = check.find_effective_depth(M_Ed=10.0, N_Ed=0.0, eps_top=0.001, eps_bottom=-0.001, warn_on_fallback=False)
        assert d_no_rebar == pytest.approx(450.0, rel=1e-12)

        # M_Ed=0 → fallback (0.9h = 450).
        object.__setattr__(
            check,
            "section",
            SimpleNamespace(
                get_effective_depth=lambda compression_face="top": 450.0 if compression_face == "top" else (_ for _ in ()).throw(ValueError("no")),
                get_bounding_box=_bbox,
            ),
        )
        assert check.find_effective_depth(M_Ed=0.0, N_Ed=0.0) == pytest.approx(450.0, rel=1e-12)

        # M_Ed=0 with only bottom rebar → still fallback (0.9h = 450).
        object.__setattr__(
            check,
            "section",
            SimpleNamespace(
                get_effective_depth=lambda compression_face="top": (_ for _ in ()).throw(ValueError("no")) if compression_face == "top" else 430.0,
                get_bounding_box=_bbox,
            ),
        )
        assert check.find_effective_depth(M_Ed=0.0, N_Ed=0.0) == pytest.approx(450.0, rel=1e-12)

        # Strain solve fails -> warning fallback to 0.9h.
        object.__setattr__(
            check,
            "section",
            SimpleNamespace(
                get_effective_depth=lambda compression_face="top": 450.0 if compression_face == "top" else 430.0,
                get_bounding_box=_bbox,
            ),
        )
        object.__setattr__(
            check,
            "_get_diagram",
            lambda ignore_compression_steel=False: SimpleNamespace(
                find_strains_for_MN=lambda M_Ed, N_Ed: (_ for _ in ()).throw(RuntimeError("boom"))
            ),
        )
        with pytest.warns(UserWarning, match="strain state unavailable"):
            out = check.find_effective_depth(M_Ed=10.0, N_Ed=0.0, eps_top=None, eps_bottom=None, warn_on_fallback=True)
        assert out == pytest.approx(450.0, rel=1e-12)

        # Both faces in tension -> fallback warning.
        with pytest.warns(UserWarning, match="no compression/tension split"):
            out2 = check.find_effective_depth(
                M_Ed=10.0,
                N_Ed=0.0,
                eps_top=-1e-4,
                eps_bottom=-2e-4,
                warn_on_fallback=True,
            )
        assert out2 == pytest.approx(450.0, rel=1e-12)

        # Compression top, but no bottom tension rebar depth -> fallback warning.
        object.__setattr__(
            check,
            "section",
            SimpleNamespace(
                get_effective_depth=lambda compression_face="top": 450.0 if compression_face == "bottom" else (_ for _ in ()).throw(ValueError("no")),
                get_bounding_box=_bbox,
            ),
        )
        with pytest.warns(UserWarning, match="no rebar in tension zone"):
            out3 = check.find_effective_depth(
                M_Ed=10.0,
                N_Ed=0.0,
                eps_top=0.001,
                eps_bottom=-0.001,
                warn_on_fallback=True,
            )
        assert out3 == pytest.approx(450.0, rel=1e-12)

        # Compression bottom, but no top tension rebar depth -> fallback warning.
        object.__setattr__(
            check,
            "section",
            SimpleNamespace(
                get_effective_depth=lambda compression_face="top": 450.0 if compression_face == "top" else (_ for _ in ()).throw(ValueError("no")),
                get_bounding_box=_bbox,
            ),
        )
        with pytest.warns(UserWarning, match="no rebar in tension zone"):
            out4 = check.find_effective_depth(
                M_Ed=10.0,
                N_Ed=0.0,
                eps_top=-0.001,
                eps_bottom=0.001,
                warn_on_fallback=True,
            )
        assert out4 == pytest.approx(450.0, rel=1e-12)

    def test_find_rho_l_branches(self, monkeypatch):
        """Test find rho l branches."""
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

        # Approximate centroid fallback with tension bars -> clamp to 0.02.
        object.__setattr__(check, "use_rigorous", False)
        check.section.rebar_groups = [
            SimpleNamespace(
                rebar=SimpleNamespace(area=6000.0),
                positions=[SimpleNamespace(y=100.0), SimpleNamespace(y=120.0)],
            )
        ]
        rho = check._find_rho_l(M_Ed=10.0, N_Ed=0.0, d=200.0)
        assert rho == pytest.approx(0.02, rel=1e-12)


class TestSigmaAndCapacityHelpers:
    """Tests for TestSigmaAndCapacityHelpers."""
    def test_sigma_cp_policies(self, monkeypatch):
        """Test sigma cp policies."""
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
        """Test vrd helpers and k."""
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

    def test_vrdc_uncracked_guard_branches(self):
        """Test vrdc uncracked guard branches."""
        check = _make_stub_shear_check()

        # breadth guard
        object.__setattr__(check, "breadth_override", 0.0)
        assert check.find_V_Rd_c_uncracked(sigma_cp=1.0) == pytest.approx(0.0, abs=1e-12)
        object.__setattr__(check, "breadth_override", 300.0)

        # zero second moment guard
        object.__setattr__(check, "section", SimpleNamespace(
            get_second_moment_area=lambda: (0.0, 0.0, 0.0),
            get_centroid=lambda: (150.0, 250.0),
            outline=SimpleNamespace(bounds=(0.0, 0.0, 300.0, 500.0), intersection=lambda *_: SimpleNamespace(area=10.0, centroid=SimpleNamespace(y=300.0))),
        ))
        assert check.find_V_Rd_c_uncracked(sigma_cp=1.0) == pytest.approx(0.0, abs=1e-12)

        # zero top area guard
        object.__setattr__(check, "section", SimpleNamespace(
            get_second_moment_area=lambda: (1.0e9, 0.0, 0.0),
            get_centroid=lambda: (150.0, 250.0),
            outline=SimpleNamespace(bounds=(0.0, 0.0, 300.0, 500.0), intersection=lambda *_: SimpleNamespace(area=0.0, centroid=SimpleNamespace(y=300.0))),
        ))
        assert check.find_V_Rd_c_uncracked(sigma_cp=1.0) == pytest.approx(0.0, abs=1e-12)

        # non-positive first moment guard
        object.__setattr__(check, "section", SimpleNamespace(
            get_second_moment_area=lambda: (1.0e9, 0.0, 0.0),
            get_centroid=lambda: (150.0, 250.0),
            outline=SimpleNamespace(bounds=(0.0, 0.0, 300.0, 500.0), intersection=lambda *_: SimpleNamespace(area=10.0, centroid=SimpleNamespace(y=250.0))),
        ))
        assert check.find_V_Rd_c_uncracked(sigma_cp=1.0) == pytest.approx(0.0, abs=1e-12)

        # negative inner term guard
        object.__setattr__(check, "section", SimpleNamespace(
            get_second_moment_area=lambda: (1.0e9, 0.0, 0.0),
            get_centroid=lambda: (150.0, 250.0),
            outline=SimpleNamespace(bounds=(0.0, 0.0, 300.0, 500.0), intersection=lambda *_: SimpleNamespace(area=10.0, centroid=SimpleNamespace(y=300.0))),
        ))
        assert check.find_V_Rd_c_uncracked(sigma_cp=-10.0) == pytest.approx(0.0, abs=1e-12)


class TestCotThetaAndNote2:
    """Tests for TestCotThetaAndNote2."""
    def test_cot_theta_limits_and_solver_routing(self, monkeypatch):
        """Test cot theta limits and solver routing."""
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
        """Test note 2 iteration branches."""
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

        object.__setattr__(check, "shear_reinforcement", None)
        with pytest.raises(ValueError, match="without shear reinforcement"):
            check._find_V_Rd_max_with_note_2_iteration(V_Ed=100.0, z=400.0, sigma_cp=1.0)


class TestRemainingPublicHelpers:
    """Tests for TestRemainingPublicHelpers."""
    def test_perform_check_wrapper_and_required_reinf(self, monkeypatch):
        """Test perform check wrapper and required reinf."""
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
        """Test min shear reinforcement helper."""
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

    def test_required_reinforcement_uses_default_fywd_without_shear_rebar(self, monkeypatch):
        """Test required reinforcement uses default fywd without shear rebar."""
        check = _make_stub_shear_check()
        object.__setattr__(check, "shear_reinforcement", None)
        object.__setattr__(check, "use_accidental", False)

        monkeypatch.setattr(ShearCheck, "find_effective_depth", lambda self, M_Ed, N_Ed: 400.0)
        monkeypatch.setattr(ShearCheck, "_find_sigma_cp", lambda self, N_Ed: 1.0)
        monkeypatch.setattr(ShearCheck, "_find_rho_l", lambda self, M_Ed, N_Ed, d: 0.01)
        monkeypatch.setattr(ShearCheck, "find_V_Rd_c", lambda self, d, rho_l, sigma_cp: 10.0)
        monkeypatch.setattr(ShearCheck, "find_lever_arm", lambda self, M_Ed, N_Ed, d: (360.0, None))
        monkeypatch.setattr(ShearCheck, "_find_cot_theta_limits", lambda self, sigma_cp, z, V_Ed: (1.0, 2.0))
        monkeypatch.setattr(sc_mod, "clamp_cot_theta", lambda cot_theta, cot_min, cot_max: 1.5)
        monkeypatch.setattr(ShearCheck, "_find_min_a_sw_over_s", lambda self, use_defaults=False: 0.0)

        req_persistent = check.get_required_shear_reinforcement(
            V_Ed=100.0,
            M_Ed=10.0,
            N_Ed=0.0,
            cot_theta=1.5,
            f_ywd=None,
        )
        assert req_persistent > 0.0

        object.__setattr__(check, "use_accidental", True)
        req_accidental = check.get_required_shear_reinforcement(
            V_Ed=100.0,
            M_Ed=10.0,
            N_Ed=0.0,
            cot_theta=1.5,
            f_ywd=None,
        )
        assert req_accidental > 0.0


class TestCheckSingleCaseBranching:
    """Tests for TestCheckSingleCaseBranching."""
    def _patch_common(self, monkeypatch, check, *, vrd_c: float, vrd_c_un: float, vrd_s: float, vrd_max: float):
        monkeypatch.setattr(
            ShearCheck,
            "find_effective_depth",
            lambda self, M_Ed, N_Ed, eps_top=None, eps_bottom=None, ignore_compression_steel=False: 400.0,
        )
        monkeypatch.setattr(
            ShearCheck,
            "_get_diagram",
            lambda self, ignore_compression_steel=False: SimpleNamespace(
                find_strains_for_MN=lambda M_Ed, N_Ed: (0.001, -0.001)
            ),
        )
        monkeypatch.setattr(ShearCheck, "_find_sigma_cp", lambda self, N_Ed: 1.0)
        monkeypatch.setattr(
            ShearCheck,
            "_find_rho_l",
            lambda self, M_Ed, N_Ed, d, eps_top=None, eps_bottom=None, ignore_compression_steel=False: 0.01,
        )
        monkeypatch.setattr(ShearCheck, "find_V_Rd_c", lambda self, d, rho_l, sigma_cp: vrd_c)
        monkeypatch.setattr(ShearCheck, "find_V_Rd_c_uncracked", lambda self, sigma_cp: vrd_c_un)
        monkeypatch.setattr(
            ShearCheck,
            "find_lever_arm",
            lambda self, M_Ed, N_Ed, d, eps_top=None, eps_bottom=None, ignore_compression_steel=False: (360.0, None),
        )
        monkeypatch.setattr(ShearCheck, "_find_cot_theta_limits", lambda self, sigma_cp, z, V_Ed: (1.0, 2.0))
        monkeypatch.setattr(ShearCheck, "_find_cot_theta_for_V_Ed", lambda self, **kwargs: 1.5)
        monkeypatch.setattr(ShearCheck, "find_V_Rd_s", lambda self, cot_theta, z, use_note_2=False: vrd_s)
        monkeypatch.setattr(ShearCheck, "find_V_Rd_max", lambda self, cot_theta, z, sigma_cp, use_note_2=False: vrd_max)
        monkeypatch.setattr(sc_mod, "find_max_allowable_link_spacing", lambda **kwargs: 999.0)

    def test_zero_moment_path_sets_none_strains(self, monkeypatch):
        """Test zero moment path sets none strains."""
        check = _make_stub_shear_check()
        object.__setattr__(check, "shear_reinforcement", None)
        captured = {}

        def _find_d(self, M_Ed, N_Ed, eps_top=None, eps_bottom=None, ignore_compression_steel=False):
            captured["eps"] = (eps_top, eps_bottom)
            return 400.0

        monkeypatch.setattr(ShearCheck, "find_effective_depth", _find_d)
        monkeypatch.setattr(ShearCheck, "_find_sigma_cp", lambda self, N_Ed: 1.0)
        monkeypatch.setattr(
            ShearCheck,
            "_find_rho_l",
            lambda self, M_Ed, N_Ed, d, eps_top=None, eps_bottom=None, ignore_compression_steel=False: 0.01,
        )
        monkeypatch.setattr(ShearCheck, "find_V_Rd_c", lambda self, d, rho_l, sigma_cp: 100.0)
        monkeypatch.setattr(ShearCheck, "find_V_Rd_c_uncracked", lambda self, sigma_cp: 90.0)
        monkeypatch.setattr(ShearCheck, "find_V_Rd_c_max_unreinforced", lambda self, d: 120.0)
        monkeypatch.setattr(
            ShearCheck,
            "_get_diagram",
            lambda self, ignore_compression_steel=False: (_ for _ in ()).throw(RuntimeError("should not solve strains")),
        )

        check._check_single_case(
            V_Ed=50.0,
            M_Ed=0.0,
            N_Ed=0.0,
            cot_theta_override=None,
            use_v_rd_s_for_cot_theta=False,
            use_uncracked_V_Rd_c=False,
            warning_threshold=0.95,
            suppress_warnings=True,
        )
        assert captured["eps"] == (None, None)

    def test_unreinforced_vrdcmax_governing_and_message(self, monkeypatch):
        """Test unreinforced vrdcmax governing and message."""
        check = _make_stub_shear_check()
        object.__setattr__(check, "shear_reinforcement", None)

        monkeypatch.setattr(
            ShearCheck,
            "find_effective_depth",
            lambda self, M_Ed, N_Ed, eps_top=None, eps_bottom=None, ignore_compression_steel=False: 400.0,
        )
        monkeypatch.setattr(ShearCheck, "_find_sigma_cp", lambda self, N_Ed: 1.0)
        monkeypatch.setattr(
            ShearCheck,
            "_find_rho_l",
            lambda self, M_Ed, N_Ed, d, eps_top=None, eps_bottom=None, ignore_compression_steel=False: 0.01,
        )
        monkeypatch.setattr(ShearCheck, "find_V_Rd_c", lambda self, d, rho_l, sigma_cp: 200.0)
        monkeypatch.setattr(ShearCheck, "find_V_Rd_c_uncracked", lambda self, sigma_cp: 150.0)
        monkeypatch.setattr(ShearCheck, "find_V_Rd_c_max_unreinforced", lambda self, d: 100.0)

        out = check._check_single_case(
            V_Ed=150.0,
            M_Ed=0.0,
            N_Ed=0.0,
            cot_theta_override=None,
            use_v_rd_s_for_cot_theta=False,
            use_uncracked_V_Rd_c=False,
            warning_threshold=0.95,
            suppress_warnings=True,
        )
        assert out.details["governing_component"] == "V_Rd_c_max"
        assert "diagonal compression limit" in out.message

    def test_reinforced_zcap_and_vrds_solver_paths(self, monkeypatch):
        """Test reinforced zcap and vrds solver paths."""
        check = _make_stub_shear_check()
        self._patch_common(monkeypatch, check, vrd_c=100.0, vrd_c_un=90.0, vrd_s=220.0, vrd_max=250.0)
        monkeypatch.setattr(sc_mod, "get_ndp", lambda key: (lambda d, d_2: 300.0) if key == "z_cap" else None)

        out = check._check_single_case(
            V_Ed=120.0,
            M_Ed=0.0,
            N_Ed=0.0,
            cot_theta_override=None,
            use_v_rd_s_for_cot_theta=True,
            use_uncracked_V_Rd_c=False,
            warning_threshold=0.95,
            suppress_warnings=True,
        )
        assert out.details["K"] is None
        assert out.details["z"] == pytest.approx(300.0, rel=1e-12)

    def test_reinforced_cot_theta_override_warning_branches(self, monkeypatch):
        """Test reinforced cot theta override warning branches."""
        check = _make_stub_shear_check()
        self._patch_common(monkeypatch, check, vrd_c=100.0, vrd_c_un=90.0, vrd_s=220.0, vrd_max=250.0)
        monkeypatch.setattr(sc_mod, "get_ndp", lambda key: None)

        with pytest.warns(UserWarning, match="greater than max value"):
            check._check_single_case(
                V_Ed=120.0,
                M_Ed=10.0,
                N_Ed=0.0,
                cot_theta_override=3.0,
                use_v_rd_s_for_cot_theta=False,
                use_uncracked_V_Rd_c=False,
                warning_threshold=0.95,
                suppress_warnings=False,
            )

        with pytest.warns(UserWarning, match="smaller than min value"):
            check._check_single_case(
                V_Ed=120.0,
                M_Ed=10.0,
                N_Ed=0.0,
                cot_theta_override=0.5,
                use_v_rd_s_for_cot_theta=False,
                use_uncracked_V_Rd_c=False,
                warning_threshold=0.95,
                suppress_warnings=False,
            )

    def test_reinforced_note2_and_governing_mode_branches(self, monkeypatch):
        """Test reinforced note2 and governing mode branches."""
        check = _make_stub_shear_check()
        self._patch_common(monkeypatch, check, vrd_c=100.0, vrd_c_un=90.0, vrd_s=250.0, vrd_max=200.0)
        monkeypatch.setattr(sc_mod, "get_ndp", lambda key: None)
        object.__setattr__(check, "use_increased_nu_1", True)
        monkeypatch.setattr(
            ShearCheck,
            "_find_V_Rd_max_with_note_2_iteration",
            lambda self, *args, **kwargs: (180.0, True),
        )

        out_a = check._check_single_case(
            V_Ed=160.0,
            M_Ed=10.0,
            N_Ed=0.0,
            cot_theta_override=None,
            use_v_rd_s_for_cot_theta=False,
            use_uncracked_V_Rd_c=False,
            warning_threshold=0.95,
            suppress_warnings=True,
        )
        assert out_a.details["used_note_2"] is True
        assert out_a.details["governing_component"] == "V_Rd_max"
        assert "compression strut (V_Rd,max)" in out_a.message

        # Force concrete-governing branches when V_Ed <= V_Rd_c.
        self._patch_common(monkeypatch, check, vrd_c=0.0, vrd_c_un=0.0, vrd_s=500.0, vrd_max=500.0)
        object.__setattr__(check, "use_increased_nu_1", False)
        out_cr = check._check_single_case(
            V_Ed=0.0,
            M_Ed=10.0,
            N_Ed=0.0,
            cot_theta_override=None,
            use_v_rd_s_for_cot_theta=False,
            use_uncracked_V_Rd_c=False,
            warning_threshold=0.95,
            suppress_warnings=True,
        )
        assert out_cr.details["governing_component"] == "V_Rd_c_cracked"
        assert "cracked concrete shear resistance" in out_cr.message

        out_un = check._check_single_case(
            V_Ed=0.0,
            M_Ed=10.0,
            N_Ed=0.0,
            cot_theta_override=None,
            use_v_rd_s_for_cot_theta=False,
            use_uncracked_V_Rd_c=True,
            warning_threshold=0.95,
            suppress_warnings=True,
        )
        assert out_un.details["governing_component"] == "V_Rd_c_uncracked"
        assert "uncracked concrete shear resistance" in out_un.message

    def test_reinforced_vrdmax_governs_when_ved_below_vrdc(self, monkeypatch):
        """Test reinforced vrdmax governs when ved below vrdc."""
        check = _make_stub_shear_check()
        self._patch_common(monkeypatch, check, vrd_c=120.0, vrd_c_un=100.0, vrd_s=300.0, vrd_max=80.0)
        monkeypatch.setattr(sc_mod, "get_ndp", lambda key: None)

        out = check._check_single_case(
            V_Ed=60.0,
            M_Ed=10.0,
            N_Ed=0.0,
            cot_theta_override=None,
            use_v_rd_s_for_cot_theta=False,
            use_uncracked_V_Rd_c=False,
            warning_threshold=0.95,
            suppress_warnings=True,
        )
        assert out.details["governing_component"] == "V_Rd_max"
        assert out.code_reference == "EC2 §6.2.3 (Eq. 6.9)"

    def test_unreinforced_uncracked_failure_message(self, monkeypatch):
        """Test unreinforced uncracked failure message."""
        check = _make_stub_shear_check()
        object.__setattr__(check, "shear_reinforcement", None)

        monkeypatch.setattr(
            ShearCheck,
            "find_effective_depth",
            lambda self, M_Ed, N_Ed, eps_top=None, eps_bottom=None, ignore_compression_steel=False: 400.0,
        )
        monkeypatch.setattr(
            ShearCheck,
            "_find_rho_l",
            lambda self, M_Ed, N_Ed, d, eps_top=None, eps_bottom=None, ignore_compression_steel=False: 0.01,
        )
        monkeypatch.setattr(ShearCheck, "_find_sigma_cp", lambda self, N_Ed: 1.0)
        monkeypatch.setattr(ShearCheck, "find_V_Rd_c", lambda self, d, rho_l, sigma_cp: 50.0)
        monkeypatch.setattr(ShearCheck, "find_V_Rd_c_uncracked", lambda self, sigma_cp: 40.0)
        monkeypatch.setattr(ShearCheck, "find_V_Rd_c_max_unreinforced", lambda self, d: 100.0)

        out = check._check_single_case(
            V_Ed=60.0,
            M_Ed=0.0,
            N_Ed=0.0,
            cot_theta_override=None,
            use_v_rd_s_for_cot_theta=False,
            use_uncracked_V_Rd_c=True,
            warning_threshold=0.95,
            suppress_warnings=True,
        )
        assert out.details["governing_component"] == "V_Rd_c_uncracked"
        assert "member without shear reinforcement" in out.message

    def test_reinforced_compression_strut_failure_message(self, monkeypatch):
        """Test reinforced compression strut failure message."""
        check = _make_stub_shear_check()
        self._patch_common(monkeypatch, check, vrd_c=50.0, vrd_c_un=40.0, vrd_s=200.0, vrd_max=100.0)
        monkeypatch.setattr(sc_mod, "get_ndp", lambda key: None)

        out = check._check_single_case(
            V_Ed=150.0,
            M_Ed=10.0,
            N_Ed=0.0,
            cot_theta_override=None,
            use_v_rd_s_for_cot_theta=False,
            use_uncracked_V_Rd_c=False,
            warning_threshold=0.95,
            suppress_warnings=True,
        )
        assert out.details["governing_component"] == "V_Rd_max"
        assert out.message == "Shear capacity exceeded: compression strut limit V_Rd,max reached."


class TestPlotWrappers:
    """Tests for ShearCheck plotting wrapper methods."""

    def test_plot_wrappers_delegate_to_viewer(self, monkeypatch):
        check = _make_stub_shear_check()

        class _FakeViewer:
            def __init__(self, c):
                self.check = c

            def plot_cot_theta_study(self, *, load_case, **kwargs):
                return ("cot_theta", load_case, kwargs)

            def plot_cot_theta_moment_shift_study(self, *, load_case, **kwargs):
                return ("cot_theta_moment_shift", load_case, kwargs)

            def plot_link_angle_study(self, *, load_case, **kwargs):
                return ("link_angle", load_case, kwargs)

            def plot_link_angle_moment_shift_study(self, *, load_case, **kwargs):
                return ("link_angle_moment_shift", load_case, kwargs)

            def plot_cot_theta_link_angle_heatmap(self, *, load_case, **kwargs):
                return ("heatmap", load_case, kwargs)

            def plot_axial_cot_theta_contour(self, *, load_case, **kwargs):
                return ("axial", load_case, kwargs)

        fake_module = types.SimpleNamespace(ShearViewer=_FakeViewer)
        monkeypatch.setitem(
            sys.modules,
            "materials.reinforced_concrete.analysis.shear_viewer",
            fake_module,
        )

        load_case = ShearLoadCase(V_Ed=100.0, M_Ed=20.0, N_Ed=30.0)

        out1 = check.plot_cot_theta_study(load_case=load_case, show=False)
        out2 = check.plot_cot_theta_moment_shift_study(load_case=load_case, show=False)
        out3 = check.plot_link_angle_study(load_case=load_case, show=False)
        out4 = check.plot_link_angle_moment_shift_study(load_case=load_case, show=False)
        out5 = check.plot_cot_theta_link_angle_heatmap(load_case=load_case, show=False)
        out6 = check.plot_axial_cot_theta_contour(load_case=load_case, N_min=-100.0, N_max=100.0, show=False)

        assert out1[0] == "cot_theta"
        assert out2[0] == "cot_theta_moment_shift"
        assert out3[0] == "link_angle"
        assert out4[0] == "link_angle_moment_shift"
        assert out5[0] == "heatmap"
        assert out6[0] == "axial"
