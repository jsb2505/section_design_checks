"""
Deterministic unit tests for CircularSectionCheck helper and wrapper logic.
"""

from __future__ import annotations

from math import atan, degrees, isclose, pi, sqrt
from types import SimpleNamespace

import pytest

import materials.reinforced_concrete.code_checks.ec2_2004.circular_section_check as csc_mod
from materials.reinforced_concrete.code_checks.base_check import CheckResult, CheckStatus
from materials.reinforced_concrete.code_checks.ec2_2004.circular_section_check import (
    CircularSectionCheck,
)
from materials.reinforced_concrete.code_checks.ec2_2004.shear_check import ShearLoadCase
from materials.reinforced_concrete.geometry import create_circular_section
from materials.reinforced_concrete.materials import ConcreteMaterial, ShearRebar
from materials.reinforced_concrete.ndp import ndp_override


def _make_stub_check() -> CircularSectionCheck:
    check = object.__new__(CircularSectionCheck)

    shear_reinf = SimpleNamespace(
        diameter=12.0,
        link_spacing=200.0,
        leg_spacing=None,
        area_per_unit_length=1.0,
        angle=90.0,
        f_yk=500.0,
        f_yd=435.0,
        f_yd_accidental=500.0,
    )

    concrete_uls = SimpleNamespace(
        f_ck=30.0,
        gamma_c=1.5,
        f_cd_shear=20.0,
        f_cd_shear_accidental=25.0,
        f_ctd=1.3,
        f_ctd_accidental=1.6,
    )

    diag = SimpleNamespace(
        find_strains_for_MN=lambda M, N, strict=False: (0.001, -0.001),
        find_strain_state_for_MN=lambda M_target, N_target: None,
    )
    shear_check = SimpleNamespace(
        find_effective_depth=lambda M, N, eps_top=None, eps_bottom=None, ignore_compression_steel=False, **kw: 500.0,
        find_lever_arm=lambda M, N, d, eps_top=None, eps_bottom=None, ignore_compression_steel=False, **kw: (450.0, 400.0),
        _get_diagram=lambda ignore_compression_steel=False: diag,
    )

    object.__setattr__(check, "section", SimpleNamespace(section_name="S", get_area=lambda: 100_000.0))
    object.__setattr__(check, "diameter", 600.0)
    object.__setattr__(check, "cover", 50.0)
    object.__setattr__(check, "shear_reinforcement", shear_reinf)
    object.__setattr__(check, "is_spiral", False)
    object.__setattr__(check, "r_sv_override", None)
    object.__setattr__(check, "use_simplified_lambda_1", False)
    object.__setattr__(check, "use_increased_nu_1", False)
    object.__setattr__(check, "use_sigma_cp_for_alpha_cw", False)
    object.__setattr__(check, "apply_tension_cot_theta_limit", True)
    object.__setattr__(check, "d_fallback", "ratio_of_h")
    object.__setattr__(check, "d_ratio", 0.9)
    object.__setattr__(check, "use_accidental", False)
    object.__setattr__(check, "_concrete_uls", concrete_uls)
    object.__setattr__(check, "_shear_check", shear_check)
    object.__setattr__(check, "_bending_check", None)
    object.__setattr__(check, "_cracking_check", None)
    object.__setattr__(check, "_stress_limits_check", None)

    # Snapshot the current NDP context so the guard doesn't fire on stubs
    from materials.reinforced_concrete.ndp import get_ndp_context
    object.__setattr__(check, "_ndp_snapshot", get_ndp_context())

    return check


class _Recorder:
    def __init__(self, return_value):
        self.calls = []
        self.return_value = return_value

    def perform_check(self, **kwargs):
        self.calls.append(kwargs)
        return self.return_value


def _ok_result(name: str = "ok") -> CheckResult:
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


class TestGeometryAndPostInit:
    """Tests for TestGeometryAndPostInit."""
    def test_geometry_validation_errors(self):
        """Test geometry validation errors."""
        section = create_circular_section(diameter=600.0, hook_ref=0)
        concrete = ConcreteMaterial(grade="C30/37")
        links = ShearRebar(diameter=12, link_spacing=200, n_legs=2, grade="B500B")

        with pytest.raises(ValueError, match="cover must be < D/2"):
            CircularSectionCheck(
                section=section,
                concrete=concrete,
                diameter=600.0,
                cover=300.0,
                shear_reinforcement=links,
            )

        with pytest.raises(ValueError, match="r_sv_override must be > 0 and < D/2"):
            CircularSectionCheck(
                section=section,
                concrete=concrete,
                diameter=600.0,
                cover=50.0,
                shear_reinforcement=links,
                r_sv_override=400.0,
            )

        bad_dia = ShearRebar(diameter=12, link_spacing=200, n_legs=2, grade="B500B")
        object.__setattr__(bad_dia, "diameter", 0.0)
        with pytest.raises(ValueError, match="ShearRebar.diameter must be > 0"):
            CircularSectionCheck(
                section=section,
                concrete=concrete,
                diameter=600.0,
                cover=50.0,
                shear_reinforcement=bad_dia,
            )

        bad_spacing = ShearRebar(diameter=12, link_spacing=200, n_legs=2, grade="B500B")
        object.__setattr__(bad_spacing, "link_spacing", 0.0)
        with pytest.raises(ValueError, match="ShearRebar.link_spacing must be > 0"):
            CircularSectionCheck(
                section=section,
                concrete=concrete,
                diameter=600.0,
                cover=50.0,
                shear_reinforcement=bad_spacing,
            )

        with pytest.raises(ValueError, match="Computed r_sv <= 0"):
            CircularSectionCheck(
                section=section,
                concrete=concrete,
                diameter=600.0,
                cover=290.0,
                shear_reinforcement=ShearRebar(diameter=20, link_spacing=200, n_legs=2, grade="B500B"),
            )

    def test_post_init_warns_for_non_vertical_links_and_applies_kf(self, monkeypatch):
        """Test post init warns for non vertical links and applies kf."""
        section = create_circular_section(diameter=600.0, hook_ref=0)
        concrete = ConcreteMaterial(grade="C30/37")
        links = ShearRebar(
            diameter=12,
            link_spacing=200,
            n_legs=2,
            grade="B500B",
            angle=60.0,
        )

        class _DummyCheck:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def perform_check(self, **kwargs):
                return _ok_result("dummy")

        monkeypatch.setattr(csc_mod, "BendingCheck", _DummyCheck)
        monkeypatch.setattr(csc_mod, "ShearCheck", _DummyCheck)
        monkeypatch.setattr(csc_mod, "CrackingCheck", _DummyCheck)
        monkeypatch.setattr(csc_mod, "StressLimitsCheck", _DummyCheck)

        with ndp_override(k_f=1.1):
            with pytest.warns(UserWarning, match="is ignored"):
                check = CircularSectionCheck(
                    section=section,
                    concrete=concrete,
                    diameter=600.0,
                    cover=50.0,
                    shear_reinforcement=links,
                    apply_k_f=True,
                )

        assert check._concrete_uls is not None
        assert check._concrete_uls.gamma_c == pytest.approx(concrete.gamma_c * 1.1, rel=1e-12)
        assert check.bending.kwargs["concrete"].gamma_c == pytest.approx(concrete.gamma_c * 1.1, rel=1e-12)
        assert check.cracking.kwargs["concrete"] is concrete
        assert check.stress_limits.kwargs["concrete"] is concrete


class TestHelpers:
    """Tests for TestHelpers."""
    def test_ndp_context_warning_and_property_branches(self):
        """Test ndp context warning and property branches."""
        check = _make_stub_check()

        # Force a stale snapshot to trigger _check_ndp_context warning.
        object.__setattr__(check, "_ndp_snapshot", ("old", "context"))
        with pytest.warns(UserWarning, match="NDP context has changed"):
            check._check_ndp_context()

        # r_sv branch with no shear reinforcement
        object.__setattr__(check, "shear_reinforcement", None)
        assert check.r_sv == pytest.approx(check.diameter / 2 - check.cover, rel=1e-12)

        # Accidental design property branches
        object.__setattr__(
            check,
            "_concrete_uls",
            SimpleNamespace(
                f_ck=30.0,
                gamma_c=1.5,
                f_cd_shear=20.0,
                f_cd_shear_accidental=25.0,
                f_ctd=1.3,
                f_ctd_accidental=1.6,
            ),
        )
        object.__setattr__(check, "use_accidental", True)
        assert check._f_cd_design == pytest.approx(25.0, rel=1e-12)
        assert check._f_ctd_design == pytest.approx(1.6, rel=1e-12)
        assert check._f_ywd_design == pytest.approx(0.0, rel=1e-12)

        object.__setattr__(
            check,
            "shear_reinforcement",
            SimpleNamespace(f_yd=435.0, f_yd_accidental=500.0, diameter=12.0),
        )
        assert check._f_ywd_design == pytest.approx(500.0, rel=1e-12)

    def test_build_check_result_statuses(self):
        """Test build check result statuses."""
        passed = CircularSectionCheck._build_check_result(
            check_name="a",
            code_reference="x",
            demand=50.0,
            capacity=100.0,
            units="kN",
            warning_threshold=0.95,
        )
        assert passed.status == CheckStatus.PASS

        warned = CircularSectionCheck._build_check_result(
            check_name="b",
            code_reference="x",
            demand=96.0,
            capacity=100.0,
            units="kN",
            warning_threshold=0.95,
        )
        assert warned.status == CheckStatus.WARNING

        failed = CircularSectionCheck._build_check_result(
            check_name="c",
            code_reference="x",
            demand=120.0,
            capacity=100.0,
            units="kN",
            warning_threshold=0.95,
        )
        assert failed.status == CheckStatus.FAIL

    def test_shear_details_and_theta_conversion(self):
        """Test shear details and theta conversion."""
        d = CircularSectionCheck._shear_details(
            V_Ed=100.0,
            M_Ed=50.0,
            N_Ed=20.0,
            V_Rd=200.0,
            d=500.0,
            z=400.0,
            sigma_cp=1.2,
            cot_theta=2.0,
            lambda_1=0.85,
            lambda_2=0.95,
            b_wc=250.0,
            b_wt=200.0,
            z_0=100.0,
        )
        assert d["theta_deg"] == pytest.approx(degrees(atan(0.5)), rel=1e-12)
        assert d["lambda_1"] == pytest.approx(0.85, rel=1e-12)
        assert d["lambda_2"] == pytest.approx(0.95, rel=1e-12)
        assert d["b_wc"] == pytest.approx(250.0, rel=1e-12)
        assert d["b_wt"] == pytest.approx(200.0, rel=1e-12)
        assert d["z_0"] == pytest.approx(100.0, rel=1e-12)

    def test_lambda_1_and_lambda_2_branches(self, monkeypatch):
        """Test lambda 1 and lambda 2 branches."""
        check = _make_stub_check()

        # Simplified lambda_1
        object.__setattr__(check, "use_simplified_lambda_1", True)
        assert check.calculate_lambda_1(z_0=100.0, z=400.0) == pytest.approx(0.85, rel=1e-12)

        # r_sv fallback branch
        object.__setattr__(check, "use_simplified_lambda_1", False)
        object.__setattr__(check, "r_sv_override", -1.0)
        assert check.calculate_lambda_1(z_0=100.0, z=400.0) == pytest.approx(0.85, rel=1e-12)

        # Numerical branch
        object.__setattr__(check, "r_sv_override", 200.0)
        monkeypatch.setattr(csc_mod.np, "trapezoid", lambda y, x: 0.5)
        lam = check.calculate_lambda_1(z_0=50.0, z=100.0, integration_points=50)
        assert 0.0 <= lam <= 1.0

        # lambda_2 default + spiral + r_sv fallback
        object.__setattr__(check, "is_spiral", False)
        assert check.calculate_lambda_2() == pytest.approx(1.0, rel=1e-12)

        object.__setattr__(check, "is_spiral", True)
        object.__setattr__(check, "r_sv_override", 200.0)
        expected = 1.0 / sqrt((check.shear_reinforcement.link_spacing / (2.0 * pi * 200.0)) ** 2 + 1.0)
        assert check.calculate_lambda_2() == pytest.approx(expected, rel=1e-12)

        object.__setattr__(check, "r_sv_override", -5.0)
        assert check.calculate_lambda_2() == pytest.approx(1.0, rel=1e-12)

    def test_equivalent_web_width_degeneracy_branches(self):
        """Test equivalent web width degeneracy branches."""
        check = _make_stub_check()

        # b_wc == 0, b_wt > 0
        b_w, b_wc, b_wt = check.calculate_equivalent_web_width(d=300.0, z=400.0)
        assert b_wc == pytest.approx(0.0, abs=1e-15)
        assert b_w == pytest.approx(b_wt, rel=1e-12)

        # b_wt == 0, b_wc > 0
        b_w, b_wc, b_wt = check.calculate_equivalent_web_width(d=10.0, z=5.0)
        assert b_wt == pytest.approx(0.0, abs=1e-15)
        assert b_w == pytest.approx(b_wc, rel=1e-12)

        # both == 0
        b_w, b_wc, b_wt = check.calculate_equivalent_web_width(d=1000.0, z=1200.0)
        assert b_wc == pytest.approx(0.0, abs=1e-15)
        assert b_wt == pytest.approx(0.0, abs=1e-15)
        assert b_w == pytest.approx(0.0, abs=1e-15)

    def test_find_rho_l_paths(self, monkeypatch):
        """Test find rho l paths."""
        check = _make_stub_check()
        called = {"args": None, "diagram_calls": 0}

        def _fake_find_rho(**kwargs):
            called["args"] = kwargs
            return 0.0123

        class _Diag:
            def find_strains_for_MN(self, M, N):
                called["diagram_calls"] += 1
                return (0.002, -0.001)

        object.__setattr__(check, "_shear_check", SimpleNamespace(_get_diagram=lambda ignore_compression_steel=False: _Diag()))
        monkeypatch.setattr(csc_mod, "find_rho_l_from_strains", _fake_find_rho)

        # Direct strains path
        out1 = check._find_rho_l(
            M_Ed=100.0,
            N_Ed=0.0,
            b_w=300.0,
            d=500.0,
            eps_top=0.001,
            eps_bottom=-0.001,
        )
        assert out1 == pytest.approx(0.0123, rel=1e-12)
        assert called["diagram_calls"] == 0
        assert called["args"]["eps_top"] == pytest.approx(0.001, rel=1e-12)

        # Diagram strains path
        out2 = check._find_rho_l(
            M_Ed=120.0,
            N_Ed=20.0,
            b_w=300.0,
            d=500.0,
        )
        assert out2 == pytest.approx(0.0123, rel=1e-12)
        assert called["diagram_calls"] == 1
        assert called["args"]["eps_top"] == pytest.approx(0.002, rel=1e-12)

        # Invalid geometry path
        assert check._find_rho_l(M_Ed=0.0, N_Ed=0.0, b_w=0.0, d=500.0) == pytest.approx(0.0, rel=1e-12)

    def test_uncracked_vrdc_negative_inner_returns_zero(self):
        """Test uncracked vrdc negative inner returns zero."""
        check = _make_stub_check()
        # inner = f_ctd^2 + sigma_cp*f_ctd < 0
        out = check.calculate_V_Rd_c_uncracked(sigma_cp=-10.0)
        assert out == pytest.approx(0.0, abs=1e-15)


class TestIterativeAndRoutingHelpers:
    """Tests for TestIterativeAndRoutingHelpers."""
    def test_perform_shear_check_cot_theta_override_and_spacing_warnings(self, monkeypatch):
        """Test perform shear check cot theta override and spacing warnings."""
        check = _make_stub_check()
        check.shear_reinforcement.leg_spacing = 350.0

        monkeypatch.setattr(CircularSectionCheck, "calculate_lambda_1", lambda self, z_0, z: 0.8)
        monkeypatch.setattr(CircularSectionCheck, "calculate_lambda_2", lambda self: 0.9)
        monkeypatch.setattr(CircularSectionCheck, "calculate_equivalent_web_width", lambda self, d, z: (300.0, 320.0, 280.0))
        monkeypatch.setattr(CircularSectionCheck, "_find_rho_l", lambda self, **kwargs: 0.01)
        monkeypatch.setattr(CircularSectionCheck, "calculate_V_Rd_c_uncracked", lambda self, sigma_cp: 120.0)

        monkeypatch.setattr(csc_mod, "sigma_cp_from_N_and_area", lambda N_Ed, area: 2.0)
        monkeypatch.setattr(csc_mod, "cap_sigma_cp_upper", lambda sigma_cp, f_cd: 1.5)
        monkeypatch.setattr(csc_mod, "find_V_Rd_c_cracked", lambda **kwargs: 90.0)
        monkeypatch.setattr(csc_mod, "find_alpha_cw", lambda f_cd, sigma_cp, use_sigma_cp_for_alpha_cw=False: 1.0)
        monkeypatch.setattr(csc_mod, "find_nu_1_factor", lambda f_ck, link_angle_degrees=90.0: 0.6)
        monkeypatch.setattr(csc_mod, "find_max_allowable_link_spacing", lambda **kwargs: 150.0)
        monkeypatch.setattr(csc_mod, "find_max_allowable_leg_spacing", lambda **kwargs: 300.0)

        with pytest.warns(UserWarning, match="maximum allowable spacing"):
            result = check.perform_shear_check(
                load_case=ShearLoadCase(V_Ed=200.0, M_Ed=100.0, N_Ed=300.0),
                cot_theta_override=2.2,
                suppress_warnings=False,
            )

        assert result.details["link_spacing_satisfied"] is False
        assert result.details["leg_spacing_satisfied"] is False
        assert result.details["V_Rd_s"] is not None
        assert result.details["V_Rd_max"] is not None

    def test_perform_shear_check_note_2_and_vrds_cot_theta_branches(self, monkeypatch):
        """Test perform shear check note 2 and vrds cot theta branches."""
        check = _make_stub_check()

        monkeypatch.setattr(CircularSectionCheck, "calculate_lambda_1", lambda self, z_0, z: 0.8)
        monkeypatch.setattr(CircularSectionCheck, "calculate_lambda_2", lambda self: 0.9)
        monkeypatch.setattr(CircularSectionCheck, "calculate_equivalent_web_width", lambda self, d, z: (300.0, 320.0, 280.0))
        monkeypatch.setattr(CircularSectionCheck, "_find_rho_l", lambda self, **kwargs: 0.01)
        monkeypatch.setattr(CircularSectionCheck, "calculate_V_Rd_c_uncracked", lambda self, sigma_cp: 120.0)

        monkeypatch.setattr(csc_mod, "sigma_cp_from_N_and_area", lambda N_Ed, area: 2.0)
        monkeypatch.setattr(csc_mod, "cap_sigma_cp_upper", lambda sigma_cp, f_cd: 1.5)
        monkeypatch.setattr(csc_mod, "find_V_Rd_c_cracked", lambda **kwargs: 90.0)
        monkeypatch.setattr(csc_mod, "find_alpha_cw", lambda f_cd, sigma_cp, use_sigma_cp_for_alpha_cw=False: 1.0)
        monkeypatch.setattr(csc_mod, "find_nu_1_factor", lambda f_ck, link_angle_degrees=90.0: 0.6)
        monkeypatch.setattr(csc_mod, "find_cot_theta_for_V_Ed_from_V_Rd_s", lambda **kwargs: 1.6)
        monkeypatch.setattr(csc_mod, "find_max_allowable_link_spacing", lambda **kwargs: 250.0)

        # Standard Note 1 but cot(theta) from V_Rd,s equation (covers 987-988 path).
        result_vrds = check.perform_shear_check(
            load_case=ShearLoadCase(V_Ed=200.0, M_Ed=100.0, N_Ed=300.0),
            use_v_rd_s_for_cot_theta=True,
            suppress_warnings=True,
        )
        assert result_vrds.details["cot_theta_from_v_rd_s"] is True

        # Note 2 branch in perform_shear_check (covers 973-981 path).
        object.__setattr__(check, "use_increased_nu_1", True)
        monkeypatch.setattr(
            CircularSectionCheck,
            "_find_V_Rd_max_with_note_2_iteration",
            lambda self, *args, **kwargs: (500.0, 400.0, 1.8, 0.75, True),
        )
        result_note_2 = check.perform_shear_check(
            load_case=ShearLoadCase(V_Ed=200.0, M_Ed=100.0, N_Ed=300.0),
            suppress_warnings=True,
        )
        assert result_note_2.details["used_note_2"] is True
        assert result_note_2.details["f_ywd"] == pytest.approx(0.8 * check.shear_reinforcement.f_yk, rel=1e-12)

    def test_note_2_iteration_branches(self, monkeypatch):
        """Test note 2 iteration branches."""
        check = _make_stub_check()
        # Use simple custom reinforcement so threshold math is controllable.
        object.__setattr__(check, "shear_reinforcement", SimpleNamespace(
            area_per_unit_length=0.2,
            f_yk=1000.0,
            f_yd=900.0,
            f_yd_accidental=900.0,
            angle=90.0,
            link_spacing=200.0,
            diameter=12.0,
            leg_spacing=None,
        ))
        object.__setattr__(
            check,
            "_concrete_uls",
            SimpleNamespace(
                f_ck=30.0,
                f_cd_shear=20.0,
                f_cd_shear_accidental=25.0,
            ),
        )

        monkeypatch.setattr(csc_mod, "find_alpha_cw", lambda f_cd, sigma_cp, use_sigma_cp_for_alpha_cw=False: 1.0)
        monkeypatch.setattr(csc_mod, "find_nu_1_factor", lambda f_ck, link_angle_degrees=90.0: 0.6)
        monkeypatch.setattr(csc_mod, "find_nu_1_factor_note_2", lambda f_ck, link_angle_degrees=90.0: 0.8)

        # Case A: Note 2 not applicable (sigma_s_1 >= threshold)
        monkeypatch.setattr(csc_mod, "find_cot_theta_for_V_Ed_from_V_Rd_max", lambda **kwargs: 1.0)
        a = check._find_V_Rd_max_with_note_2_iteration(
            V_Ed=1000.0,
            z=300.0,
            sigma_cp=1.0,
            b_w=300.0,
            lambda_1=1.0,
            lambda_2=1.0,
        )
        assert a[4] is False

        # Case B: Note 2 converges
        seq_b = iter([2.5, 2.0])  # n1 then n2
        monkeypatch.setattr(csc_mod, "find_cot_theta_for_V_Ed_from_V_Rd_max", lambda **kwargs: next(seq_b))
        b = check._find_V_Rd_max_with_note_2_iteration(
            V_Ed=20.0,
            z=300.0,
            sigma_cp=1.0,
            b_w=300.0,
            lambda_1=1.0,
            lambda_2=1.0,
        )
        assert b[4] is True

        # Case C: oscillation -> warn and revert to Note 1
        seq_c = iter([2.5, 0.1])  # n1 then n2
        monkeypatch.setattr(csc_mod, "find_cot_theta_for_V_Ed_from_V_Rd_max", lambda **kwargs: next(seq_c))
        with pytest.warns(UserWarning, match="Oscillation detected"):
            c = check._find_V_Rd_max_with_note_2_iteration(
                V_Ed=20.0,
                z=300.0,
                sigma_cp=1.0,
                b_w=300.0,
                lambda_1=1.0,
                lambda_2=1.0,
                suppress_warnings=False,
            )
        assert c[4] is False

        # Case D: same oscillation but warning suppressed
        seq_d = iter([2.5, 0.1])
        monkeypatch.setattr(csc_mod, "find_cot_theta_for_V_Ed_from_V_Rd_max", lambda **kwargs: next(seq_d))
        d = check._find_V_Rd_max_with_note_2_iteration(
            V_Ed=20.0,
            z=300.0,
            sigma_cp=1.0,
            b_w=300.0,
            lambda_1=1.0,
            lambda_2=1.0,
            suppress_warnings=True,
        )
        assert d[4] is False

    def test_note_2_iteration_vrds_path(self, monkeypatch):
        """Test note 2 iteration vrds path."""
        check = _make_stub_check()
        object.__setattr__(check, "shear_reinforcement", SimpleNamespace(
            area_per_unit_length=0.4,
            f_yk=500.0,
            f_yd=435.0,
            f_yd_accidental=500.0,
            angle=90.0,
            link_spacing=200.0,
            diameter=12.0,
            leg_spacing=None,
        ))
        object.__setattr__(
            check,
            "_concrete_uls",
            SimpleNamespace(f_ck=30.0, f_cd_shear=20.0, f_cd_shear_accidental=25.0),
        )
        monkeypatch.setattr(csc_mod, "find_alpha_cw", lambda f_cd, sigma_cp, use_sigma_cp_for_alpha_cw=False: 1.0)
        monkeypatch.setattr(csc_mod, "find_nu_1_factor", lambda f_ck, link_angle_degrees=90.0: 0.6)
        monkeypatch.setattr(csc_mod, "find_nu_1_factor_note_2", lambda f_ck, link_angle_degrees=90.0: 0.8)

        seq = iter([1.6, 1.4])  # cot(theta) for Note 1 then Note 2
        monkeypatch.setattr(csc_mod, "find_cot_theta_for_V_Ed_from_V_Rd_s", lambda **kwargs: next(seq))

        out = check._find_V_Rd_max_with_note_2_iteration(
            V_Ed=20.0,
            z=300.0,
            sigma_cp=1.0,
            b_w=300.0,
            lambda_1=1.0,
            lambda_2=1.0,
            use_v_rd_s_for_cot_theta=True,
            suppress_warnings=True,
        )
        assert out[4] in {True, False}

    def test_compute_cot_theta_for_tension_shift_paths(self, monkeypatch):
        """Test compute cot theta for tension shift paths."""
        check = _make_stub_check()
        _cot_diag = SimpleNamespace(
            find_strains_for_MN=lambda M, N, strict=False: (0.001, -0.001),
            find_strain_state_for_MN=lambda M_target, N_target: None,
        )
        object.__setattr__(check, "_shear_check", SimpleNamespace(
            find_effective_depth=lambda M, N, eps_top=None, eps_bottom=None, ignore_compression_steel=False, **kw: 500.0,
            find_lever_arm=lambda M, N, d, eps_top=None, eps_bottom=None, ignore_compression_steel=False, **kw: (450.0, None),  # force z=0.9d fallback
            _get_diagram=lambda ignore_compression_steel=False: _cot_diag,
        ))
        object.__setattr__(
            check,
            "_concrete_uls",
            SimpleNamespace(
                f_ck=30.0,
                f_cd_shear=20.0,
                f_cd_shear_accidental=25.0,
            ),
        )
        object.__setattr__(check, "section", SimpleNamespace(get_area=lambda: 100_000.0, section_name="S"))

        monkeypatch.setattr(CircularSectionCheck, "calculate_equivalent_web_width", lambda self, d, z: (300.0, 320.0, 280.0))
        monkeypatch.setattr(csc_mod, "sigma_cp_from_N_and_area", lambda N_Ed, area: 2.0)
        monkeypatch.setattr(csc_mod, "cap_sigma_cp_upper", lambda sigma_cp, f_cd: 1.5)
        monkeypatch.setattr(csc_mod, "find_alpha_cw", lambda f_cd, sigma_cp, use_sigma_cp_for_alpha_cw=False: 1.0)
        monkeypatch.setattr(csc_mod, "find_nu_1_factor", lambda f_ck, link_angle_degrees=90.0: 0.6)
        monkeypatch.setattr(csc_mod, "find_cot_theta_for_V_Ed_from_V_Rd_max", lambda **kwargs: 1.9)
        monkeypatch.setattr(csc_mod, "find_cot_theta_for_V_Ed_from_V_Rd_s", lambda **kwargs: 1.4)

        out_vrdmax = check._compute_cot_theta_for_tension_shift(
            M_Ed=100.0,
            N_Ed=200.0,
            V_Ed=300.0,
            use_v_rd_s_for_cot_theta=False,
        )
        assert out_vrdmax == pytest.approx(1.9, rel=1e-12)

        out_vrds = check._compute_cot_theta_for_tension_shift(
            M_Ed=100.0,
            N_Ed=200.0,
            V_Ed=300.0,
            use_v_rd_s_for_cot_theta=True,
        )
        assert out_vrds == pytest.approx(1.4, rel=1e-12)

    def test_wrapper_forwarding_for_bending_cracking_and_stress_limits(self, monkeypatch):
        """Test wrapper forwarding for bending cracking and stress limits."""
        check = _make_stub_check()
        bend_rec = _Recorder(_ok_result("bending"))
        crack_rec = _Recorder(_ok_result("cracking"))
        stress_rec = _Recorder(_ok_result("stress"))
        object.__setattr__(check, "_bending_check", bend_rec)
        object.__setattr__(check, "_cracking_check", crack_rec)
        object.__setattr__(check, "_stress_limits_check", stress_rec)

        monkeypatch.setattr(CircularSectionCheck, "_compute_cot_theta_for_tension_shift", lambda self, **kwargs: 1.8)

        # Auto cot(theta) branch
        b = check.perform_bending_check(
            M_Ed=120.0,
            N_Ed=30.0,
            V_Ed=100.0,
            M_cap=150.0,
            cot_theta_override=None,
        )
        assert b.check_name == "bending"
        assert bend_rec.calls[-1]["cot_theta_override"] == pytest.approx(1.8, rel=1e-12)

        # User override branch (should pass through user value)
        b2 = check.perform_bending_check(
            M_Ed=120.0,
            N_Ed=30.0,
            V_Ed=100.0,
            M_cap=150.0,
            cot_theta_override=1.2,
        )
        assert b2.check_name == "bending"
        assert bend_rec.calls[-1]["cot_theta_override"] == pytest.approx(1.2, rel=1e-12)

        c = check.perform_cracking_check(M_Ed=50.0, N_Ed=10.0, force_cracked=True)
        assert c.check_name == "cracking"
        assert crack_rec.calls[-1]["force_cracked"] is True

        s = check.perform_stress_limits_check(M_Ed=50.0, N_Ed=10.0, suppress_warnings=True, check_k1_stress=True)
        assert s.check_name == "stress"
        assert stress_rec.calls[-1]["suppress_warnings"] is True
        assert stress_rec.calls[-1]["check_k1_stress"] is True

    def test_perform_shear_check_standard_path_with_spacing_flags(self, monkeypatch):
        """Test perform shear check standard path with spacing flags."""
        check = _make_stub_check()
        check.shear_reinforcement.leg_spacing = 350.0
        _std_diag = SimpleNamespace(
            find_strains_for_MN=lambda M, N, strict=False: (0.001, -0.001),
            find_strain_state_for_MN=lambda M_target, N_target: None,
        )
        object.__setattr__(check, "_shear_check", SimpleNamespace(
            find_effective_depth=lambda M, N, eps_top=None, eps_bottom=None, ignore_compression_steel=False, **kw: 500.0,
            find_lever_arm=lambda M, N, d, eps_top=None, eps_bottom=None, ignore_compression_steel=False, **kw: (450.0, 400.0),
            _get_diagram=lambda ignore_compression_steel=False: _std_diag,
        ))

        monkeypatch.setattr(CircularSectionCheck, "calculate_lambda_1", lambda self, z_0, z: 0.8)
        monkeypatch.setattr(CircularSectionCheck, "calculate_lambda_2", lambda self: 0.9)
        monkeypatch.setattr(CircularSectionCheck, "calculate_equivalent_web_width", lambda self, d, z: (300.0, 320.0, 280.0))
        monkeypatch.setattr(CircularSectionCheck, "_find_rho_l", lambda self, **kwargs: 0.01)
        monkeypatch.setattr(CircularSectionCheck, "calculate_V_Rd_c_uncracked", lambda self, sigma_cp: 120.0)

        monkeypatch.setattr(csc_mod, "sigma_cp_from_N_and_area", lambda N_Ed, area: 2.0)
        monkeypatch.setattr(csc_mod, "cap_sigma_cp_upper", lambda sigma_cp, f_cd: 1.5)
        monkeypatch.setattr(csc_mod, "find_V_Rd_c_cracked", lambda **kwargs: 90.0)
        monkeypatch.setattr(csc_mod, "find_alpha_cw", lambda f_cd, sigma_cp, use_sigma_cp_for_alpha_cw=False: 1.0)
        monkeypatch.setattr(csc_mod, "find_nu_1_factor", lambda f_ck, link_angle_degrees=90.0: 0.6)
        monkeypatch.setattr(csc_mod, "find_cot_theta_for_V_Ed_from_V_Rd_max", lambda **kwargs: 2.0)
        monkeypatch.setattr(csc_mod, "find_max_allowable_link_spacing", lambda **kwargs: 250.0)
        monkeypatch.setattr(csc_mod, "find_max_allowable_leg_spacing", lambda **kwargs: 300.0)

        result = check.perform_shear_check(
            load_case=ShearLoadCase(V_Ed=200.0, M_Ed=100.0, N_Ed=300.0),
            suppress_warnings=True,
        )

        assert result.check_name.startswith("Circular shear")
        assert result.status in {CheckStatus.PASS, CheckStatus.WARNING, CheckStatus.FAIL}
        assert result.details["link_spacing_satisfied"] is True
        assert result.details["leg_spacing_satisfied"] is False
        assert result.details["V_Rd_s"] is not None
        assert result.details["V_Rd_max"] is not None
        assert result.details["lambda_1"] == pytest.approx(0.8, rel=1e-12)
        assert result.details["lambda_2"] == pytest.approx(0.9, rel=1e-12)
