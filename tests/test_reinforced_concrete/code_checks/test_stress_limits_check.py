"""
Tests for StressLimitsCheck and its pure stress-limit helpers.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from materials.reinforced_concrete.code_checks.base_check import CheckStatus
from materials.reinforced_concrete.code_checks.ec2_2004.stress_limits_check import (
    StressLimitResult,
    StressLimitsCheck,
    check_characteristic_concrete_stress,
    check_characteristic_reinforcement_stress,
    check_imposed_deformation_stress,
    check_quasi_permanent_concrete_stress,
    check_reinforcement_yielding,
    compute_nonlinear_creep_coefficient,
)
from materials.reinforced_concrete.geometry import (
    create_linear_rebar_layer,
    create_rectangular_section,
)
from materials.reinforced_concrete.materials import ConcreteMaterial, Rebar
from materials.reinforced_concrete.ndp import get_ndp


def _make_section(
    *,
    bottom_E_s: float = 200000.0,
    top_E_s: float = 200000.0,
    include_top: bool = True,
) -> object:
    section = create_rectangular_section(width=300.0, height=500.0)

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
            rebar=Rebar(diameter=20, grade="B500B", E_s=top_E_s),
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
        eps_top: float = 0.001,
        eps_bottom: float = -0.001,
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

    def find_strains_for_MN(self, M_Ed: float, N_Ed: float, strict: bool = True):
        if self.raise_on_find:
            raise ValueError("outside diagram")
        return self.eps_top, self.eps_bottom

    def get_fibre_forces_from_end_strains(self, eps_top: float, eps_bottom: float):
        y = np.zeros_like(self._forces, dtype=float)
        return self._forces, y, self._areas


class TestStressLimitPureFunctions:
    """Tests for TestStressLimitPureFunctions."""
    def test_check_characteristic_concrete_stress_threshold_and_message(self):
        """Test check characteristic concrete stress threshold and message."""
        f_ck = 30.0
        k_1 = float(get_ndp("k_1_stress"))
        at_limit = k_1 * f_ck
        over_limit = at_limit + 0.1

        exceeded_at, msg_at = check_characteristic_concrete_stress(at_limit, f_ck)
        exceeded_over, msg_over = check_characteristic_concrete_stress(over_limit, f_ck)

        assert exceeded_at is False
        assert msg_at == ""
        assert exceeded_over is True
        assert "EC2" in msg_over

    def test_check_quasi_permanent_concrete_stress_threshold_and_message(self):
        """Test check quasi permanent concrete stress threshold and message."""
        f_ck = 30.0
        k_2 = float(get_ndp("k_2_stress"))
        at_limit = k_2 * f_ck
        over_limit = at_limit + 0.1

        exceeded_at, msg_at = check_quasi_permanent_concrete_stress(at_limit, f_ck)
        exceeded_over, msg_over = check_quasi_permanent_concrete_stress(over_limit, f_ck)

        assert exceeded_at is False
        assert msg_at == ""
        assert exceeded_over is True
        assert "Non-linear creep" in msg_over

    def test_check_reinforcement_limits_and_yielding(self):
        """Test check reinforcement limits and yielding."""
        f_yk = 500.0
        k_3 = float(get_ndp("k_3_stress"))
        k_4 = float(get_ndp("k_4_stress"))

        sig_k3 = k_3 * f_yk + 0.5
        sig_k4 = k_4 * f_yk + 0.5
        sig_yield = f_yk + 0.5

        k3_exceeded, k3_msg = check_characteristic_reinforcement_stress(sig_k3, f_yk)
        k4_exceeded, k4_msg = check_imposed_deformation_stress(sig_k4, f_yk)
        yielded, y_msg = check_reinforcement_yielding(sig_yield, f_yk)

        assert k3_exceeded is True
        assert "Reinforcement stress limit exceeded" in k3_msg
        assert k4_exceeded is True
        assert "Imposed deformation stress limit exceeded" in k4_msg
        assert yielded is True
        assert "has yielded" in y_msg

    def test_compute_nonlinear_creep_coefficient_and_invalid_fck(self):
        """Test compute nonlinear creep coefficient and invalid fck."""
        phi_nl = compute_nonlinear_creep_coefficient(
            sigma_c=18.0,
            f_ck=30.0,
            creep_coefficient=1.5,
        )
        assert phi_nl > 0.0

        with pytest.raises(ValueError, match="f_ck cannot be equal to or less than zero"):
            compute_nonlinear_creep_coefficient(sigma_c=10.0, f_ck=0.0, creep_coefficient=1.5)


class TestStressLimitsHelpersAndCaching:
    """Tests for TestStressLimitsHelpersAndCaching."""
    def test_E_cm_eff_and_ndp_properties(self, concrete_c30):
        """Test E cm eff and ndp properties."""
        check = StressLimitsCheck(
            section=_make_section(),
            concrete=concrete_c30,
        )

        expected = concrete_c30.get_elastic_modulus() / (1.0 + check.creep_coefficient)
        assert check.E_cm_eff == pytest.approx(expected, rel=1e-12)
        assert check.k_1_stress == pytest.approx(float(get_ndp("k_1_stress")), rel=1e-12)
        assert check.k_2_stress == pytest.approx(float(get_ndp("k_2_stress")), rel=1e-12)
        assert check.k_3_stress == pytest.approx(float(get_ndp("k_3_stress")), rel=1e-12)
        assert check.k_4_stress == pytest.approx(float(get_ndp("k_4_stress")), rel=1e-12)

    def test_get_diagram_cache_reuse_and_invalidation(self, monkeypatch, concrete_c30):
        """Test get diagram cache reuse and invalidation."""
        created: list[object] = []

        def _fake_factory(**kwargs):
            obj = object()
            created.append(obj)
            return obj

        monkeypatch.setattr(
            "materials.reinforced_concrete.code_checks.ec2_2004.stress_limits_check.create_interaction_diagram",
            _fake_factory,
        )

        check = StressLimitsCheck(section=_make_section(), concrete=concrete_c30)
        d1 = check._get_diagram(ignore_compression_steel=False)
        d2 = check._get_diagram(ignore_compression_steel=False)
        assert d1 is d2
        assert len(created) == 1

        d3 = check._get_diagram(ignore_compression_steel=True)
        d4 = check._get_diagram(ignore_compression_steel=True)
        assert d3 is d4
        assert len(created) == 2

        check.creep_coefficient = 2.0
        d5 = check._get_diagram(ignore_compression_steel=False)
        assert d5 is not d1
        assert len(created) == 3

    def test_peak_concrete_stress_handles_zero_area_and_compression_sign(self, concrete_c30):
        """Test peak concrete stress handles zero area and compression sign."""
        check = StressLimitsCheck(section=_make_section(), concrete=concrete_c30)

        diag_zero = _FakeDiagram(
            fibre_mat=np.array(["concrete", "concrete"]),
            forces=np.array([10.0, 20.0]),
            areas=np.array([0.0, 0.0]),
        )
        assert check._get_peak_concrete_stress(0.001, -0.001, diag_zero) == 0.0

        diag_mixed = _FakeDiagram(
            fibre_mat=np.array(["concrete", "concrete", "steel"]),
            forces=np.array([30.0, -5.0, 100.0]),
            areas=np.array([2.0, 1.0, 1.0]),
        )
        peak = check._get_peak_concrete_stress(0.001, -0.001, diag_mixed)
        assert peak == pytest.approx(15.0, rel=1e-12)

    def test_get_max_steel_stress_tension_only(self, concrete_c30):
        """Test get max steel stress tension only."""
        check = StressLimitsCheck(section=_make_section(), concrete=concrete_c30)

        no_tension = check._get_max_steel_stress(eps_top=0.001, eps_bottom=0.0005)
        with_tension = check._get_max_steel_stress(eps_top=0.001, eps_bottom=-0.002)

        assert no_tension == pytest.approx(0.0, abs=1e-12)
        assert with_tension > 0.0

    def test_get_f_yk_max_with_and_without_rebars(self, concrete_c30):
        """Test get f yk max with and without rebars."""
        section_with = _make_section()
        check_with = StressLimitsCheck(section=section_with, concrete=concrete_c30)
        assert check_with._get_f_yk_max() == pytest.approx(500.0, abs=1e-12)

        section_empty = create_rectangular_section(width=300.0, height=500.0)
        check_empty = StressLimitsCheck(section=section_empty, concrete=concrete_c30)
        assert check_empty._get_f_yk_max() == pytest.approx(500.0, abs=1e-12)

    def test_build_diagram_with_E_cm_eff_wrapper(self, monkeypatch, concrete_c30):
        """Test build diagram with E cm eff wrapper."""
        captured = {}

        def _fake_factory(**kwargs):
            captured.update(kwargs)
            return "diagram"

        monkeypatch.setattr(
            "materials.reinforced_concrete.code_checks.ec2_2004.stress_limits_check.create_interaction_diagram",
            _fake_factory,
        )

        check = StressLimitsCheck(section=_make_section(), concrete=concrete_c30)
        out = check._build_diagram_with_E_cm_eff(E_cm_eff=12345.0, ignore_compression_steel=True)
        assert out == "diagram"
        assert captured["elastic_modulus"] == pytest.approx(12345.0, rel=1e-12)
        assert captured["ignore_compression_steel"] is True

    def test_peak_concrete_stress_negative_peak_clamps_to_zero(self, concrete_c30):
        """Test peak concrete stress negative peak clamps to zero."""
        check = StressLimitsCheck(section=_make_section(), concrete=concrete_c30)
        diag_negative = _FakeDiagram(
            fibre_mat=np.array(["concrete", "concrete"]),
            forces=np.array([-10.0, -20.0]),
            areas=np.array([1.0, 1.0]),
        )
        assert check._get_peak_concrete_stress(0.001, -0.001, diag_negative) == 0.0

    def test_get_max_steel_stress_zero_height_raises(self, concrete_c30):
        """Test get max steel stress zero height raises."""
        check = StressLimitsCheck(section=_make_section(), concrete=concrete_c30)

        class _Outline:
            bounds = (0.0, 0.0, 300.0, 0.0)

        object.__setattr__(check.section, "outline", _Outline())
        with pytest.raises(ValueError, match="cannot be equal to or less than zero"):
            check._get_max_steel_stress(eps_top=0.001, eps_bottom=-0.001)


class TestStressLimitsDetailedAndPerform:
    """Tests for TestStressLimitsDetailedAndPerform."""
    def test_calculate_detailed_applies_nonlinear_creep_when_k2_exceeded(self, monkeypatch, concrete_c30):
        """Test calculate detailed applies nonlinear creep when k2 exceeded."""
        check = StressLimitsCheck(
            section=_make_section(),
            concrete=concrete_c30,
            check_k1_stress=True,
            check_k2_stress=True,
            check_k3_stress=True,
            check_yielding=True,
            check_k4_stress=True,
            apply_nonlinear_creep=True,
            iterate_nonlinear_creep=False,
        )

        base_diag = _FakeDiagram(eps_top=0.001, eps_bottom=-0.001)
        nl_diag = _FakeDiagram(eps_top=0.002, eps_bottom=-0.002)

        monkeypatch.setattr(
            StressLimitsCheck,
            "_get_diagram",
            lambda self, ignore_compression_steel=False: base_diag,
        )
        monkeypatch.setattr(
            StressLimitsCheck,
            "_build_diagram_with_E_cm_eff",
            lambda self, E_cm_eff, ignore_compression_steel=False: nl_diag,
        )

        def _fake_peak(self, eps_top, eps_bottom, diagram=None):
            diag = diagram if diagram is not None else base_diag
            if diag is base_diag:
                return self.concrete.f_ck * 0.6
            return self.concrete.f_ck * 0.3

        def _fake_steel(self, eps_top, eps_bottom):
            return 350.0 if abs(eps_top - 0.001) < 1e-12 else 250.0

        monkeypatch.setattr(StressLimitsCheck, "_get_peak_concrete_stress", _fake_peak)
        monkeypatch.setattr(StressLimitsCheck, "_get_max_steel_stress", _fake_steel)

        r = check.calculate_detailed(M_Ed=50.0, N_Ed=100.0)

        assert r.k1_exceeded is False
        assert r.k2_exceeded is True
        assert r.k3_exceeded is False
        assert r.yielding is False
        assert r.k4_exceeded is False
        assert r.nonlinear_creep_applied is True
        assert r.creep_coefficient_used > check.creep_coefficient
        assert r.sigma_c_peak == pytest.approx(concrete_c30.f_ck * 0.3, rel=1e-12)
        assert r.sigma_s_max == pytest.approx(250.0, rel=1e-12)
        assert len(r.messages) >= 1

    def test_calculate_detailed_message_appends_for_enabled_exceeded_checks(self, monkeypatch, concrete_c30):
        """Test calculate detailed message appends for enabled exceeded checks."""
        check = StressLimitsCheck(
            section=_make_section(),
            concrete=concrete_c30,
            check_k1_stress=True,
            check_k2_stress=False,
            check_k3_stress=True,
            check_yielding=True,
            check_k4_stress=True,
            apply_nonlinear_creep=False,
        )

        monkeypatch.setattr(
            StressLimitsCheck,
            "_get_diagram",
            lambda self, ignore_compression_steel=False: _FakeDiagram(),
        )
        monkeypatch.setattr(StressLimitsCheck, "_get_peak_concrete_stress", lambda *args, **kwargs: concrete_c30.f_ck)
        monkeypatch.setattr(StressLimitsCheck, "_get_max_steel_stress", lambda *args, **kwargs: 550.0)

        r = check.calculate_detailed(M_Ed=60.0, N_Ed=50.0)
        assert r.k1_exceeded is True
        assert r.k3_exceeded is True
        assert r.yielding is True
        assert r.k4_exceeded is True
        assert len(r.messages) == 4

    def test_calculate_detailed_nonlinear_creep_breaks_on_small_modulus_change(self, monkeypatch, concrete_c30):
        """Test calculate detailed nonlinear creep breaks on small modulus change."""
        check = StressLimitsCheck(
            section=_make_section(),
            concrete=concrete_c30,
            check_k1_stress=False,
            check_k2_stress=True,
            check_k3_stress=False,
            check_yielding=False,
            check_k4_stress=False,
            apply_nonlinear_creep=True,
            iterate_nonlinear_creep=True,
        )

        monkeypatch.setattr(
            StressLimitsCheck,
            "_get_diagram",
            lambda self, ignore_compression_steel=False: _FakeDiagram(),
        )
        monkeypatch.setattr(StressLimitsCheck, "_get_peak_concrete_stress", lambda *args, **kwargs: concrete_c30.f_ck * 0.8)
        monkeypatch.setattr(StressLimitsCheck, "_get_max_steel_stress", lambda *args, **kwargs: 100.0)
        monkeypatch.setattr(
            "materials.reinforced_concrete.code_checks.ec2_2004.stress_limits_check.compute_nonlinear_creep_coefficient",
            lambda sigma_c, f_ck, creep_coefficient: creep_coefficient,
        )
        build_calls = {"n": 0}
        monkeypatch.setattr(
            StressLimitsCheck,
            "_build_diagram_with_E_cm_eff",
            lambda self, E_cm_eff, ignore_compression_steel=False: build_calls.__setitem__("n", build_calls["n"] + 1),
        )

        r = check.calculate_detailed(M_Ed=40.0, N_Ed=20.0)
        assert r.k2_exceeded is True
        assert build_calls["n"] == 0

    def test_perform_check_returns_inf_on_solver_error(self, monkeypatch, concrete_c30):
        """Test perform check returns inf on solver error."""
        check = StressLimitsCheck(section=_make_section(), concrete=concrete_c30)
        monkeypatch.setattr(
            StressLimitsCheck,
            "calculate_detailed",
            lambda self, M_Ed, N_Ed=0.0, ignore_compression_steel=False: (_ for _ in ()).throw(
                ValueError("outside domain")
            ),
        )

        result = check.perform_check(M_Ed=5000.0, N_Ed=0.0)

        assert result.status == CheckStatus.FAIL
        assert result.utilization == float("inf")
        assert "outside domain" in result.details["error"]

    def test_perform_check_governing_utilization_selection(self, monkeypatch, concrete_c30):
        """Test perform check governing utilization selection."""
        check = StressLimitsCheck(
            section=_make_section(),
            concrete=concrete_c30,
            check_k1_stress=True,
            check_k2_stress=True,
            check_k3_stress=True,
            check_yielding=True,
            check_k4_stress=True,
        )

        fake = StressLimitResult(
            sigma_c_peak=18.0,
            sigma_s_max=450.0,
            f_yk=500.0,
            k1_exceeded=False,
            k2_exceeded=True,
            k3_exceeded=True,
            yielding=False,
            k4_exceeded=False,
            nonlinear_creep_applied=False,
            creep_coefficient_used=check.creep_coefficient,
            messages=[],
        )
        monkeypatch.setattr(
            StressLimitsCheck,
            "calculate_detailed",
            lambda self, M_Ed, N_Ed=0.0, ignore_compression_steel=False: fake,
        )

        result = check.perform_check(M_Ed=120.0, N_Ed=250.0, warning_threshold=0.95)

        assert result.details["governing_check"] == "k2_concrete_qp"
        assert result.utilization > 1.0
        assert result.status == CheckStatus.FAIL

    def test_perform_check_no_enabled_checks_returns_none_governing(self, monkeypatch, concrete_c30):
        """Test perform check no enabled checks returns none governing."""
        check = StressLimitsCheck(
            section=_make_section(),
            concrete=concrete_c30,
            check_k1_stress=False,
            check_k2_stress=False,
            check_k3_stress=False,
            check_yielding=False,
            check_k4_stress=False,
        )
        fake = StressLimitResult(
            sigma_c_peak=10.0,
            sigma_s_max=100.0,
            f_yk=500.0,
            messages=[],
        )
        monkeypatch.setattr(
            StressLimitsCheck,
            "calculate_detailed",
            lambda self, M_Ed, N_Ed=0.0, ignore_compression_steel=False: fake,
        )

        result = check.perform_check(M_Ed=10.0, N_Ed=0.0)
        assert result.utilization == pytest.approx(0.0, abs=1e-12)
        assert result.details["governing_check"] == "none"
        assert result.status == CheckStatus.PASS

    def test_perform_check_warning_emission_and_suppression(self, monkeypatch, concrete_c30):
        """Test perform check warning emission and suppression."""
        check = StressLimitsCheck(
            section=_make_section(),
            concrete=concrete_c30,
            check_k1_stress=False,
            check_k2_stress=False,
            check_k3_stress=False,
            check_yielding=False,
            check_k4_stress=False,
        )
        fake = StressLimitResult(
            sigma_c_peak=10.0,
            sigma_s_max=100.0,
            f_yk=500.0,
            messages=["warning A", "warning B"],
        )
        monkeypatch.setattr(
            StressLimitsCheck,
            "calculate_detailed",
            lambda self, M_Ed, N_Ed=0.0, ignore_compression_steel=False: fake,
        )

        with warnings.catch_warnings(record=True) as caught_warn:
            warnings.simplefilter("always")
            check.perform_check(M_Ed=10.0, N_Ed=0.0, suppress_warnings=False)
        assert len(caught_warn) == 2
        messages = [str(w.message) for w in caught_warn]
        assert "warning A" in messages
        assert "warning B" in messages

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            check.perform_check(M_Ed=10.0, N_Ed=0.0, suppress_warnings=True)
        assert len(caught) == 0
