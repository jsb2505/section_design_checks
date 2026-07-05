"""
Unit tests for BeamCheck wrapper and delegation behaviour.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import materials.reinforced_concrete.code_checks.ec2_2004 as ec2_mod
import materials.reinforced_concrete.code_checks.ec2_2004.beam_check as beam_mod
from materials.reinforced_concrete.code_checks.base_check import CheckResult, CheckStatus
from materials.reinforced_concrete.code_checks.ec2_2004 import BeamCheck
from materials.reinforced_concrete.code_checks.ec2_2004.shear_check import ShearLoadCase
from materials.reinforced_concrete.constitutive import ConcreteModelType
from materials.reinforced_concrete.ndp import get_ndp_context


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


class _Recorder:
    def __init__(self, return_value: CheckResult):
        self.calls: list[dict] = []
        self.return_value = return_value

    def perform_check(self, **kwargs):
        self.calls.append(kwargs)
        return self.return_value


class _DummyCheck:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def perform_check(self, **kwargs):
        return _ok_result("dummy")


def test_post_init_builds_all_delegates(
    monkeypatch, rectangular_beam_with_rebars, concrete_c30, shear_links
):
    monkeypatch.setattr(beam_mod, "BendingCheck", _DummyCheck)
    monkeypatch.setattr(beam_mod, "ShearCheck", _DummyCheck)
    monkeypatch.setattr(beam_mod, "CrackingCheck", _DummyCheck)
    monkeypatch.setattr(beam_mod, "StressLimitsCheck", _DummyCheck)

    check = BeamCheck(
        section=rectangular_beam_with_rebars,
        concrete=concrete_c30,
        shear_reinforcement=shear_links,
    )

    assert isinstance(check.bending, _DummyCheck)
    assert isinstance(check.shear, _DummyCheck)
    assert isinstance(check.cracking, _DummyCheck)
    assert isinstance(check.stress_limits, _DummyCheck)

    assert check.shear.kwargs["shear_reinforcement"] is shear_links
    assert check.cracking.kwargs["concrete_model_type"] == ConcreteModelType.LINEAR_ELASTIC
    assert check.stress_limits.kwargs["concrete_model_type"] == ConcreteModelType.LINEAR_ELASTIC


def test_wrapper_methods_forward_to_sub_checks():
    check = object.__new__(BeamCheck)

    default_shear_reinf = SimpleNamespace(name="default")
    bending_rec = _Recorder(_ok_result("bending"))
    shear_rec = _Recorder(_ok_result("shear"))
    cracking_rec = _Recorder(_ok_result("cracking"))
    stress_rec = _Recorder(_ok_result("stress"))

    object.__setattr__(check, "shear_reinforcement", default_shear_reinf)
    object.__setattr__(check, "_bending_check", bending_rec)
    object.__setattr__(check, "_shear_check", shear_rec)
    object.__setattr__(check, "_cracking_check", cracking_rec)
    object.__setattr__(check, "_stress_limits_check", stress_rec)
    object.__setattr__(check, "_ndp_snapshot", get_ndp_context())

    bend_result = check.perform_bending_check(
        M_Ed=120.0,
        N_Ed=20.0,
        V_Ed=100.0,
        M_cap=150.0,
        cot_theta_override=1.4,
        suppress_warnings=True,
    )
    assert bend_result.check_name == "bending"
    assert bending_rec.calls[-1]["My_Ed"] == pytest.approx(120.0, rel=1e-12)
    assert bending_rec.calls[-1]["shear_reinforcement"] is default_shear_reinf
    assert bending_rec.calls[-1]["cot_theta_override"] == pytest.approx(1.4, rel=1e-12)

    load_case = ShearLoadCase(V_Ed=200.0, M_Ed=80.0, N_Ed=50.0)
    shear_result = check.perform_shear_check(
        load_case=load_case,
        use_uncracked_V_Rd_c=True,
        suppress_warnings=True,
    )
    assert shear_result.check_name == "shear"
    assert shear_rec.calls[-1]["load_case"] == load_case
    assert shear_rec.calls[-1]["use_uncracked_V_Rd_c"] is True

    cracking_result = check.perform_cracking_check(
        M_Ed=50.0,
        N_Ed=10.0,
        force_cracked=True,
        suppress_warnings=True,
    )
    assert cracking_result.check_name == "cracking"
    assert cracking_rec.calls[-1]["force_cracked"] is True
    assert cracking_rec.calls[-1]["suppress_warnings"] is True

    stress_result = check.perform_stress_limits_check(
        M_Ed=50.0,
        N_Ed=10.0,
        suppress_warnings=True,
        check_k1_stress=True,
    )
    assert stress_result.check_name == "stress"
    assert stress_rec.calls[-1]["suppress_warnings"] is True
    assert stress_rec.calls[-1]["check_k1_stress"] is True


def test_with_updates_rebuilds_sub_checks(
    monkeypatch, rectangular_beam_with_rebars, concrete_c30, shear_links
):
    monkeypatch.setattr(beam_mod, "BendingCheck", _DummyCheck)
    monkeypatch.setattr(beam_mod, "ShearCheck", _DummyCheck)
    monkeypatch.setattr(beam_mod, "CrackingCheck", _DummyCheck)
    monkeypatch.setattr(beam_mod, "StressLimitsCheck", _DummyCheck)

    check = BeamCheck(
        section=rectangular_beam_with_rebars,
        concrete=concrete_c30,
        shear_reinforcement=shear_links,
        use_mechanical_lever_arm=True,
        w_k_limit=0.30,
    )

    updated = check.with_updates(use_mechanical_lever_arm=False, w_k_limit=0.20)

    assert updated is not check
    assert check.use_mechanical_lever_arm is True
    assert updated.use_mechanical_lever_arm is False
    assert updated.w_k_limit == pytest.approx(0.20, rel=1e-12)
    assert check.shear.kwargs["use_mechanical_lever_arm"] is True
    assert updated.shear.kwargs["use_mechanical_lever_arm"] is False


def test_ndp_context_warning():
    check = object.__new__(BeamCheck)
    object.__setattr__(check, "_ndp_snapshot", ("old", "context"))

    with pytest.warns(UserWarning, match="NDP context has changed"):
        check._check_ndp_context()


def test_ec2_package_exports_beam_check():
    assert hasattr(ec2_mod, "BeamCheck")
    assert "BeamCheck" in ec2_mod.__all__
