"""Additional branch tests for bending_check.py."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import materials.reinforced_concrete.code_checks.ec2_2004.bending_check as bending_mod
from materials.reinforced_concrete.code_checks.ec2_2004.bending_check import BendingCheck
from materials.reinforced_concrete.constitutive import ConcreteModelType


class _FakeDiagram:
    def __init__(
        self,
        *,
        raise_on_find: bool = False,
        fixed_n: tuple[float | None, float | None, float | None] = (100.0, 200.0, -200.0),
    ) -> None:
        self.raise_on_find = raise_on_find
        self.fixed_n = fixed_n
        self.plot_mn_kwargs: dict | None = None
        self.plot_stress_kwargs: dict | None = None

    def get_capacity_vector(self, *, N_Ed: float, M_Ed: float, return_details: bool = False):
        return SimpleNamespace(N_Rd=2000.0, M_Rd=500.0, utilization=0.5)

    def find_strains_for_MN(self, M_Ed: float, N_Ed: float):
        if self.raise_on_find:
            raise RuntimeError("no strains")
        return 0.001, -0.001

    def find_strain_state_for_MN(self, *args, **kwargs):
        return None

    def get_capacity_fixed_n(self, *, N_Ed: float):
        return self.fixed_n

    def plot_mn(self, **kwargs):
        self.plot_mn_kwargs = kwargs
        return {"kind": "mn"}

    def plot_stress_strain(self, **kwargs):
        self.plot_stress_kwargs = kwargs
        return {"kind": "stress"}


def test_validator_allows_linear_elastic_with_override(rectangular_beam_with_rebars, concrete_c30):
    """Test validator allows linear elastic with override."""
    check = BendingCheck(
        section=rectangular_beam_with_rebars,
        concrete=concrete_c30,
        concrete_model_type=ConcreteModelType.LINEAR_ELASTIC,
        concrete_model_override=SimpleNamespace(cache_key="override"),
    )
    assert check.concrete_model_type == ConcreteModelType.LINEAR_ELASTIC


def test_validator_rejects_linear_elastic_without_override(rectangular_beam_with_rebars, concrete_c30):
    """Test validator rejects linear elastic without override."""
    with pytest.raises(ValueError, match="LINEAR_ELASTIC concrete model is only valid for SLS checks"):
        BendingCheck(
            section=rectangular_beam_with_rebars,
            concrete=concrete_c30,
            concrete_model_type=ConcreteModelType.LINEAR_ELASTIC,
        )


def test_take_snapshot_includes_override_keys(rectangular_beam_with_rebars, concrete_c30):
    """Test take snapshot includes override keys."""
    check = BendingCheck(section=rectangular_beam_with_rebars, concrete=concrete_c30)
    check.concrete_model_override = SimpleNamespace(cache_key="conc-key")
    check.steel_models_override = [SimpleNamespace(cache_key="s1"), SimpleNamespace()]

    snap = check._take_snapshot()

    assert snap["concrete_override_key"] == "conc-key"
    assert len(snap["steel_override_keys"]) == 2


def test_get_diagram_ignore_compression_steel_cache(monkeypatch, rectangular_beam_with_rebars, concrete_c30):
    """Test get diagram ignore compression steel cache."""
    created: list[dict] = []

    def _fake_factory(**kwargs):
        created.append(kwargs)
        return object()

    monkeypatch.setattr(bending_mod, "create_interaction_diagram", _fake_factory)
    check = BendingCheck(section=rectangular_beam_with_rebars, concrete=concrete_c30)

    d1 = check._get_diagram(ignore_compression_steel=True)
    d2 = check._get_diagram(ignore_compression_steel=True)

    assert d1 is d2
    assert len(created) == 1
    assert created[0]["ignore_compression_steel"] is True


def test_find_tension_steel_area_and_fyk_zero_height_returns_none():
    """Test find tension steel area and fyk zero height returns none."""
    check = object.__new__(BendingCheck)
    object.__setattr__(
        check,
        "section",
        SimpleNamespace(outline=SimpleNamespace(bounds=(0.0, 0.0, 300.0, 0.0)), rebar_groups=[]),
    )

    area, fyk = BendingCheck._find_tension_steel_area_and_f_yk(
        check,
        eps_top=0.001,
        eps_bottom=-0.001,
    )

    assert area == 0.0
    assert fyk is None


def test_check_single_case_requires_ved_when_mcap():
    """Test check single case requires ved when mcap."""
    check = object.__new__(BendingCheck)

    with pytest.raises(ValueError, match="V_Ed must be provided when M_cap is provided"):
        BendingCheck._check_single_case(
            check,
            My_Ed=100.0,
            N_Ed=0.0,
            V_Ed=None,
            M_cap=200.0,
            shear_reinforcement=None,
            warning_threshold=0.95,
        )


def test_perform_check_handles_strain_solver_exception_in_as_min_path(
    monkeypatch, rectangular_beam_with_rebars, concrete_c30
):
    """Test perform check handles strain solver exception in as min path."""
    check = BendingCheck(section=rectangular_beam_with_rebars, concrete=concrete_c30)
    fake = _FakeDiagram(raise_on_find=True)
    monkeypatch.setattr(
        BendingCheck,
        "_get_diagram",
        lambda self, ignore_compression_steel=False: fake,
    )

    result = check.perform_check(My_Ed=100.0, N_Ed=200.0, suppress_warnings=True)

    assert result.details["A_s_min_check_applicable"] is False
    assert result.details["A_s_min_required"] is None


def test_perform_check_handles_effective_depth_value_error_in_as_min_path(
    monkeypatch, rectangular_beam_with_rebars, concrete_c30
):
    """Test perform check handles effective depth value error in as min path."""
    check = BendingCheck(section=rectangular_beam_with_rebars, concrete=concrete_c30)
    fake = _FakeDiagram(raise_on_find=False)
    monkeypatch.setattr(
        BendingCheck,
        "_get_diagram",
        lambda self, ignore_compression_steel=False: fake,
    )
    monkeypatch.setattr(
        BendingCheck,
        "_find_tension_steel_area_and_f_yk",
        lambda self, eps_top, eps_bottom, **kwargs: (120.0, 500.0),
    )

    def _raise_depth(*args, **kwargs):
        raise ValueError("no depth")

    monkeypatch.setattr(bending_mod, "find_effective_depth_for_flexure", _raise_depth)

    result = check.perform_check(My_Ed=100.0, N_Ed=200.0, suppress_warnings=True)

    assert result.details["A_s_min_check_applicable"] is False
    assert result.details["A_s_min_effective_depth_d"] is None


def test_get_moment_capacity_negative_bound_returns_none(monkeypatch, rectangular_beam_with_rebars, concrete_c30):
    """Test get moment capacity negative bound returns none."""
    check = BendingCheck(section=rectangular_beam_with_rebars, concrete=concrete_c30)
    fake = _FakeDiagram(fixed_n=(-50.0, 220.0, -220.0))
    monkeypatch.setattr(
        BendingCheck,
        "_get_diagram",
        lambda self, ignore_compression_steel=False: fake,
    )

    m_pos, m_neg = check.get_moment_capacity(N_Ed=-60.0)

    assert m_pos is None
    assert m_neg is None


def test_plot_wrappers_delegate_to_diagram(monkeypatch, rectangular_beam_with_rebars, concrete_c30):
    """Test plot wrappers delegate to diagram."""
    check = BendingCheck(section=rectangular_beam_with_rebars, concrete=concrete_c30)
    fake = _FakeDiagram()
    monkeypatch.setattr(
        BendingCheck,
        "_get_diagram",
        lambda self, ignore_compression_steel=False: fake,
    )

    fig_mn = check.plot_mn(
        load_points=[{"N_Ed": 100.0, "M_Ed": 50.0}],
        show=False,
        ignore_compression_steel=True,
    )
    fig_ss = check.plot_stress_strain(
        M_Ed=50.0,
        N_Ed=100.0,
        show=False,
        ignore_compression_steel=True,
    )

    assert fig_mn == {"kind": "mn"}
    assert fig_ss == {"kind": "stress"}
    assert fake.plot_mn_kwargs is not None
    assert fake.plot_stress_kwargs is not None
