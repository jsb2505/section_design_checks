"""Additional branch tests for interaction diagram internals."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import materials.reinforced_concrete.analysis.interaction_diagram as interaction_diagram
from materials.reinforced_concrete.analysis.interaction_diagram import (
    InteractionPoint,
    MNInteractionDiagram,
    _ray_segment_intersection_alpha,
)
from materials.reinforced_concrete.code_checks.ec2_2004.shear_utils import TensionShiftResult


class _FakeConcreteModel:
    def __init__(self, *, eps_u: float = 0.0035, is_ec2_confined: bool = False) -> None:
        self._eps_u = float(eps_u)
        self.is_ec2_confined = bool(is_ec2_confined)

    def get_stress_array(self, strains: np.ndarray) -> np.ndarray:
        arr = np.asarray(strains, dtype=float)
        return np.where(arr > 0.0, 10_000.0 * arr, 0.0)

    def get_tangent_modulus_array(self, strains: np.ndarray) -> np.ndarray:
        return np.full_like(np.asarray(strains, dtype=float), 10_000.0)

    def get_ultimate_strain(self) -> float:
        return self._eps_u


class _FakeSteelModel:
    def __init__(self, *, epsilon_y: float = 0.002, ultimate: float = np.inf) -> None:
        self.epsilon_y = float(epsilon_y)
        self._ultimate = float(ultimate)

    def get_stress_array(self, strains: np.ndarray) -> np.ndarray:
        arr = np.asarray(strains, dtype=float)
        return 200_000.0 * arr

    def get_tangent_modulus_array(self, strains: np.ndarray) -> np.ndarray:
        return np.full_like(np.asarray(strains, dtype=float), 200_000.0)

    def get_ultimate_strain(self) -> float:
        return self._ultimate


def _ip(m: float, n: float) -> InteractionPoint:
    return InteractionPoint(
        N=float(n),
        M=float(m),
        neutral_axis_depth=100.0,
        compression_from_bottom=False,
        max_concrete_strain=0.001,
        max_steel_strain=0.001,
    )


@pytest.fixture
def diagram(rectangular_beam_with_rebars, concrete_c30):
    return MNInteractionDiagram(
        section=rectangular_beam_with_rebars,
        concrete=concrete_c30,
        n_fibres_width=6,
        n_fibres_height=8,
    )


def test_ray_segment_parallel_returns_none() -> None:
    assert _ray_segment_intersection_alpha((1.0, 0.0), (0.0, 1.0), (2.0, 1.0)) is None


def test_init_accepts_override_models(rectangular_beam_with_rebars, concrete_c30) -> None:
    fake_concrete = _FakeConcreteModel()
    fake_steel = _FakeSteelModel(ultimate=0.015)
    diag = MNInteractionDiagram(
        section=rectangular_beam_with_rebars,
        concrete=concrete_c30,
        concrete_model_override=fake_concrete,
        steel_models_override=[fake_steel],
    )
    assert diag.concrete_model is fake_concrete
    assert diag.steel_models[0] is fake_steel


def test_init_empty_steel_override_raises(rectangular_beam_with_rebars, concrete_c30) -> None:
    with pytest.raises(ValueError, match="at least one model"):
        MNInteractionDiagram(
            section=rectangular_beam_with_rebars,
            concrete=concrete_c30,
            steel_models_override=[],
        )


def test_init_confined_with_override_raises(rectangular_beam_with_rebars, concrete_c30) -> None:
    with pytest.raises(ValueError, match="concrete_model_override"):
        MNInteractionDiagram(
            section=rectangular_beam_with_rebars,
            concrete=concrete_c30,
            confined_concrete=True,
            confinement_rho_s=0.02,
            concrete_model_override=_FakeConcreteModel(),
        )


def test_init_rejects_double_confinement_model(
    rectangular_beam_with_rebars, concrete_c30, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        interaction_diagram,
        "create_concrete_stress_strain",
        lambda **_: _FakeConcreteModel(is_ec2_confined=True),
    )
    with pytest.raises(ValueError, match="already has EC2"):
        MNInteractionDiagram(
            section=rectangular_beam_with_rebars,
            concrete=concrete_c30,
            confined_concrete=True,
            confinement_rho_s=0.02,
            confinement_f_yh=500.0,
        )


def test_init_confined_invalid_fyh_raises(rectangular_beam_with_rebars, concrete_c30) -> None:
    with pytest.raises(ValueError, match="must be > 0"):
        MNInteractionDiagram(
            section=rectangular_beam_with_rebars,
            concrete=concrete_c30,
            confined_concrete=True,
            confinement_rho_s=0.02,
            confinement_f_yh=0.0,
        )


def test_init_invalid_section_height_raises(
    rectangular_beam_with_rebars, concrete_c30, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _FakeMesh:
        def __init__(self, *args, **kwargs) -> None:
            self.total_fibres = 1

        def get_fibre_arrays(self):
            return (
                np.array([0.0]),
                np.array([0.0]),
                np.array([1.0]),
                np.array(["concrete"]),
                np.array([0]),
                np.array([0]),
                np.array([0]),
            )

    monkeypatch.setattr(interaction_diagram, "FibreMesh", _FakeMesh)
    monkeypatch.setattr(type(rectangular_beam_with_rebars), "get_bounding_box", lambda self: (0.0, 10.0, 300.0, 10.0))
    with pytest.raises(ValueError, match="Section height must be > 0"):
        MNInteractionDiagram(section=rectangular_beam_with_rebars, concrete=concrete_c30)


def test_calculate_section_forces_without_section_centroid(diagram: MNInteractionDiagram) -> None:
    stresses = np.ones_like(diagram._fibre_area)
    n1, _ = diagram.calculate_section_forces(stresses, use_section_centroid=True)
    n2, m2 = diagram.calculate_section_forces(stresses, use_section_centroid=False)
    assert n2 == pytest.approx(n1)
    assert np.isfinite(m2)


def test_concrete_tangent_modulus_tension_stiffening_regions(
    rectangular_beam_with_rebars, concrete_c30
) -> None:
    diag = MNInteractionDiagram(
        section=rectangular_beam_with_rebars,
        concrete=concrete_c30,
        tension_stiffening=True,
    )
    eps_cr = float(diag.concrete.f_ctm) / float(diag.concrete.E_cm)
    strains = np.array([-0.5 * eps_cr, -1.5 * eps_cr, -20.0 * eps_cr, +0.0005], dtype=float)
    tangent = diag._concrete_tangent_modulus_with_options(strains)
    assert tangent[0] == pytest.approx(float(diag.concrete.E_cm), rel=1e-6)
    assert tangent[1] < 0.0
    assert tangent[2] == pytest.approx(0.0, abs=1e-12)
    assert np.isfinite(tangent[3])


def test_ignore_compression_steel_and_fibre_forces(
    rectangular_beam_with_rebars, concrete_c30
) -> None:
    diag_keep = MNInteractionDiagram(section=rectangular_beam_with_rebars, concrete=concrete_c30)
    diag_drop = MNInteractionDiagram(
        section=rectangular_beam_with_rebars,
        concrete=concrete_c30,
        ignore_compression_steel=True,
    )

    point_keep = diag_keep.calculate_point_from_end_strains(eps_top=0.001, eps_bottom=0.001)
    point_drop = diag_drop.calculate_point_from_end_strains(eps_top=0.001, eps_bottom=0.001)
    assert point_drop.N < point_keep.N

    forces, _, _ = diag_drop.get_fibre_forces_from_end_strains(eps_top=0.001, eps_bottom=0.001)
    strains = diag_drop._strain_field_from_end_strains(eps_top=0.001, eps_bottom=0.001)
    steel_mask = diag_drop._fibre_mat == "steel"
    assert np.allclose(forces[steel_mask & (strains > 0.0)], 0.0)


def test_static_helper_edge_branches() -> None:
    assert np.allclose(MNInteractionDiagram._cosine_space(1), np.array([0.0]))
    assert MNInteractionDiagram._dedupe_pairs([]) == []

    with pytest.raises(ValueError, match="n_out must be >= 3"):
        MNInteractionDiagram._resample_closed_polyline_by_chord([_ip(0, 0), _ip(1, 1)], n_out=2)

    short = [_ip(0, 0), _ip(1, 0), _ip(0, 0)]
    assert MNInteractionDiagram._resample_closed_polyline_by_chord(short, n_out=6) == short

    identical = [_ip(0, 0), _ip(0, 0), _ip(0, 0), _ip(0, 0)]
    out_identical = MNInteractionDiagram._resample_closed_polyline_by_chord(identical, n_out=5)
    assert len(out_identical) == 5
    assert all(p.M == 0.0 and p.N == 0.0 for p in out_identical)

    repeated = [_ip(0, 0), _ip(1, 0), _ip(1, 0), _ip(0, 1), _ip(0, 0)]
    out_repeated = MNInteractionDiagram._resample_closed_polyline_by_chord(repeated, n_out=9)
    assert len(out_repeated) == 9


def test_resample_closed_polyline_padding_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    points = [_ip(0, 0), _ip(0, 0), _ip(1, 0), _ip(0, 1), _ip(0, 0)]
    monkeypatch.setattr(
        interaction_diagram.np,
        "searchsorted",
        lambda _arr, target, side="right": np.ones_like(np.asarray(target), dtype=int),
    )
    out = MNInteractionDiagram._resample_closed_polyline_by_chord(points, n_out=7)
    assert len(out) == 7
    assert all((p.M, p.N) == (out[0].M, out[0].N) for p in out)


def test_eps_tension_limit_with_infinite_ultimate(rectangular_beam_with_rebars, concrete_c30) -> None:
    diag = MNInteractionDiagram(
        section=rectangular_beam_with_rebars,
        concrete=concrete_c30,
        concrete_model_override=_FakeConcreteModel(),
        steel_models_override=[_FakeSteelModel(epsilon_y=0.003, ultimate=np.inf)],
    )
    assert diag._eps_tension_limit() == pytest.approx(0.03)


def test_find_strains_fast_path_uses_analytical_jacobian(
    diagram: MNInteractionDiagram, monkeypatch: pytest.MonkeyPatch
) -> None:
    jac_calls: list[tuple[float, float]] = []
    x0_seen: list[np.ndarray] = []

    monkeypatch.setattr(diagram, "_estimate_initial_strains", lambda *_: (99.0, -99.0))
    monkeypatch.setattr(
        diagram,
        "_compute_analytical_jacobian",
        lambda eps_top, eps_bottom: jac_calls.append((float(eps_top), float(eps_bottom))) or np.eye(2),
    )

    def _fake_least_squares(fun, x0, bounds, jac, **_kwargs):
        x0 = np.asarray(x0, dtype=float)
        x0_seen.append(x0)
        _ = fun(x0)
        if callable(jac):
            _ = jac(x0)
        return SimpleNamespace(x=np.array([0.001, -0.001]), fun=np.array([0.1, 0.2]), success=True)

    monkeypatch.setattr(interaction_diagram, "least_squares", _fake_least_squares)
    eps_top, eps_bottom = diagram.find_strains_for_MN(M_target=10.0, N_target=20.0)
    assert x0_seen
    assert jac_calls
    assert eps_top == pytest.approx(0.001)
    assert eps_bottom == pytest.approx(-0.001)


def test_find_strains_strict_raises_after_fallback(
    diagram: MNInteractionDiagram, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = {"n": 0}

    def _fake_least_squares(*_args, **_kwargs):
        calls["n"] += 1
        return SimpleNamespace(x=np.array([0.0005, -0.0004]), fun=np.array([10.0, 10.0]), success=False)

    monkeypatch.setattr(interaction_diagram, "least_squares", _fake_least_squares)
    with pytest.raises(ValueError, match="could not match"):
        diagram.find_strains_for_MN(M_target=500.0, N_target=100.0, strict=True)
    assert calls["n"] > 1


def test_find_strains_returns_after_fallback_pass1_success(
    diagram: MNInteractionDiagram, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = {"n": 0}

    def _fake_least_squares(*_args, **_kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return SimpleNamespace(x=np.array([0.0005, -0.0004]), fun=np.array([5.0, 5.0]), success=False)
        return SimpleNamespace(x=np.array([0.0015, -0.0010]), fun=np.array([0.1, 0.1]), success=True)

    monkeypatch.setattr(interaction_diagram, "least_squares", _fake_least_squares)
    out = diagram.find_strains_for_MN(M_target=100.0, N_target=50.0, strict=False)
    assert out == pytest.approx((0.0015, -0.0010))
    assert calls["n"] >= 2


def test_find_strains_tension_stiffening_runs_second_pass(
    rectangular_beam_with_rebars, concrete_c30, monkeypatch: pytest.MonkeyPatch
) -> None:
    diag = MNInteractionDiagram(
        section=rectangular_beam_with_rebars,
        concrete=concrete_c30,
        tension_stiffening=True,
    )
    jac_args: list[object] = []

    def _fake_least_squares(fun, x0, jac, **_kwargs):
        x0 = np.asarray(x0, dtype=float)
        jac_args.append(jac)
        _ = fun(x0)
        if callable(jac):
            _ = jac(x0)
            return SimpleNamespace(x=np.array([0.0001, -0.0001]), fun=np.array([5.0, 5.0]), success=False)
        return SimpleNamespace(x=np.array([0.0015, -0.0010]), fun=np.array([0.05, 0.05]), success=True)

    monkeypatch.setattr(interaction_diagram, "least_squares", _fake_least_squares)
    eps_top, eps_bottom = diag.find_strains_for_MN(M_target=60.0, N_target=30.0)
    assert any(j == "2-point" for j in jac_args)
    assert eps_top == pytest.approx(0.0015)
    assert eps_bottom == pytest.approx(-0.0010)


def test_find_strains_confined_uses_numerical_jacobian(
    rectangular_beam_with_rebars, concrete_c30, monkeypatch: pytest.MonkeyPatch
) -> None:
    diag = MNInteractionDiagram(
        section=rectangular_beam_with_rebars,
        concrete=concrete_c30,
        confined_concrete=True,
        confinement_rho_s=0.02,
        confinement_f_yh=500.0,
    )
    seen: dict[str, object] = {}

    def _fake_least_squares(fun, x0, jac, max_nfev, **_kwargs):
        _ = fun(np.asarray(x0, dtype=float))
        seen["jac"] = jac
        seen["max_nfev"] = max_nfev
        return SimpleNamespace(x=np.array([0.001, 0.001]), fun=np.array([0.0, 0.0]), success=True)

    monkeypatch.setattr(interaction_diagram, "least_squares", _fake_least_squares)
    diag.find_strains_for_MN(M_target=50.0, N_target=100.0)
    assert seen["jac"] == "2-point"
    assert seen["max_nfev"] == 200


def test_find_strains_unstable_solution_raises(
    diagram: MNInteractionDiagram, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _fake_least_squares(*_args, **_kwargs):
        return SimpleNamespace(x=np.array([np.nan, np.nan]), fun=np.array([np.inf, np.inf]), success=False)

    monkeypatch.setattr(interaction_diagram, "least_squares", _fake_least_squares)
    with pytest.raises(ValueError, match="numerically unstable"):
        diagram.find_strains_for_MN(M_target=40.0, N_target=20.0)


@pytest.mark.parametrize("n_target", [100.0, -100.0, 0.0])
def test_find_strains_zero_moment_candidate_branches(
    diagram: MNInteractionDiagram, monkeypatch: pytest.MonkeyPatch, n_target: float
) -> None:
    monkeypatch.setattr(
        interaction_diagram,
        "least_squares",
        lambda *_args, **_kwargs: SimpleNamespace(
            x=np.array([0.0, 0.0]), fun=np.array([0.0, 0.0]), success=True
        ),
    )
    out = diagram.find_strains_for_MN(M_target=0.0, N_target=n_target)
    assert out == pytest.approx((0.0, 0.0))


def test_find_strains_negative_moment_high_eccentricity_branch(
    diagram: MNInteractionDiagram, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        interaction_diagram,
        "least_squares",
        lambda *_args, **_kwargs: SimpleNamespace(
            x=np.array([-0.001, 0.001]), fun=np.array([0.0, 0.0]), success=True
        ),
    )
    out = diagram.find_strains_for_MN(M_target=-500.0, N_target=100.0)
    assert out == pytest.approx((-0.001, 0.001))


def test_compute_analytical_jacobian_returns_finite_matrix(diagram: MNInteractionDiagram) -> None:
    jac = diagram._compute_analytical_jacobian(eps_top=0.002, eps_bottom=-0.001)
    assert jac.shape == (2, 2)
    assert np.all(np.isfinite(jac))


def test_confined_concrete_denominator_guard_branch(
    rectangular_beam_with_rebars, concrete_c30
) -> None:
    diag = MNInteractionDiagram(
        section=rectangular_beam_with_rebars,
        concrete=concrete_c30,
        confined_concrete=True,
        confinement_rho_s=0.02,
        confinement_f_yh=500.0,
    )

    rho_s = 0.02
    f_yh_k = 500.0
    f_co_k = 30.0
    eps_co = 0.002
    f_l_k = 0.5 * 0.75 * rho_s * f_yh_k
    term = 1.0 + 7.94 * f_l_k / f_co_k
    f_cc_k = f_co_k * (2.254 * np.sqrt(term) - 2.0 * f_l_k / f_co_k - 1.254)
    eps_cc = eps_co * (1.0 + 5.0 * (max(f_cc_k / f_co_k, 1e-6) - 1.0))
    e_target = f_cc_k / eps_cc  # makes denom = E_cm - (f_cc_k/eps_cc) -> ~0

    diag.concrete = SimpleNamespace(
        f_ck=f_co_k,
        epsilon_c2=eps_co,
        alpha_cc=1.0,
        gamma_c=1.0,
        E_cm=e_target,
        f_ctm=2.9,
    )
    stresses = diag._concrete_stress_with_options(np.array([0.001], dtype=float))
    assert np.isfinite(stresses[0])


@pytest.mark.parametrize(
    ("m_val", "n_val", "expected_fn"),
    [
        (0.0, 10.0, lambda eps_cu, eps_y: (+eps_cu * 0.8, +eps_cu * 0.8)),
        (5.0, 10.0, lambda eps_cu, eps_y: (+eps_cu * 0.8, +eps_cu * 0.2)),
        (-5.0, 10.0, lambda eps_cu, eps_y: (+eps_cu * 0.2, +eps_cu * 0.8)),
        (0.0, -10.0, lambda eps_cu, eps_y: (-eps_y * 2.0, -eps_y * 2.0)),
        (5.0, -10.0, lambda eps_cu, eps_y: (-eps_y, -eps_y * 3.0)),
        (-5.0, -10.0, lambda eps_cu, eps_y: (-eps_y * 3.0, -eps_y)),
        (5.0, 0.0, lambda eps_cu, eps_y: (+eps_cu * 0.8, -eps_y * 2.0)),
        (-5.0, 0.0, lambda eps_cu, eps_y: (-eps_y * 2.0, +eps_cu * 0.8)),
        (0.0, 0.0, lambda eps_cu, eps_y: (0.0, 0.0)),
    ],
)
def test_estimate_initial_strains_quadrants(
    diagram: MNInteractionDiagram,
    m_val: float,
    n_val: float,
    expected_fn,
) -> None:
    eps_cu = float(diagram.concrete_model.get_ultimate_strain())
    eps_y = float(diagram.steel_models[0].epsilon_y)
    expected = expected_fn(eps_cu, eps_y)
    assert diagram._estimate_initial_strains(m_val, n_val) == pytest.approx(expected)


def test_get_effective_depth_branches(diagram: MNInteractionDiagram, monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_section_depth(self, compression_face: str, **_kwargs) -> float:
        return 520.0 if compression_face == "top" else 410.0

    monkeypatch.setattr(type(diagram.section), "get_effective_depth", _fake_section_depth)
    d_top = diagram.section.get_effective_depth(compression_face="top")
    d_bottom = diagram.section.get_effective_depth(compression_face="bottom")
    assert diagram.get_effective_depth(M_Ed=0.0, N_Ed=123.0) == pytest.approx(min(d_top, d_bottom))

    monkeypatch.setattr(diagram, "find_strains_for_MN", lambda *_args, **_kwargs: (0.002, 0.001))
    auto_depth = diagram.get_effective_depth(M_Ed=20.0, N_Ed=100.0)
    assert auto_depth == pytest.approx(d_top)

    d_from_top = diagram.get_effective_depth(M_Ed=20.0, N_Ed=0.0, eps_top=0.002, eps_bottom=0.001)
    d_from_bottom = diagram.get_effective_depth(M_Ed=20.0, N_Ed=0.0, eps_top=0.001, eps_bottom=0.002)
    assert d_from_top == pytest.approx(d_top)
    assert d_from_bottom == pytest.approx(d_bottom)


def test_get_lever_arm_invalid_depth_raises(diagram: MNInteractionDiagram) -> None:
    with pytest.raises(ValueError, match="must be > 0"):
        diagram.get_lever_arm(M_Ed=10.0, N_Ed=0.0, d=0.0)


def test_get_lever_arm_simple_09d_path(diagram: MNInteractionDiagram) -> None:
    z, z_mech = diagram.get_lever_arm(M_Ed=10.0, N_Ed=0.0, d=500.0, prefer_rigorous=False)
    assert z == pytest.approx(450.0)
    assert z_mech is None


def test_get_lever_arm_computes_depth_when_missing(
    diagram: MNInteractionDiagram, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(diagram, "get_effective_depth", lambda *_args, **_kwargs: 500.0)
    z, z_mech = diagram.get_lever_arm(M_Ed=10.0, N_Ed=0.0, d=None, prefer_rigorous=False)
    assert z == pytest.approx(450.0)
    assert z_mech is None


def test_get_lever_arm_fallback_and_cap_branches(
    diagram: MNInteractionDiagram, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.warns(UserWarning, match="fallback to 0.9d"):
        z0, z_mech0 = diagram.get_lever_arm(
            M_Ed=0.0,
            N_Ed=0.0,
            d=500.0,
            prefer_rigorous=True,
            warn_on_fallback=True,
        )
    assert z0 == pytest.approx(450.0)
    assert z_mech0 is None

    monkeypatch.setattr(diagram, "_compute_lever_arm_from_centroids", lambda *_: None)
    with pytest.warns(UserWarning, match="unable to compute"):
        z1, z_mech1 = diagram.get_lever_arm(
            M_Ed=10.0,
            N_Ed=0.0,
            d=500.0,
            eps_top=0.001,
            eps_bottom=-0.001,
            prefer_rigorous=True,
            warn_on_fallback=True,
        )
    assert z1 == pytest.approx(450.0)
    assert z_mech1 is None

    monkeypatch.setattr(diagram, "_compute_lever_arm_from_centroids", lambda *_: 20.0)
    with pytest.warns(UserWarning, match="likely axial-dominated"):
        z2, z_mech2 = diagram.get_lever_arm(
            M_Ed=10.0,
            N_Ed=0.0,
            d=500.0,
            eps_top=0.001,
            eps_bottom=-0.001,
            prefer_rigorous=True,
            min_z_ratio=0.10,
            warn_on_fallback=True,
        )
    assert z2 == pytest.approx(450.0)
    assert z_mech2 == pytest.approx(20.0)

    monkeypatch.setattr(diagram, "_compute_lever_arm_from_centroids", lambda *_: 600.0)
    with pytest.warns(UserWarning, match="Lever arm capped"):
        z3, z_mech3 = diagram.get_lever_arm(
            M_Ed=10.0,
            N_Ed=0.0,
            d=500.0,
            eps_top=0.001,
            eps_bottom=-0.001,
            prefer_rigorous=True,
            cap_to_09d=True,
        )
    assert z3 == pytest.approx(450.0)
    assert z_mech3 == pytest.approx(600.0)


def test_get_lever_arm_valid_rigorous_value(diagram: MNInteractionDiagram, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(diagram, "find_strains_for_MN", lambda **_kwargs: (0.002, -0.001))
    monkeypatch.setattr(diagram, "_compute_lever_arm_from_centroids", lambda *_: 300.0)
    z, z_mech = diagram.get_lever_arm(
        M_Ed=50.0,
        N_Ed=10.0,
        d=500.0,
        prefer_rigorous=True,
        cap_to_09d=False,
    )
    assert z == pytest.approx(300.0)
    assert z_mech == pytest.approx(300.0)


def test_compute_lever_arm_from_centroids_branches(
    diagram: MNInteractionDiagram, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        diagram,
        "get_fibre_forces_from_end_strains",
        lambda *_: (np.array([5.0, 10.0]), np.array([0.0, 100.0]), np.array([1.0, 1.0])),
    )
    assert diagram._compute_lever_arm_from_centroids(0.001, 0.001) is None

    monkeypatch.setattr(
        diagram,
        "get_fibre_forces_from_end_strains",
        lambda *_: (np.array([-5.0, 10.0]), np.array([np.nan, 100.0]), np.array([1.0, 1.0])),
    )
    assert diagram._compute_lever_arm_from_centroids(0.001, -0.001) is None

    monkeypatch.setattr(
        diagram,
        "get_fibre_forces_from_end_strains",
        lambda *_: (np.array([-5.0, 10.0]), np.array([0.0, 200.0]), np.array([1.0, 1.0])),
    )
    assert diagram._compute_lever_arm_from_centroids(0.001, -0.001) == pytest.approx(200.0)


def test_compute_lever_arm_from_centroids_zero_total_branch(
    diagram: MNInteractionDiagram, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        diagram,
        "get_fibre_forces_from_end_strains",
        lambda *_: (np.array([-1.0, 2.0]), np.array([0.0, 100.0]), np.array([1.0, 1.0])),
    )
    real_sum = interaction_diagram.np.sum
    calls = {"n": 0}

    def _fake_sum(values, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return 0.0
        return real_sum(values, *args, **kwargs)

    monkeypatch.setattr(interaction_diagram.np, "sum", _fake_sum)
    assert diagram._compute_lever_arm_from_centroids(0.001, -0.001) is None


def test_capacity_vector_insufficient_points(diagram: MNInteractionDiagram, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(diagram, "generate_diagram_points", lambda **_kwargs: (_ip(0, 0), _ip(1, 1)))
    result = diagram.get_capacity_vector(N_Ed=10.0, M_Ed=5.0)
    assert result.N_Rd is None
    assert result.M_Rd is None
    assert result.is_safe is False


def test_capacity_vector_origin_with_details(diagram: MNInteractionDiagram, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(diagram, "generate_diagram_points", lambda **_kwargs: (_ip(0, 0), _ip(1, 0), _ip(0, 1), _ip(0, 0)))
    result = diagram.get_capacity_vector(N_Ed=0.0, M_Ed=0.0, return_details=True)
    assert result.N_Rd == 0.0
    assert result.M_Rd == 0.0
    assert result.details is not None
    assert result.details["neutral_axis_depth"] is None


def test_capacity_vector_no_intersection(diagram: MNInteractionDiagram, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(diagram, "generate_diagram_points", lambda **_kwargs: (_ip(0, 0), _ip(1, 0), _ip(1, 1), _ip(0, 0)))
    monkeypatch.setattr(interaction_diagram, "_ray_segment_intersection_alpha", lambda *_args, **_kwargs: None)
    result = diagram.get_capacity_vector(N_Ed=10.0, M_Ed=5.0)
    assert result.N_Rd is None
    assert result.M_Rd is None
    assert result.utilization == pytest.approx(float("inf"))


def test_capacity_vector_multiple_intersections_warns_and_uses_min(
    diagram: MNInteractionDiagram, monkeypatch: pytest.MonkeyPatch
) -> None:
    pts = (_ip(0, 0), _ip(1, 0), _ip(1, 0), _ip(0, 1), _ip(-1, 0))
    monkeypatch.setattr(diagram, "generate_diagram_points", lambda **_kwargs: pts)
    values = iter([0.8, 0.6, 0.4, 0.2])
    monkeypatch.setattr(interaction_diagram, "_ray_segment_intersection_alpha", lambda *_args, **_kwargs: next(values, None))

    with pytest.warns(UserWarning, match="intersections"):
        result = diagram.get_capacity_vector(N_Ed=5.0, M_Ed=10.0)
    assert result.N_Rd == pytest.approx(1.0)
    assert result.M_Rd == pytest.approx(2.0)
    assert result.utilization == pytest.approx(5.0)


def test_capacity_vector_return_details_success(
    diagram: MNInteractionDiagram, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(diagram, "generate_diagram_points", lambda **_kwargs: (_ip(0, 0), _ip(1, 0), _ip(0, 1), _ip(0, 0)))
    values = iter([2.0, None, None])
    monkeypatch.setattr(interaction_diagram, "_ray_segment_intersection_alpha", lambda *_args, **_kwargs: next(values, None))
    monkeypatch.setattr(diagram, "find_strains_for_MN", lambda *_args, **_kwargs: (0.002, -0.001))
    result = diagram.get_capacity_vector(N_Ed=20.0, M_Ed=10.0, return_details=True)
    assert result.details is not None
    assert result.details["neutral_axis_depth"] is not None
    assert result.details["max_concrete_strain"] > 0.0


def test_capacity_vector_return_details_uniform_strain(
    diagram: MNInteractionDiagram, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(diagram, "generate_diagram_points", lambda **_kwargs: (_ip(0, 0), _ip(1, 0), _ip(0, 1), _ip(0, 0)))
    values = iter([1.5, None, None])
    monkeypatch.setattr(interaction_diagram, "_ray_segment_intersection_alpha", lambda *_args, **_kwargs: next(values, None))
    monkeypatch.setattr(diagram, "find_strains_for_MN", lambda *_args, **_kwargs: (0.001, 0.001))
    result = diagram.get_capacity_vector(N_Ed=20.0, M_Ed=10.0, return_details=True)
    assert result.details is not None
    assert result.details["neutral_axis_depth"] is None
    assert result.details["compression_from_bottom"] is True


def test_capacity_vector_return_details_failure_warns(
    diagram: MNInteractionDiagram, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(diagram, "generate_diagram_points", lambda **_kwargs: (_ip(0, 0), _ip(1, 0), _ip(0, 1), _ip(0, 0)))
    values = iter([1.2, None, None])
    monkeypatch.setattr(interaction_diagram, "_ray_segment_intersection_alpha", lambda *_args, **_kwargs: next(values, None))
    monkeypatch.setattr(diagram, "find_strains_for_MN", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("fail")))
    with pytest.warns(UserWarning, match="Failed to compute exact strain state"):
        result = diagram.get_capacity_vector(N_Ed=20.0, M_Ed=10.0, return_details=True)
    assert result.details is None


def test_intersections_with_horizontal_edge_cases() -> None:
    assert MNInteractionDiagram._intersections_with_horizontal([], N0=0.0) == []
    on_line = MNInteractionDiagram._intersections_with_horizontal([(0.0, 0.0), (2.0, 0.0), (2.0, 1.0)], N0=0.0)
    assert on_line[0] == pytest.approx(0.0)
    assert on_line[-1] == pytest.approx(2.0)
    out_of_range = MNInteractionDiagram._intersections_with_horizontal([(0.0, 0.0), (1.0, 1.0)], N0=1.0 + 1e-10, tol=1e-9)
    assert out_of_range == []


def test_get_capacity_fixed_n_branches(diagram: MNInteractionDiagram, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(diagram, "generate_diagram_points", lambda **_kwargs: (_ip(0, 0), _ip(1, 0), _ip(0, 1)))
    assert diagram.get_capacity_fixed_n(N_Ed=100.0) == (None, None, None)

    monkeypatch.setattr(diagram, "generate_diagram_points", lambda **_kwargs: (_ip(0, 0), _ip(1, 0), _ip(0, 1), _ip(-1, 0)))
    monkeypatch.setattr(diagram, "_intersections_with_horizontal", lambda *_args, **_kwargs: [])
    assert diagram.get_capacity_fixed_n(N_Ed=100.0) == (None, None, None)


def test_plot_stress_strain_wrapper_calls_viewer(
    diagram: MNInteractionDiagram, monkeypatch: pytest.MonkeyPatch
) -> None:
    import materials.reinforced_concrete.analysis.stress_strain_viewer as ss_viewer

    seen: dict[str, object] = {}

    class _FakeViewer:
        def __init__(self, diag):
            seen["diag"] = diag

        def plot(self, **kwargs):
            seen["kwargs"] = kwargs
            return "viewer-ok"

    monkeypatch.setattr(ss_viewer, "StressStrainViewer", _FakeViewer)
    out = diagram.plot_stress_strain(M_Ed=10.0, N_Ed=20.0, show=False, section_render="filled")
    assert out == "viewer-ok"
    assert seen["diag"] is diagram
    assert seen["kwargs"]["M_Ed"] == pytest.approx(10.0)
    assert seen["kwargs"]["N_Ed"] == pytest.approx(20.0)


def test_compute_z_d_for_moment_zero_moment_path(
    diagram: MNInteractionDiagram, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, object] = {}
    monkeypatch.setattr(
        diagram,
        "find_strains_for_MN",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected call")),
    )

    def _fake_get_effective_depth(*, M_Ed, N_Ed, eps_top, eps_bottom):
        seen["eps"] = (eps_top, eps_bottom)
        return 500.0

    monkeypatch.setattr(diagram, "get_effective_depth", _fake_get_effective_depth)
    monkeypatch.setattr(diagram, "get_lever_arm", lambda **_kwargs: (450.0, None))
    z, d = diagram._compute_z_d_for_moment(M_Ed=0.0, N_Ed=100.0)
    assert z == pytest.approx(450.0)
    assert d == pytest.approx(500.0)
    assert seen["eps"] == (None, None)


def test_compute_z_d_for_moment_nonzero_path(
    diagram: MNInteractionDiagram, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(diagram, "find_strains_for_MN", lambda *_args, **_kwargs: (0.002, -0.001))
    seen: dict[str, object] = {}

    def _fake_get_effective_depth(*, M_Ed, N_Ed, eps_top, eps_bottom):
        seen["depth_eps"] = (eps_top, eps_bottom)
        return 520.0

    def _fake_get_lever_arm(**kwargs):
        seen["lever_eps"] = (kwargs["eps_top"], kwargs["eps_bottom"])
        return (460.0, None)

    monkeypatch.setattr(diagram, "get_effective_depth", _fake_get_effective_depth)
    monkeypatch.setattr(diagram, "get_lever_arm", _fake_get_lever_arm)
    z, d = diagram._compute_z_d_for_moment(M_Ed=50.0, N_Ed=100.0)
    assert z == pytest.approx(460.0)
    assert d == pytest.approx(520.0)
    assert seen["depth_eps"] == (0.002, -0.001)
    assert seen["lever_eps"] == (0.002, -0.001)


def test_apply_tension_shift_basic_no_shear(
    diagram: MNInteractionDiagram, monkeypatch: pytest.MonkeyPatch
) -> None:
    import materials.reinforced_concrete.code_checks.ec2_2004.shear_utils as shear_utils

    seen: dict[str, object] = {}
    monkeypatch.setattr(diagram, "_compute_z_d_for_moment", lambda **_kwargs: (300.0, 500.0))

    def _fake_calculate_tension_shift(**kwargs):
        seen["kwargs"] = kwargs
        return TensionShiftResult(
            M_design=kwargs["M_Ed"] + 1.0,
            M_add=1.0,
            shift_distance_a_l=kwargs["d"],
            cot_theta=None,
            capped_by_M_cap=False,
            z=kwargs["z"],
            d=kwargs["d"],
        )

    monkeypatch.setattr(shear_utils, "calculate_tension_shift", _fake_calculate_tension_shift)
    result = diagram.apply_tension_shift(M_Ed=100.0, V_Ed=40.0, N_Ed=20.0)
    assert result.M_design == pytest.approx(101.0)
    assert seen["kwargs"]["b_w"] is None
    assert seen["kwargs"]["sigma_cp"] == pytest.approx(0.0)


def test_apply_tension_shift_iterative_with_shear_reinforcement(
    diagram: MNInteractionDiagram, shear_links, monkeypatch: pytest.MonkeyPatch
) -> None:
    import materials.reinforced_concrete.code_checks.ec2_2004.shear_utils as shear_utils

    z_d_values = iter([(300.0, 500.0), (330.0, 510.0), (331.0, 511.0)])
    monkeypatch.setattr(diagram, "_compute_z_d_for_moment", lambda **_kwargs: next(z_d_values))
    monkeypatch.setattr(shear_utils, "calculate_section_breadth", lambda section: 250.0)
    monkeypatch.setattr(shear_utils, "sigma_cp_from_N_and_area", lambda N_Ed, area: 1.5)
    monkeypatch.setattr(shear_utils, "cap_sigma_cp_upper", lambda sigma_cp, f_cd: 1.0)

    calls: list[dict[str, object]] = []

    def _fake_calculate_tension_shift(**kwargs):
        calls.append(kwargs)
        z = float(kwargs["z"])
        return TensionShiftResult(
            M_design=float(kwargs["M_Ed"]) + z / 1000.0,
            M_add=abs(float(kwargs["V_Ed"])) * 0.1,
            shift_distance_a_l=float(kwargs["d"]),
            cot_theta=1.2,
            capped_by_M_cap=False,
            z=float(kwargs["z"]),
            d=float(kwargs["d"]),
        )

    monkeypatch.setattr(shear_utils, "calculate_tension_shift", _fake_calculate_tension_shift)
    result = diagram.apply_tension_shift(
        M_Ed=100.0,
        V_Ed=40.0,
        N_Ed=200.0,
        shear_reinforcement=shear_links,
        iterate_z=True,
        prefer_rigorous=True,
    )

    assert len(calls) >= 3
    assert calls[0]["b_w"] == pytest.approx(250.0)
    assert calls[0]["sigma_cp"] == pytest.approx(1.0)
    assert result.z == pytest.approx(331.0)
