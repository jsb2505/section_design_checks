"""
Targeted unit tests for biaxial interaction helper branches.
"""

from __future__ import annotations

import builtins
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import section_design_checks.reinforced_concrete.analysis.biaxial_interaction as biax
from section_design_checks.reinforced_concrete.analysis.biaxial_interaction import (
    BiaxialInteractionPoint,
    BiaxialMNInteractionSurface,
)
from section_design_checks.reinforced_concrete.materials import ConcreteMaterial


def _pt(N: float, My: float, Mz: float, depth: float = 100.0, angle: float = 0.0) -> BiaxialInteractionPoint:
    return BiaxialInteractionPoint(
        N=N,
        My=My,
        Mz=Mz,
        neutral_axis_depth=depth,
        neutral_axis_angle=angle,
        max_concrete_strain=0.0035,
        max_steel_strain=0.01,
    )


def _make_surface_stub() -> BiaxialMNInteractionSurface:
    s = object.__new__(BiaxialMNInteractionSurface)
    s._surface_cache = {}
    s._surface_indices_cache = {}
    s._hull_cache = {}
    s._dense_surface_points = None
    s._dense_params = None
    s._dense_grid_indices = []
    s._grid_indices = []
    s._grid_shape = (0, 0)
    s.elastic_modulus = None
    s.include_tension = False
    s.crack_to_neutral_axis_on_first_tension_failure = True
    s.confinement_eps_su = 0.10  # transverse-steel rupture strain (ctor default)
    return s


class _FakeFigure:
    def __init__(self):
        self.traces = []
        self.layout_updates = []
        self.saved_paths = []
        self.shown = False

    def add_trace(self, trace):
        self.traces.append(trace)

    def update_layout(self, **kwargs):
        self.layout_updates.append(kwargs)

    def write_html(self, path):
        self.saved_paths.append(path)

    def show(self):
        self.shown = True


def _install_fake_plotly(monkeypatch):
    go_mod = types.ModuleType("plotly.graph_objects")
    go_mod.Figure = _FakeFigure
    go_mod.Surface = lambda **kwargs: kwargs
    go_mod.Scatter3d = lambda **kwargs: kwargs

    plotly_mod = types.ModuleType("plotly")
    plotly_mod.graph_objects = go_mod

    monkeypatch.setitem(sys.modules, "plotly", plotly_mod)
    monkeypatch.setitem(sys.modules, "plotly.graph_objects", go_mod)


class TestPointAndInitGuards:
    """Tests for TestPointAndInitGuards."""
    def test_point_repr(self):
        """Test point repr."""
        p = _pt(100.0, 20.0, 30.0)
        text = repr(p)
        assert "BiaxialPoint(" in text
        assert "N=100.0" in text
        assert "My=20.0" in text
        assert "Mz=30.0" in text

    def test_init_confined_concrete_guardrails(self, monkeypatch):
        """Test init confined concrete guardrails."""
        section = SimpleNamespace(
            rebar_groups=[SimpleNamespace(rebar=SimpleNamespace(f_yk=500.0))],
            get_centroid=lambda: (0.0, 0.0),
            get_bounding_box=lambda: (0.0, 0.0, 1.0, 1.0),
            section_name="Dummy",
        )
        concrete = ConcreteMaterial(grade="C30/37")

        monkeypatch.setattr(
            biax,
            "create_steel_stress_strain",
            lambda **kwargs: SimpleNamespace(get_ultimate_strain=lambda: 0.05, epsilon_y=0.0025),
        )
        monkeypatch.setattr(
            biax,
            "FibreMesh",
            lambda **kwargs: SimpleNamespace(
                total_fibres=1,
                get_fibre_arrays=lambda: (
                    np.array([0.0]),
                    np.array([0.0]),
                    np.array([1.0]),
                    np.array(["concrete"]),
                    np.array([0], dtype=int),
                    np.array([0], dtype=int),
                    np.array([0], dtype=int),
                ),
            ),
        )

        monkeypatch.setattr(
            biax,
            "create_concrete_stress_strain",
            lambda **kwargs: SimpleNamespace(is_ec2_confined=True),
        )
        with pytest.raises(ValueError, match="already has EC2"):
            BiaxialMNInteractionSurface(section=section, concrete=concrete, confined_concrete=True, confinement_rho_s=0.01)

        monkeypatch.setattr(
            biax,
            "create_concrete_stress_strain",
            lambda **kwargs: SimpleNamespace(is_ec2_confined=False),
        )
        with pytest.raises(ValueError, match="confinement_rho_s must be provided"):
            BiaxialMNInteractionSurface(section=section, concrete=concrete, confined_concrete=True)

        with pytest.raises(ValueError, match="must be in"):
            BiaxialMNInteractionSurface(
                section=section,
                concrete=concrete,
                confined_concrete=True,
                confinement_rho_s=0.2,
            )

        section_bad_fyh = SimpleNamespace(
            rebar_groups=[SimpleNamespace(rebar=SimpleNamespace(f_yk=-1.0))],
            get_centroid=lambda: (0.0, 0.0),
            get_bounding_box=lambda: (0.0, 0.0, 1.0, 1.0),
            section_name="Dummy",
        )
        with pytest.raises(ValueError, match="must be > 0"):
            BiaxialMNInteractionSurface(
                section=section_bad_fyh,
                concrete=concrete,
                confined_concrete=True,
                confinement_rho_s=0.01,
            )


class TestCoreHelpers:
    """Tests for TestCoreHelpers."""
    def test_eps_tension_limit_finite_and_infinite(self):
        """Test eps tension limit finite and infinite."""
        surface = _make_surface_stub()
        surface.steel_models = [
            SimpleNamespace(get_ultimate_strain=lambda: float("inf"), epsilon_y=0.001),
            SimpleNamespace(get_ultimate_strain=lambda: 0.05, epsilon_y=0.002),
        ]
        assert surface._eps_tension_limit() == pytest.approx(0.05, rel=1e-12)

        surface_all_inf = _make_surface_stub()
        surface_all_inf.steel_models = [
            SimpleNamespace(get_ultimate_strain=lambda: float("inf"), epsilon_y=0.0003),
            SimpleNamespace(get_ultimate_strain=lambda: float("inf"), epsilon_y=0.0008),
        ]
        # max(10*eps_y_max, 0.01) => max(0.008, 0.01) = 0.01
        assert surface_all_inf._eps_tension_limit() == pytest.approx(0.01, rel=1e-12)

    def test_concrete_stress_with_options_confined_and_tension_stiffening(self):
        """Test concrete stress with options confined and tension stiffening."""
        surface = _make_surface_stub()
        surface.concrete_model = SimpleNamespace(get_stress_array=lambda arr: np.array(arr, dtype=float) * 0.0)
        surface.concrete = SimpleNamespace(
            f_ck=30.0,
            epsilon_c2=0.002,
            E_cm=33_000.0,
            alpha_cc=1.0,
            gamma_c=1.5,
            f_ctm=2.9,
        )
        surface.confined_concrete = True
        surface.confinement_rho_s = 0.01
        surface.confinement_f_yh = 500.0
        surface.tension_stiffening = True

        strains = np.array([0.001, 0.002, -0.00001, -0.001], dtype=float)
        out = surface._concrete_stress_with_options(strains)

        assert out.shape == strains.shape
        # Compression strains should remain non-negative with confinement model.
        assert out[0] >= 0.0
        assert out[1] >= 0.0
        # Tension strains should be non-positive with tension stiffening model.
        assert out[2] <= 0.0
        assert out[3] <= 0.0

    def test_concrete_stress_with_options_confined_handles_near_zero_denominator(self):
        """Test concrete stress with options confined handles near zero denominator."""
        surface = _make_surface_stub()
        surface.concrete_model = SimpleNamespace(get_stress_array=lambda arr: np.zeros_like(arr, dtype=float))
        surface.confined_concrete = True
        surface.confinement_rho_s = 0.01
        surface.confinement_f_yh = 500.0
        surface.tension_stiffening = False

        f_co_k = 30.0
        eps_co = 0.002
        rho_s = 0.01
        f_yh_k = 500.0
        k_e = 0.75
        f_l_k = 0.5 * k_e * rho_s * f_yh_k
        term = 1.0 + 7.94 * f_l_k / f_co_k
        f_cc_k = f_co_k * (2.254 * np.sqrt(term) - 2.0 * f_l_k / f_co_k - 1.254)
        f_ratio = max(f_cc_k / f_co_k, 1e-6)
        eps_cc = max(eps_co * (1.0 + 5.0 * (f_ratio - 1.0)), 1e-9)
        e_match = f_cc_k / eps_cc

        surface.concrete = SimpleNamespace(
            f_ck=f_co_k,
            epsilon_c2=eps_co,
            E_cm=e_match,  # makes denom ~= 0 to hit protective branch
            alpha_cc=1.0,
            gamma_c=1.5,
            f_ctm=2.9,
        )

        out = surface._concrete_stress_with_options(np.array([0.001], dtype=float))
        assert out.shape == (1,)

    def test_build_convex_hull_and_hull_cache(self, monkeypatch):
        """Test build convex hull and hull cache."""
        surface = _make_surface_stub()
        points = (
            _pt(0.0, 0.0, 0.0),
            _pt(1.0, 0.0, 0.0),
            _pt(0.0, 1.0, 0.0),
            _pt(0.0, 0.0, 1.0),
        )
        class _FakeHull:
            def __init__(self, pts):
                self.points = pts

        monkeypatch.setattr(biax, "ConvexHull", _FakeHull)

        hull = surface._build_convex_hull(points)
        assert hull.points.shape[0] == 4

        with pytest.raises(ValueError, match="At least 4 points"):
            surface._build_convex_hull(points[:3])

        calls = {"gen": 0, "build": 0}
        monkeypatch.setattr(
            surface,
            "generate_surface_pivot",
            lambda n_angles, n_axial_levels: (calls.__setitem__("gen", calls["gen"] + 1) or points),
        )

        def _fake_build(pts):
            calls["build"] += 1
            return "HULL"

        monkeypatch.setattr(surface, "_build_convex_hull", _fake_build)

        h1 = surface._get_hull(10, 20)
        h2 = surface._get_hull(10, 20)
        assert h1 == "HULL"
        assert h2 == "HULL"
        assert calls["gen"] == 1
        assert calls["build"] == 1

    def test_capacity_vector_exact_branches_and_wrappers(self, monkeypatch):
        """Test capacity vector exact branches and wrappers."""
        surface = _make_surface_stub()

        # Origin special case.
        assert surface.get_capacity_vector_exact(0.0, 0.0, 0.0) == (0.0, 0.0, 0.0, True, 0.0)

        monkeypatch.setattr(surface, "_get_hull", lambda n_angles, n_axial_levels: (_ for _ in ()).throw(RuntimeError("boom")))
        fail = surface.get_capacity_vector_exact(1.0, 0.0, 0.0)
        assert fail[3] is False
        assert np.isinf(fail[4])

        # No forward-facing facet.
        hull_no_forward = SimpleNamespace(equations=np.array([[-1.0, 0.0, 0.0, -1.0]]))
        no_forward = surface.get_capacity_vector_exact(1.0, 0.0, 0.0, hull=hull_no_forward)
        assert no_forward[3] is False
        assert np.isinf(no_forward[4])

        # Surface points path uses _build_convex_hull.
        calls = {"built": False}
        def _fake_build(points):
            calls["built"] = True
            return SimpleNamespace(equations=np.array([[1.0, 0.0, 0.0, -2.0]]))
        monkeypatch.setattr(surface, "_build_convex_hull", _fake_build)
        via_points = surface.get_capacity_vector_exact(1.0, 0.0, 0.0, surface_points=[_pt(0, 0, 0)] * 4)
        assert calls["built"] is True
        assert via_points[0] == pytest.approx(2.0, rel=1e-12)

        # Forward facet but no positive t candidate.
        hull_no_t = SimpleNamespace(equations=np.array([[1.0, 0.0, 0.0, 1.0]]))
        no_t = surface.get_capacity_vector_exact(1.0, 0.0, 0.0, hull=hull_no_t)
        assert no_t[3] is False
        assert np.isinf(no_t[4])

        # Valid intersection.
        hull_ok = SimpleNamespace(equations=np.array([[1.0, 0.0, 0.0, -2.0], [0.0, -1.0, 0.0, -1.0]]))
        ok = surface.get_capacity_vector_exact(1.0, 0.0, 0.0, hull=hull_ok)
        assert ok[0] == pytest.approx(2.0, rel=1e-12)
        assert ok[1] == pytest.approx(0.0, rel=1e-12)
        assert ok[2] == pytest.approx(0.0, rel=1e-12)
        assert ok[3] is True
        assert ok[4] == pytest.approx(0.5, rel=1e-12)

        # load_mag near-zero guard branch.
        monkeypatch.setattr(biax.np.linalg, "norm", lambda _vec: 0.0)
        near_zero_mag = surface.get_capacity_vector_exact(1.0, 0.0, 0.0, hull=hull_ok)
        assert near_zero_mag == (0.0, 0.0, 0.0, True, 0.0)

        monkeypatch.setattr(surface, "get_capacity_vector_exact", lambda **kwargs: (1.0, 2.0, 3.0, False, 1.25))
        is_safe, util = surface.get_utilization_vector(1.0, 2.0, 3.0)
        assert is_safe is False
        assert util == pytest.approx(1.25, rel=1e-12)
        assert surface.get_capacity_vector(1.0, 2.0, 3.0) == (1.0, 2.0, 3.0, False, 1.25)

    def test_calculate_point_pivot_no_steel_and_ignore_compression_steel(self):
        # No-steel branch uses rebar_y_min = y_min.
        """Test calculate point pivot no steel and ignore compression steel."""
        s_no_steel = _make_surface_stub()
        s_no_steel.section_centroid_x = 0.0
        s_no_steel.section_centroid_y = 0.0
        s_no_steel._fibre_x = np.array([0.0], dtype=float)
        s_no_steel._fibre_y = np.array([1.0], dtype=float)
        s_no_steel._fibre_area = np.array([1.0], dtype=float)
        s_no_steel._fibre_mat = np.array(["concrete"])
        s_no_steel._fibre_mi = np.array([0], dtype=int)
        s_no_steel.concrete = SimpleNamespace(epsilon_cu2=0.0035, epsilon_c2=0.002)
        s_no_steel._eps_tension_limit = lambda: 0.01
        s_no_steel._concrete_stress_with_options = lambda arr: np.ones_like(arr) * 10.0
        s_no_steel.steel_models = []
        s_no_steel.ignore_compression_steel = False
        p_no_steel = s_no_steel.calculate_point_pivot(na_depth=3.0, neutral_axis_angle=0.0)
        assert isinstance(p_no_steel, BiaxialInteractionPoint)
        assert p_no_steel.max_steel_strain == pytest.approx(0.0, abs=1e-12)

        # Compression-steel suppression branch.
        s_steel = _make_surface_stub()
        s_steel.section_centroid_x = 0.0
        s_steel.section_centroid_y = 0.0
        s_steel._fibre_x = np.array([0.0], dtype=float)
        s_steel._fibre_y = np.array([-1.0], dtype=float)
        s_steel._fibre_area = np.array([1.0], dtype=float)
        s_steel._fibre_mat = np.array(["steel"])
        s_steel._fibre_mi = np.array([0], dtype=int)
        s_steel.concrete = SimpleNamespace(epsilon_cu2=0.0035, epsilon_c2=0.002)
        s_steel._eps_tension_limit = lambda: 0.01
        s_steel._concrete_stress_with_options = lambda arr: np.zeros_like(arr)
        s_steel.steel_models = [SimpleNamespace(get_stress_array=lambda arr: np.full_like(arr, 100.0))]
        s_steel.ignore_compression_steel = True
        p_steel = s_steel.calculate_point_pivot(na_depth=3.0, neutral_axis_angle=0.0)
        assert p_steel.max_steel_strain > 0.0
        assert p_steel.N == pytest.approx(0.0, abs=1e-12)

    def test_get_strain_at_y_pivot_zones(self):
        """Test get strain at y pivot zones."""
        surface = _make_surface_stub()
        surface.concrete = SimpleNamespace(epsilon_cu2=0.0035, epsilon_c2=0.002)
        surface._eps_tension_limit = lambda: 0.01

        # Zone A (na_depth <= x_bal)
        eps_a = surface._get_strain_at_y_pivot(
            y=0.2,
            na_depth=20.0,
            y_max=1.0,
            y_min=0.0,
            h=1.0,
            rebar_y_min=0.0,
            d_eff=100.0,
        )
        # Zone B
        eps_b = surface._get_strain_at_y_pivot(
            y=0.2,
            na_depth=50.0,
            y_max=1.0,
            y_min=0.0,
            h=100.0,
            rebar_y_min=0.0,
            d_eff=100.0,
        )
        # Zone C
        eps_c = surface._get_strain_at_y_pivot(
            y=0.2,
            na_depth=150.0,
            y_max=1.0,
            y_min=0.0,
            h=100.0,
            rebar_y_min=0.0,
            d_eff=100.0,
        )
        assert isinstance(eps_a, float)
        assert isinstance(eps_b, float)
        assert isinstance(eps_c, float)


class TestSurfaceGenerationAndPlotting:
    """Tests for TestSurfaceGenerationAndPlotting."""
    def test_dense_cache_downsample_generate_and_prepare_matrices(self, monkeypatch):
        """Test dense cache downsample generate and prepare matrices."""
        surface = _make_surface_stub()
        surface._hull_cache = {(1, 1): "old"}

        calls = {"raw": 0}
        monkeypatch.setattr(
            surface,
            "_generate_surface_raw",
            lambda n_angles, n_axial_levels: (
                calls.__setitem__("raw", calls["raw"] + 1) or (_pt(1, 2, 3), _pt(4, 5, 6))
            ),
        )

        first = surface._get_dense_surface_points(8, 10)
        second = surface._get_dense_surface_points(8, 10)
        assert first == second
        assert calls["raw"] == 1
        assert surface._surface_cache == {}
        assert surface._hull_cache == {}

        dense = tuple(_pt(float(i), float(i), float(i)) for i in range(12))
        # No downsampling needed: points returned as-is with full-grid indices
        same, same_idx = BiaxialMNInteractionSurface._downsample_surface(dense, 4, 3, 4, 3)
        assert same == dense
        assert same_idx == [divmod(k, 4) for k in range(12)]
        # Sparse grid with position metadata: selected via dense indices,
        # returned with OUTPUT-grid indices (the crash fix)
        sparse_idx = [divmod(k, 4) for k in range(5)]  # dense cells (0,0)..(1,0)
        down_sparse, down_sparse_idx = BiaxialMNInteractionSurface._downsample_surface(
            dense[:5], 4, 3, 2, 2, dense_grid_indices=sparse_idx
        )
        # Output grid selects dense cells (0,0), (0,2), (1,0), (1,2); the
        # sparse set contains (0,0), (0,2) and (1,0) but not (1,2)
        assert down_sparse == (dense[0], dense[2], dense[4])
        assert down_sparse_idx == [(0, 0), (0, 1), (1, 0)]
        assert all(i < 2 and j < 2 for i, j in down_sparse_idx)
        # Sparse grid WITHOUT position metadata: best-effort passthrough
        down_nometa, _ = BiaxialMNInteractionSurface._downsample_surface(dense[:5], 4, 3, 2, 2)
        assert down_nometa == dense[:5]
        down, down_idx = BiaxialMNInteractionSurface._downsample_surface(dense, 4, 3, 2, 2)
        assert len(down) == 4
        assert down_idx == [(0, 0), (0, 1), (1, 0), (1, 1)]

        # generate_surface_pivot cache hit
        surface_cache = _make_surface_stub()
        surface_cache._surface_cache = {(1, 1): (_pt(0, 0, 0),)}
        cached = surface_cache.generate_surface_pivot(n_angles=1, n_axial_levels=1)
        assert cached == (_pt(0, 0, 0),)

        # generate_surface_pivot default dense params + downsample path
        surface2 = _make_surface_stub()
        seen = {}
        monkeypatch.setattr(
            surface2,
            "_get_dense_surface_points",
            lambda n_dense_angles, n_dense_axial: (
                seen.update({"a": n_dense_angles, "n": n_dense_axial}) or dense
            ),
        )
        monkeypatch.setattr(
            surface2,
            "_downsample_surface",
            lambda **kwargs: (tuple([_pt(9, 9, 9)]), [(0, 0)]),
        )
        out = surface2.generate_surface_pivot(n_angles=12, n_axial_levels=10)
        assert out == (_pt(9, 9, 9),)
        assert seen["a"] == 144
        assert seen["n"] == 80

        # _prepare_surface_matrices
        surface3 = _make_surface_stub()
        surface3.section_width = 300.0
        surface3.section_height = 500.0
        surface3.calculate_axial_limits = lambda: (-10.0, 10.0)
        surface3.calculate_point_pivot = lambda na_depth, ang: _pt(
            N=na_depth,
            My=na_depth + ang,
            Mz=na_depth - ang,
            depth=na_depth,
            angle=ang,
        )
        surface_pts = tuple(
            _pt(N=float(i), My=float(i + 10), Mz=float(i + 20))
            for i in range(6)
        )  # 2 axial x 3 angles

        My, Mz, N = surface3._prepare_surface_matrices(surface_pts, n_axial_levels=2, n_angles=3)
        assert My.shape == (4, 4)
        assert Mz.shape == (4, 4)
        assert N.shape == (4, 4)
        # Longitude loop closure
        assert np.allclose(My[1:3, 0], My[1:3, -1])
        assert np.allclose(Mz[1:3, 0], Mz[1:3, -1])
        assert np.allclose(N[1:3, 0], N[1:3, -1])

    def test_generate_surface_raw_branches_and_plot(self, monkeypatch, tmp_path: Path):
        """Test generate surface raw branches and plot."""
        surface = _make_surface_stub()
        surface.section_width = 1.0
        surface.section_height = 1.0
        surface.calculate_axial_limits = lambda: (0.0, 1.0)
        surface.calculate_point_pivot = lambda na_depth, angle: _pt(N=na_depth, My=angle, Mz=2.0 * na_depth, depth=na_depth, angle=angle)
        monkeypatch.setattr(biax, "brentq", lambda f, a, b, xtol=1e-5: 0.0)
        pts = surface._generate_surface_raw(n_angles=1, n_axial_levels=1)
        assert len(pts) == 1
        assert pts[0].N == pytest.approx(0.0, rel=1e-12)

        # When brentq cannot bracket (N always returns constant), points are skipped
        # (no non-physical fallback). This is correct behaviour — a missing point is
        # better than a point with fabricated N.
        surface2 = _make_surface_stub()
        surface2.section_width = 1.0
        surface2.section_height = 1.0
        surface2.calculate_axial_limits = lambda: (-1.0, 1.0)
        surface2.calculate_point_pivot = lambda na_depth, angle: _pt(N=5.0, My=na_depth, Mz=0.0, depth=na_depth, angle=angle)
        pts2 = surface2._generate_surface_raw(n_angles=1, n_axial_levels=2)
        assert len(pts2) == 0  # Both points fail to bracket, so no points are generated

        surface3 = _make_surface_stub()
        surface3.section_width = 1.0
        surface3.section_height = 1.0
        surface3.calculate_axial_limits = lambda: (0.0, 1.0)
        surface3.calculate_point_pivot = lambda na_depth, angle: (_ for _ in ()).throw(RuntimeError("bad"))
        pts3 = surface3._generate_surface_raw(n_angles=1, n_axial_levels=1)
        assert pts3 == ()

        # Expanded tangent bracket branch that breaks on first expanded bound.
        surface_break = _make_surface_stub()
        surface_break.section_width = 1.0
        surface_break.section_height = 1.0
        surface_break.calculate_axial_limits = lambda: (0.0, 2.0)
        surface_break.calculate_point_pivot = lambda na_depth, angle: _pt(
            N=(-1.0 if na_depth > 20.0 else 1.0),
            My=na_depth,
            Mz=0.0,
            depth=na_depth,
            angle=angle,
        )
        monkeypatch.setattr(biax, "brentq", lambda func, a, b, xtol=1e-5: 0.0)
        pts_break = surface_break._generate_surface_raw(n_angles=1, n_axial_levels=1)
        assert len(pts_break) == 1

        # Plot branch coverage with fake plotly.
        _install_fake_plotly(monkeypatch)
        surface_plot = _make_surface_stub()
        surface_plot.generate_surface_pivot = lambda n_angles, n_axial_levels: (_pt(0, 0, 0),)
        surface_plot._prepare_surface_matrices = lambda surface_pts, n_axial_levels, n_angles: (
            np.array([[1.0]]),
            np.array([[2.0]]),
            np.array([[3.0]]),
        )

        cap = {
            100.0: (10.0, 20.0, 30.0, True, 0.7),
            150.0: (15.0, 25.0, 35.0, True, 0.9),
            200.0: (None, None, None, False, 1.2),
        }
        surface_plot.get_capacity_vector = lambda **kwargs: cap[float(kwargs["N_Ed"])]

        fig = surface_plot.plot(
            load_points=[
                {"N_Ed": 100.0, "My_Ed": 1.0, "Mz_Ed": 2.0, "name": "LC1"},
                {"N_Ed": 150.0, "My_Ed": 2.0, "Mz_Ed": 3.0, "name": "LCM"},
                {"N_Ed": 200.0, "My_Ed": 3.0, "Mz_Ed": 4.0, "name": "LC2"},
            ],
            show_vectors=True,
            show_metadata=True,
            n_angles=4,
            n_axial_levels=3,
            save_path=str(tmp_path / "biax.html"),
            show=False,
            title="Custom",
        )

        assert isinstance(fig, _FakeFigure)
        # Surface + origin + 3 markers + 4 vectors (for LC1 and LCM)
        assert len(fig.traces) == 9
        marker_lc1 = [t for t in fig.traces if t.get("name") == "LC1"][0]
        marker_lcm = [t for t in fig.traces if t.get("name") == "LCM"][0]
        marker_lc2 = [t for t in fig.traces if t.get("name") == "LC2"][0]
        assert marker_lc1["marker"]["color"] == "green"
        assert marker_lcm["marker"]["color"] == "orange"
        assert marker_lc2["marker"]["color"] == "red"
        assert "Status: Outside boundary" in marker_lc2["hovertemplate"]
        assert fig.layout_updates[-1]["title"]["text"] == "Custom"
        assert fig.saved_paths

        fig2 = surface_plot.plot(
            load_points=[{"N_Ed": 100.0, "My_Ed": 1.0, "Mz_Ed": 2.0, "name": "LC3"}],
            show_vectors=False,
            show_metadata=False,
            show=True,
        )
        marker_lc3 = [t for t in fig2.traces if t.get("name") == "LC3"][0]
        assert marker_lc3["hovertemplate"] == "LC3<extra></extra>"
        assert fig2.shown is True

    def test_plot_raises_import_error_when_plotly_is_missing(self, monkeypatch):
        """Test plot raises import error when plotly is missing."""
        surface = _make_surface_stub()
        real_import = builtins.__import__

        def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "plotly.graph_objects":
                raise ImportError("missing")
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", _fake_import)
        with pytest.raises(ImportError, match="Plotly is required for plotting"):
            surface.plot(show=False)

    def test_surface_repr(self):
        """Test surface repr."""
        surface = _make_surface_stub()
        surface.section = SimpleNamespace(section_name="S1")
        surface.concrete = SimpleNamespace(grade="C30/37")
        text = repr(surface)
        assert "BiaxialMNInteractionSurface(" in text
        assert "section=S1" in text
        assert "concrete=C30/37" in text
