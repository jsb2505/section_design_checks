"""
Tests for crack width visualization helpers and plotting wrappers.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass

import numpy as np
import pytest

from materials.reinforced_concrete.analysis.crack_width_viewer import (
    CrackWidthViewer,
    _compute_load_case_result,
    _eval_w_k,
    _find_crack_width_boundary,
    _get_domain_bounds,
)
from materials.reinforced_concrete.code_checks.ec2_2004.cracking_check import CrackingResult
from materials.reinforced_concrete.constitutive import ConcreteModelType


@dataclass
class _Point:
    M: float
    N: float


class _FakeDiagram:
    def __init__(self, points):
        self._points = points

    def generate_diagram_points(self, n_points: int):
        return self._points


class _FakeCheck:
    def __init__(self, *, w_k_limit: float = 0.3):
        self.w_k_limit = w_k_limit
        self.section = object()
        self.concrete = object()

    def calculate_detailed(self, *, M_Ed: float, N_Ed: float, force_cracked: bool = False):
        # simple deterministic shape
        w_k = abs(M_Ed) / 100.0 + abs(N_Ed) / 1000.0
        return CrackingResult(
            w_k=w_k,
            w_k_limit=self.w_k_limit,
            s_r_max=100.0,
            eps_sm_minus_eps_cm=0.001,
            sigma_s=200.0,
            rho_p_eff=0.01,
            h_c_ef=120.0,
            x=200.0,
            is_cracked=w_k > 0.0,
            phi_eq=16.0,
            cover=40.0,
        )


class _FakeFigure:
    def __init__(self):
        self.traces = []
        self.layout_updates = []
        self.shown = False
        self.saved_path = None

    def add_trace(self, trace):
        self.traces.append(trace)

    def update_layout(self, **kwargs):
        self.layout_updates.append(kwargs)

    def show(self):
        self.shown = True

    def write_html(self, path):
        self.saved_path = path


def _install_fake_plotly(monkeypatch):
    go_mod = types.ModuleType("plotly.graph_objects")
    go_mod.Figure = _FakeFigure
    go_mod.Mesh3d = lambda **kwargs: {"type": "Mesh3d", **kwargs}
    go_mod.Scatter3d = lambda **kwargs: {"type": "Scatter3d", **kwargs}
    go_mod.Contour = lambda **kwargs: {"type": "Contour", **kwargs}
    go_mod.Scatter = lambda **kwargs: {"type": "Scatter", **kwargs}

    plotly_mod = types.ModuleType("plotly")
    plotly_mod.graph_objects = go_mod

    monkeypatch.setitem(sys.modules, "plotly", plotly_mod)
    monkeypatch.setitem(sys.modules, "plotly.graph_objects", go_mod)


class TestCrackWidthViewerHelpers:
    def test_compute_load_case_result(self):
        check = _FakeCheck(w_k_limit=0.3)
        r = _compute_load_case_result(check, M_Ed=20.0, N_Ed=50.0, name="LC1")
        assert r.name == "LC1"
        assert r.M_Ed == pytest.approx(20.0, rel=1e-12)
        assert r.N_Ed == pytest.approx(50.0, rel=1e-12)
        assert r.w_k == pytest.approx(0.25, rel=1e-12)
        assert r.passes is True

    def test_get_domain_bounds_uses_generated_points(self, monkeypatch):
        points = [
            _Point(M=-120.0, N=-300.0),
            _Point(M=180.0, N=500.0),
            _Point(M=60.0, N=0.0),
        ]
        monkeypatch.setattr(
            "materials.reinforced_concrete.analysis.crack_width_viewer.create_interaction_diagram",
            lambda **kwargs: _FakeDiagram(points),
        )

        check = _FakeCheck()
        M_min, M_max, N_min, N_max = _get_domain_bounds(
            check=check,
            concrete_model_type=ConcreteModelType.PARABOLA_RECTANGLE,
            n_points=50,
        )
        assert (M_min, M_max, N_min, N_max) == pytest.approx((-120.0, 180.0, -300.0, 500.0), rel=1e-12)

    def test_eval_w_k_and_failure_to_nan(self):
        check = _FakeCheck()
        assert _eval_w_k(check, M=10.0, N=0.0, force_cracked=False) == pytest.approx(0.1, rel=1e-12)

        class _FailingCheck:
            def calculate_detailed(self, **kwargs):
                raise ValueError("fail")

        assert np.isnan(_eval_w_k(_FailingCheck(), M=10.0, N=0.0, force_cracked=False))

    def test_find_crack_width_boundary_finds_positive_and_negative_branches(self, monkeypatch):
        # w_k = |M|/100, limit 0.5 => |M| = 50 boundary
        monkeypatch.setattr(
            "materials.reinforced_concrete.analysis.crack_width_viewer._eval_w_k",
            lambda check, M, N, force_cracked: abs(M) / 100.0,
        )

        pos, neg = _find_crack_width_boundary(
            check=_FakeCheck(w_k_limit=0.5),
            N_values=np.array([-100.0, 0.0, 100.0]),
            M_min=-120.0,
            M_max=120.0,
            w_k_limit=0.5,
            force_cracked=True,
        )

        assert len(pos) == 3
        assert len(neg) == 3
        assert all(abs(m - 50.0) < 1.0 for m, _ in pos)
        assert all(abs(m + 50.0) < 1.0 for m, _ in neg)

    def test_find_crack_width_boundary_handles_nan_eval(self, monkeypatch):
        monkeypatch.setattr(
            "materials.reinforced_concrete.analysis.crack_width_viewer._eval_w_k",
            lambda *args, **kwargs: float("nan"),
        )
        pos, neg = _find_crack_width_boundary(
            check=_FakeCheck(w_k_limit=0.3),
            N_values=np.array([0.0]),
            M_min=-100.0,
            M_max=100.0,
            w_k_limit=0.3,
            force_cracked=True,
        )
        assert pos == []
        assert neg == []

    def test_find_crack_width_boundary_brentq_valueerror_is_ignored(self, monkeypatch):
        monkeypatch.setattr(
            "materials.reinforced_concrete.analysis.crack_width_viewer._eval_w_k",
            lambda check, M, N, force_cracked: abs(M) / 100.0,
        )
        monkeypatch.setattr(
            "scipy.optimize.brentq",
            lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("bracket fail")),
        )
        pos, neg = _find_crack_width_boundary(
            check=_FakeCheck(w_k_limit=0.5),
            N_values=np.array([0.0, 50.0]),
            M_min=-120.0,
            M_max=120.0,
            w_k_limit=0.5,
            force_cracked=True,
        )
        assert pos == []
        assert neg == []


class TestCrackWidthViewerPlots:
    def test_plot_load_cases_builds_3d_plot(self, monkeypatch):
        _install_fake_plotly(monkeypatch)
        viewer = CrackWidthViewer(_FakeCheck(w_k_limit=0.3))

        fig = viewer.plot_load_cases(
            load_cases=[
                {"M_Ed": 20.0, "N_Ed": 0.0, "name": "A"},
                {"M_Ed": 50.0, "N_Ed": 100.0, "name": "B"},
            ],
            show=False,
        )

        assert isinstance(fig, _FakeFigure)
        # 1 limit plane + 2 traces per load case (stem + marker)
        assert len(fig.traces) == 1 + 2 * 2
        assert fig.traces[0]["type"] == "Mesh3d"
        assert fig.shown is False

    def test_plot_load_cases_show_true_calls_show(self, monkeypatch):
        _install_fake_plotly(monkeypatch)
        viewer = CrackWidthViewer(_FakeCheck())

        fig = viewer.plot_load_cases(load_cases=[{"M_Ed": 10.0, "N_Ed": 0.0}], show=True)
        assert fig.shown is True

    def test_plot_load_cases_save_path_calls_write_html(self, monkeypatch, tmp_path):
        _install_fake_plotly(monkeypatch)
        viewer = CrackWidthViewer(_FakeCheck())
        save_path = tmp_path / "crack_cases.html"
        fig = viewer.plot_load_cases(
            load_cases=[{"M_Ed": 10.0, "N_Ed": 0.0}],
            show=False,
            save_path=save_path,
        )
        assert fig.saved_path == str(save_path)

    def test_plot_load_cases_import_error_branch(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def _fake_import(name, *args, **kwargs):
            if name == "plotly.graph_objects":
                raise ImportError("missing plotly")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _fake_import)
        viewer = CrackWidthViewer(_FakeCheck())
        with pytest.raises(ImportError, match="Plotly is required"):
            viewer.plot_load_cases(load_cases=[{"M_Ed": 10.0, "N_Ed": 0.0}], show=False)

    def test_plot_contours_builds_contour_and_boundary(self, monkeypatch):
        _install_fake_plotly(monkeypatch)
        viewer = CrackWidthViewer(_FakeCheck(w_k_limit=0.4))

        monkeypatch.setattr(
            "materials.reinforced_concrete.analysis.crack_width_viewer._get_domain_bounds",
            lambda check, concrete_model_type, n_points: (-100.0, 100.0, -200.0, 200.0),
        )
        monkeypatch.setattr(
            "materials.reinforced_concrete.analysis.crack_width_viewer._eval_w_k",
            lambda check, M, N, force_cracked: abs(M) / 200.0 + abs(N) / 1000.0,
        )
        monkeypatch.setattr(
            "materials.reinforced_concrete.analysis.crack_width_viewer._find_crack_width_boundary",
            lambda check, N_values, M_min, M_max, w_k_limit, force_cracked: (
                [(20.0, float(n)) for n in N_values[:3]],
                [(-20.0, float(n)) for n in N_values[:3]],
            ),
        )

        fig = viewer.plot_contours(
            load_cases=[{"M_Ed": 10.0, "N_Ed": 0.0, "name": "LC1"}],
            n_grid=5,
            n_boundary_points=5,
            show=False,
        )

        assert isinstance(fig, _FakeFigure)
        types_seen = [t["type"] for t in fig.traces]
        assert "Contour" in types_seen
        # 2 boundary curves + 1 load case marker
        assert types_seen.count("Scatter") >= 3

    def test_plot_contours_show_true_calls_show(self, monkeypatch):
        _install_fake_plotly(monkeypatch)
        viewer = CrackWidthViewer(_FakeCheck())

        monkeypatch.setattr(
            "materials.reinforced_concrete.analysis.crack_width_viewer._get_domain_bounds",
            lambda check, concrete_model_type, n_points: (-10.0, 10.0, -10.0, 10.0),
        )
        monkeypatch.setattr(
            "materials.reinforced_concrete.analysis.crack_width_viewer._eval_w_k",
            lambda check, M, N, force_cracked: 0.1,
        )
        monkeypatch.setattr(
            "materials.reinforced_concrete.analysis.crack_width_viewer._find_crack_width_boundary",
            lambda check, N_values, M_min, M_max, w_k_limit, force_cracked: ([], []),
        )

        fig = viewer.plot_contours(n_grid=3, n_boundary_points=3, show=True)
        assert fig.shown is True

    def test_plot_contours_save_path_calls_write_html(self, monkeypatch, tmp_path):
        _install_fake_plotly(monkeypatch)
        viewer = CrackWidthViewer(_FakeCheck())
        save_path = tmp_path / "crack_contours.html"

        monkeypatch.setattr(
            "materials.reinforced_concrete.analysis.crack_width_viewer._get_domain_bounds",
            lambda check, concrete_model_type, n_points: (-10.0, 10.0, -10.0, 10.0),
        )
        monkeypatch.setattr(
            "materials.reinforced_concrete.analysis.crack_width_viewer._eval_w_k",
            lambda check, M, N, force_cracked: 0.1,
        )
        monkeypatch.setattr(
            "materials.reinforced_concrete.analysis.crack_width_viewer._find_crack_width_boundary",
            lambda check, N_values, M_min, M_max, w_k_limit, force_cracked: ([], []),
        )

        fig = viewer.plot_contours(n_grid=3, n_boundary_points=3, show=False, save_path=save_path)
        assert fig.saved_path == str(save_path)

    def test_plot_contours_import_error_branch(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def _fake_import(name, *args, **kwargs):
            if name == "plotly.graph_objects":
                raise ImportError("missing plotly")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _fake_import)
        viewer = CrackWidthViewer(_FakeCheck())
        with pytest.raises(ImportError, match="Plotly is required"):
            viewer.plot_contours(n_grid=3, n_boundary_points=3, show=False)
