"""
Tests for shear viewer plotting helpers.
"""

from __future__ import annotations

import builtins
import sys
import types
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from materials.reinforced_concrete.analysis.shear_viewer import ShearViewer


@dataclass(frozen=True)
class _FakeRebar:
    area_per_unit_length: float = 1.0
    angle: float = 90.0
    f_yk: float = 500.0
    f_yd: float = 435.0

    def model_copy(self, *, update: dict | None = None):
        data = {
            "area_per_unit_length": self.area_per_unit_length,
            "angle": self.angle,
            "f_yk": self.f_yk,
            "f_yd": self.f_yd,
        }
        if update:
            data.update(update)
        return _FakeRebar(**data)


class _FakeDiagram:
    def find_strains_for_MN(self, M_Ed: float, N_Ed: float, strict: bool = False):
        return (0.001, -0.001)


class _FakeCheck:
    def __init__(self, *, with_rebar: bool = True):
        self.shear_reinforcement = _FakeRebar() if with_rebar else None
        self.use_sigma_cp_for_alpha_cw = False
        self.concrete = SimpleNamespace(f_ck=30.0)
        self.breadth = 300.0
        self.f_cd_design = 20.0
        self.f_ywd_design = 435.0

    def _get_diagram(self, ignore_compression_steel: bool = False):
        return _FakeDiagram()

    def find_effective_depth(
        self,
        M_Ed: float,
        N_Ed: float,
        eps_top=None,
        eps_bottom=None,
        ignore_compression_steel: bool = False,
    ) -> float:
        return 500.0

    def _find_sigma_cp(self, N_Ed: float) -> float:
        return float(N_Ed) / 1000.0

    def _find_rho_l(
        self,
        M_Ed: float,
        N_Ed: float,
        d: float,
        eps_top=None,
        eps_bottom=None,
        ignore_compression_steel: bool = False,
    ) -> float:
        return 0.01

    def find_V_Rd_c(self, d: float, rho_l: float, sigma_cp: float) -> float:
        return 120.0 + 5.0 * sigma_cp

    def find_V_Rd_c_uncracked(self, sigma_cp: float) -> float:
        return 140.0 + 5.0 * sigma_cp

    def find_lever_arm(
        self,
        M_Ed: float,
        N_Ed: float,
        d: float,
        eps_top=None,
        eps_bottom=None,
        ignore_compression_steel: bool = False,
        force_virtual: bool = False,
    ):
        return (450.0, None)

    def _find_cot_theta_limits(self, sigma_cp: float, z: float, V_Ed: float):
        return (1.0, 2.5)

    def _find_cot_theta_for_V_Ed(self, **kwargs):
        return 1.7

    def find_V_Rd_s(self, cot_theta: float, z: float, use_note_2: bool = False) -> float:
        return 170.0 + 30.0 * cot_theta

    def find_V_Rd_max(self, cot_theta: float, z: float, sigma_cp: float, use_note_2: bool = False) -> float:
        return 260.0 - 10.0 * (cot_theta - 1.2) ** 2 + 2.0 * sigma_cp


class _FakeFigure:
    def __init__(self):
        self.traces = []
        self.layout_updates = []
        self.shown = False
        self.saved_path = None

    def add_trace(self, trace, **kwargs):
        self.traces.append((trace, kwargs))

    def update_layout(self, **kwargs):
        self.layout_updates.append(kwargs)

    def update_xaxes(self, **kwargs):
        pass

    def update_yaxes(self, **kwargs):
        pass

    def show(self):
        self.shown = True

    def write_html(self, path: str):
        self.saved_path = path


def _install_fake_plotly(monkeypatch):
    go_mod = types.ModuleType("plotly.graph_objects")
    go_mod.Figure = _FakeFigure
    go_mod.Scatter = lambda **kwargs: {"type": "Scatter", **kwargs}
    go_mod.Heatmap = lambda **kwargs: {"type": "Heatmap", **kwargs}
    go_mod.Contour = lambda **kwargs: {"type": "Contour", **kwargs}

    subplots_mod = types.ModuleType("plotly.subplots")
    subplots_mod.make_subplots = lambda **kwargs: _FakeFigure()

    plotly_mod = types.ModuleType("plotly")
    plotly_mod.graph_objects = go_mod
    plotly_mod.subplots = subplots_mod

    monkeypatch.setitem(sys.modules, "plotly", plotly_mod)
    monkeypatch.setitem(sys.modules, "plotly.graph_objects", go_mod)
    monkeypatch.setitem(sys.modules, "plotly.subplots", subplots_mod)


class TestShearViewer:
    """Tests for ShearViewer plotting API."""

    def test_plot_cot_theta_study_builds_traces(self, monkeypatch):
        _install_fake_plotly(monkeypatch)
        viewer = ShearViewer(_FakeCheck())

        fig = viewer.plot_cot_theta_study(
            load_case={"V_Ed": 150.0, "M_Ed": 10.0, "N_Ed": 50.0},
            n_points=6,
            show=False,
        )

        assert isinstance(fig, _FakeFigure)
        trace_types = [t[0]["type"] for t in fig.traces]
        assert trace_types.count("Scatter") >= 6
        trace_names = [t[0].get("name") for t in fig.traces if t[0]["type"] == "Scatter"]
        assert "V_Rd,max design" in trace_names
        assert "V_Rd,s design" in trace_names
        assert "V_Rd,s = V_Rd,max" in trace_names
        assert fig.shown is False

    def test_plot_cot_theta_moment_shift_study_builds_traces(self, monkeypatch):
        _install_fake_plotly(monkeypatch)
        viewer = ShearViewer(_FakeCheck())

        fig = viewer.plot_cot_theta_moment_shift_study(
            load_case={"V_Ed": 150.0, "M_Ed": 10.0, "N_Ed": 50.0},
            n_points=6,
            show=False,
        )

        assert isinstance(fig, _FakeFigure)
        trace_types = [t[0]["type"] for t in fig.traces]
        assert trace_types.count("Scatter") >= 3

    def test_plot_cot_theta_moment_shift_study_adds_utilization_intercept_trace(self, monkeypatch):
        _install_fake_plotly(monkeypatch)
        viewer = ShearViewer(_FakeCheck())

        fig = viewer.plot_cot_theta_moment_shift_study(
            load_case={"V_Ed": 220.0, "M_Ed": 10.0, "N_Ed": 50.0},
            n_points=12,
            show=False,
        )

        scatter_traces = [t[0] for t in fig.traces if t[0]["type"] == "Scatter"]
        names = [t.get("name") for t in scatter_traces]
        assert "Utilization = 1.0 intercept" in names

        intercept_trace = next(t for t in scatter_traces if t.get("name") == "Utilization = 1.0 intercept")
        assert "cot(theta)" in intercept_trace.get("hovertemplate", "")
        assert "M_add at util=1.0" in intercept_trace.get("hovertemplate", "")

    def test_plot_link_angle_study_builds_traces(self, monkeypatch):
        _install_fake_plotly(monkeypatch)
        viewer = ShearViewer(_FakeCheck())

        fig = viewer.plot_link_angle_study(
            load_case={"V_Ed": 150.0, "M_Ed": 10.0, "N_Ed": 50.0},
            n_points=6,
            show=False,
        )

        assert isinstance(fig, _FakeFigure)
        trace_types = [t[0]["type"] for t in fig.traces]
        assert trace_types.count("Scatter") >= 4
        trace_names = [t[0].get("name") for t in fig.traces if t[0]["type"] == "Scatter"]
        assert "V_Rd,s(alpha)" in trace_names
        assert "V_Rd,max(alpha)" in trace_names

    def test_plot_link_angle_moment_shift_study_builds_traces(self, monkeypatch):
        _install_fake_plotly(monkeypatch)
        viewer = ShearViewer(_FakeCheck())

        fig = viewer.plot_link_angle_moment_shift_study(
            load_case={"V_Ed": 150.0, "M_Ed": 10.0, "N_Ed": 50.0},
            n_points=6,
            show=False,
        )

        assert isinstance(fig, _FakeFigure)
        trace_types = [t[0]["type"] for t in fig.traces]
        assert trace_types.count("Scatter") >= 3

    def test_plot_cot_theta_link_angle_heatmap_has_heatmap_and_contour(self, monkeypatch):
        _install_fake_plotly(monkeypatch)
        viewer = ShearViewer(_FakeCheck())

        fig = viewer.plot_cot_theta_link_angle_heatmap(
            load_case={"V_Ed": 150.0, "M_Ed": 10.0, "N_Ed": 50.0},
            n_cot=4,
            n_angles=3,
            metric="utilization",
            show=False,
        )

        trace_types = [t[0]["type"] for t in fig.traces]
        assert "Heatmap" in trace_types
        assert "Contour" in trace_types

    def test_plot_axial_cot_theta_contour_has_heatmap_and_contour(self, monkeypatch):
        _install_fake_plotly(monkeypatch)
        viewer = ShearViewer(_FakeCheck())

        fig = viewer.plot_axial_cot_theta_contour(
            load_case={"V_Ed": 150.0, "M_Ed": 10.0, "N_Ed": 50.0},
            N_min=-200.0,
            N_max=200.0,
            n_axial=4,
            n_cot=4,
            metric="utilization",
            show=False,
        )

        trace_types = [t[0]["type"] for t in fig.traces]
        assert "Heatmap" in trace_types
        assert "Contour" in trace_types

    def test_plot_methods_require_shear_reinforcement(self):
        viewer = ShearViewer(_FakeCheck(with_rebar=False))
        with pytest.raises(ValueError, match="Shear reinforcement is required"):
            viewer.plot_cot_theta_study(
                load_case={"V_Ed": 150.0, "M_Ed": 10.0, "N_Ed": 50.0},
                show=False,
            )

    def test_plotly_import_error_branch(self, monkeypatch):
        real_import = builtins.__import__

        def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name in {"plotly.graph_objects", "plotly.subplots"}:
                raise ImportError("plotly missing")
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.delitem(sys.modules, "plotly", raising=False)
        monkeypatch.delitem(sys.modules, "plotly.graph_objects", raising=False)
        monkeypatch.delitem(sys.modules, "plotly.subplots", raising=False)
        monkeypatch.setattr(builtins, "__import__", _fake_import)

        viewer = ShearViewer(_FakeCheck())
        with pytest.raises(ImportError, match="Plotly is required"):
            viewer.plot_cot_theta_study(
                load_case={"V_Ed": 150.0, "M_Ed": 10.0, "N_Ed": 50.0},
                show=False,
            )
