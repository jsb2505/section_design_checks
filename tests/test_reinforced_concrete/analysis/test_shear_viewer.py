"""
Tests for shear viewer plotting helpers.
"""

from __future__ import annotations

import builtins
import math
import sys
import types
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
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
    def __init__(self, *, m_limit: float = 100.0, n_limit: float = 200.0):
        self.m_limit = float(m_limit)
        self.n_limit = float(n_limit)

    def find_strains_for_MN(self, M_Ed: float, N_Ed: float, strict: bool = False):
        return (0.001, -0.001)

    def generate_diagram_points(self, n_points: int):
        return [
            SimpleNamespace(M=0.0, N=self.n_limit),
            SimpleNamespace(M=self.m_limit, N=0.0),
            SimpleNamespace(M=0.0, N=-self.n_limit),
            SimpleNamespace(M=-self.m_limit, N=0.0),
            SimpleNamespace(M=0.0, N=self.n_limit),
        ]

    def get_capacity_fixed_n(self, N_Ed: float, *, n_points: int = 120):
        n_cap = max(-self.n_limit, min(self.n_limit, float(N_Ed)))
        ratio = max(0.0, 1.0 - abs(n_cap) / self.n_limit)
        m_cap = self.m_limit * ratio
        return (n_cap, m_cap, -m_cap)

    def get_capacity_vector(self, N_Ed: float, M_Ed: float, n_points: int = 120, return_details: bool = False):
        _, m_pos, m_neg = self.get_capacity_fixed_n(N_Ed, n_points=n_points)
        is_safe = bool(m_neg <= float(M_Ed) <= m_pos)
        return SimpleNamespace(
            N_Rd=float(N_Ed),
            M_Rd=max(min(float(M_Ed), m_pos), m_neg),
            is_safe=is_safe,
            utilization=0.75 if is_safe else 1.25,
        )


class _FakeCheck:
    def __init__(self, *, with_rebar: bool = True, diagram: _FakeDiagram | None = None):
        self.shear_reinforcement = _FakeRebar() if with_rebar else None
        self.use_sigma_cp_for_alpha_cw = False
        self.concrete = SimpleNamespace(f_ck=30.0)
        self.breadth = 300.0
        self.f_cd_design = 20.0
        self.f_ywd_design = 435.0
        self._diagram = diagram or _FakeDiagram()

    def _get_diagram(self, ignore_compression_steel: bool = False):
        return self._diagram

    def find_effective_depth(
        self,
        My_Ed: float,
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
        My_Ed: float,
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
        My_Ed: float,
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
    go_mod.Isosurface = lambda **kwargs: {"type": "Isosurface", **kwargs}

    subplots_mod = types.ModuleType("plotly.subplots")
    subplots_mod.make_subplots = lambda **kwargs: _FakeFigure()

    plotly_mod = types.ModuleType("plotly")
    plotly_mod.graph_objects = go_mod
    plotly_mod.subplots = subplots_mod

    monkeypatch.setitem(sys.modules, "plotly", plotly_mod)
    monkeypatch.setitem(sys.modules, "plotly.graph_objects", go_mod)
    monkeypatch.setitem(sys.modules, "plotly.subplots", subplots_mod)


def _assert_slider_animation_controls_top_right(layout: dict) -> None:
    assert "updatemenus" in layout
    assert len(layout["updatemenus"]) == 1
    controls = layout["updatemenus"][0]
    assert controls["type"] == "buttons"
    assert controls["direction"] == "left"
    assert controls["x"] == 1.0
    assert controls["y"] == 1.16
    assert controls["xanchor"] == "right"
    assert controls["yanchor"] == "top"
    assert [button["label"] for button in controls["buttons"]] == ["Play", "Pause"]


def _assert_force_slider_has_nice_steps(layout: dict) -> None:
    slider_steps = layout["sliders"][0]["steps"]
    assert 30 <= len(slider_steps) <= 70

    labels = [step["label"] for step in slider_steps]
    assert all("." not in label for label in labels)
    values = [float(step["label"]) for step in slider_steps]
    assert all(np.isclose(value % 5.0, 0.0) for value in values)
    increments = [values[i + 1] - values[i] for i in range(len(values) - 1)]
    assert increments
    assert all(delta > 0.0 for delta in increments)

    main_step = round(increments[0], 6)
    assert all(abs(round(delta, 6) - main_step) <= 1e-6 for delta in increments)

    magnitude = 10.0 ** math.floor(math.log10(abs(main_step)))
    normalized = main_step / magnitude
    assert any(abs(normalized - candidate) <= 1e-6 for candidate in (1.0, 2.0, 5.0, 10.0))


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
        assert any(name in trace_names for name in ("V_Rd,s = V_Rd,max", "V_Ed,max"))
        assert "Cot(theta),min" not in trace_names
        assert "Cot(theta),max" not in trace_names
        assert fig.shown is False

    def test_plot_cot_theta_study_adds_cot_theta_min_intercept_trace(self, monkeypatch):
        _install_fake_plotly(monkeypatch)
        viewer = ShearViewer(_FakeCheck())

        fig = viewer.plot_cot_theta_study(
            load_case={"V_Ed": 220.0, "M_Ed": 10.0, "N_Ed": 50.0},
            n_points=12,
            show=False,
        )

        scatter_traces = [t[0] for t in fig.traces if t[0]["type"] == "Scatter"]
        names = [t.get("name") for t in scatter_traces]
        assert "Cot(theta),min" in names

        intercept_trace = next(t for t in scatter_traces if t.get("name") == "Cot(theta),min")
        assert len({float(x) for x in intercept_trace["x"]}) == 1
        assert min(intercept_trace["y"]) == pytest.approx(0.0)
        assert max(intercept_trace["y"]) == pytest.approx(220.0)
        assert "cot(theta)" in intercept_trace.get("hovertemplate", "")
        assert "V_Ed" in intercept_trace.get("hovertemplate", "")

    def test_plot_cot_theta_study_adds_cot_theta_max_intercept_trace(self, monkeypatch):
        _install_fake_plotly(monkeypatch)
        viewer = ShearViewer(_FakeCheck())

        fig = viewer.plot_cot_theta_study(
            load_case={"V_Ed": 250.0, "M_Ed": 10.0, "N_Ed": 50.0},
            n_points=12,
            show=False,
        )

        scatter_traces = [t[0] for t in fig.traces if t[0]["type"] == "Scatter"]
        names = [t.get("name") for t in scatter_traces]
        assert "Cot(theta),max" in names

        intercept_trace = next(t for t in scatter_traces if t.get("name") == "Cot(theta),max")
        assert len({float(x) for x in intercept_trace["x"]}) == 1
        assert min(intercept_trace["y"]) == pytest.approx(0.0)
        assert max(intercept_trace["y"]) == pytest.approx(250.0)
        assert "cot(theta)" in intercept_trace.get("hovertemplate", "")
        assert "V_Ed" in intercept_trace.get("hovertemplate", "")

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
        assert "M_add" in intercept_trace.get("hovertemplate", "")

    def test_plot_link_angle_study_builds_traces(self, monkeypatch):
        _install_fake_plotly(monkeypatch)
        viewer = ShearViewer(_FakeCheck())
        import materials.reinforced_concrete.analysis.shear_viewer as sv_mod

        monkeypatch.setattr(
            sv_mod,
            "get_ndp",
            lambda key: {
                "cot_theta_lower_lim": 1.0,
                "cot_theta_upper_lim": 2.5,
            }[key],
        )
        monkeypatch.setattr(viewer.check, "_find_cot_theta_limits", lambda sigma_cp, z, V_Ed: (1.4, 1.6))

        fig = viewer.plot_link_angle_study(
            load_case={"V_Ed": 150.0, "M_Ed": 10.0, "N_Ed": 50.0},
            n_cot=3,
            n_points=6,
            show=False,
        )

        assert isinstance(fig, _FakeFigure)
        trace_types = [t[0]["type"] for t in fig.traces]
        assert trace_types.count("Scatter") >= 4
        trace_names = [t[0].get("name") for t in fig.traces if t[0]["type"] == "Scatter"]
        assert "V_Rd,s(alpha)" in trace_names
        assert "V_Rd,max(alpha)" in trace_names
        assert hasattr(fig, "frames")
        assert len(fig.frames) == 31
        layout = fig.layout_updates[-1]
        assert "sliders" in layout
        assert layout["sliders"][0]["steps"][0]["label"] == "1.00"
        assert layout["sliders"][0]["steps"][-1]["label"] == "2.50"
        _assert_slider_animation_controls_top_right(layout)
        assert layout["yaxis"]["autorange"] is False

    def test_plot_link_angle_moment_shift_study_builds_traces(self, monkeypatch):
        _install_fake_plotly(monkeypatch)
        viewer = ShearViewer(_FakeCheck())
        import materials.reinforced_concrete.analysis.shear_viewer as sv_mod

        monkeypatch.setattr(
            sv_mod,
            "get_ndp",
            lambda key: {
                "cot_theta_lower_lim": 1.0,
                "cot_theta_upper_lim": 2.5,
            }[key],
        )
        monkeypatch.setattr(viewer.check, "_find_cot_theta_limits", lambda sigma_cp, z, V_Ed: (1.4, 1.6))

        fig = viewer.plot_link_angle_moment_shift_study(
            load_case={"V_Ed": 150.0, "M_Ed": 10.0, "N_Ed": 50.0},
            n_cot=3,
            n_points=6,
            show=False,
        )

        assert isinstance(fig, _FakeFigure)
        trace_types = [t[0]["type"] for t in fig.traces]
        assert trace_types.count("Scatter") >= 3
        assert hasattr(fig, "frames")
        assert len(fig.frames) == 31
        layout = fig.layout_updates[-1]
        assert "sliders" in layout
        assert layout["sliders"][0]["steps"][0]["label"] == "1.00"
        assert layout["sliders"][0]["steps"][-1]["label"] == "2.50"
        _assert_slider_animation_controls_top_right(layout)
        assert layout["yaxis"]["autorange"] is False
        assert layout["yaxis2"]["autorange"] is False

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

    def test_heatmap_zmax_stays_finite_with_noncomputable_cells(self, monkeypatch):
        """A V_Rd <= 0 cell becomes +inf; zmax must derive from finite cells only.

        Before the fix, np.nanmax did not ignore inf, so a single non-computable
        cell made zmax = inf and broke the colour scale / rendering.
        """
        import math

        _install_fake_plotly(monkeypatch)
        viewer = ShearViewer(_FakeCheck())
        # Force every cell non-computable (V_Rd = 0 -> utilization = +inf).
        monkeypatch.setattr(viewer, "_find_angle_sweep_capacity", lambda **kwargs: (0.0, 0.0))

        fig = viewer.plot_cot_theta_link_angle_heatmap(
            load_case={"V_Ed": 150.0, "M_Ed": 10.0, "N_Ed": 50.0},
            n_cot=4,
            n_angles=3,
            metric="utilization",
            show=False,
        )

        heatmap = next(t[0] for t in fig.traces if t[0]["type"] == "Heatmap")
        assert math.isfinite(heatmap["zmax"]), f"zmax should be finite, got {heatmap['zmax']}"

    def test_plot_force_cot_theta_contour_has_heatmap_and_contour(self, monkeypatch):
        _install_fake_plotly(monkeypatch)
        viewer = ShearViewer(_FakeCheck())

        fig = viewer.plot_force_cot_theta_contour(
            load_case={"V_Ed": 150.0, "M_Ed": 10.0, "N_Ed": 50.0},
            n_axial=4,
            n_moment=3,
            n_cot=4,
            metric="utilization",
            show=False,
        )

        trace_types = [t[0]["type"] for t in fig.traces]
        assert "Heatmap" in trace_types
        assert "Contour" in trace_types
        assert "Scatter" in trace_types
        trace_names = [t[0].get("name") for t in fig.traces if t[0]["type"] == "Scatter"]
        assert "Upper M-N limit" in trace_names
        assert "Lower M-N limit" in trace_names
        assert "_top_mask" in trace_names
        assert "_bottom_mask" in trace_names
        upper_trace = next(t[0] for t in fig.traces if t[0].get("name") == "Upper M-N limit")
        lower_trace = next(t[0] for t in fig.traces if t[0].get("name") == "Lower M-N limit")
        assert max(upper_trace["y"]) < viewer.check._diagram.n_limit
        assert min(lower_trace["y"]) > -viewer.check._diagram.n_limit
        heatmap_trace = next(t[0] for t in fig.traces if t[0]["type"] == "Heatmap")
        heatmap_z = np.asarray(heatmap_trace["z"], dtype=float)
        assert np.isnan(heatmap_z).any()
        assert np.isfinite(heatmap_z).any()
        assert hasattr(fig, "frames")
        assert 30 <= len(fig.frames) <= 70
        layout = fig.layout_updates[-1]
        assert "sliders" in layout
        _assert_slider_animation_controls_top_right(layout)
        _assert_force_slider_has_nice_steps(layout)

    def test_plot_force_cot_theta_contour_moment_on_y_axis_uses_axial_slider(self, monkeypatch):
        _install_fake_plotly(monkeypatch)
        viewer = ShearViewer(_FakeCheck())

        fig = viewer.plot_force_cot_theta_contour(
            load_case={"V_Ed": 150.0, "M_Ed": 10.0, "N_Ed": 50.0},
            n_axial=5,
            n_moment=4,
            moment_on_y_axis=True,
            n_cot=4,
            metric="utilization",
            show=False,
        )

        assert hasattr(fig, "frames")
        assert 30 <= len(fig.frames) <= 70
        layout = fig.layout_updates[-1]
        _assert_slider_animation_controls_top_right(layout)
        assert "sliders" in layout
        assert layout.get("yaxis_title") == "Moment My_Ed (kN·m)"
        assert layout["sliders"][0]["currentvalue"]["prefix"] == "N_Ed (kN): "
        upper_trace = next(t[0] for t in fig.traces if t[0].get("name") == "Upper M-N limit")
        lower_trace = next(t[0] for t in fig.traces if t[0].get("name") == "Lower M-N limit")
        assert max(upper_trace["y"]) < viewer.check._diagram.m_limit
        assert min(lower_trace["y"]) > -viewer.check._diagram.m_limit
        _assert_force_slider_has_nice_steps(layout)

    def test_plot_axial_moment_contour_has_heatmap_contour_and_frames(self, monkeypatch):
        _install_fake_plotly(monkeypatch)
        viewer = ShearViewer(_FakeCheck())

        fig = viewer.plot_axial_moment_contour(
            V_Ed=150.0,
            n_moment=4,
            n_axial=4,
            n_cot=3,
            show=False,
        )

        trace_types = [t[0]["type"] for t in fig.traces]
        assert "Heatmap" in trace_types
        assert "Contour" in trace_types
        assert "Scatter" in trace_types
        assert hasattr(fig, "frames")
        assert len(fig.frames) == 31
        layout = fig.layout_updates[-1]
        _assert_slider_animation_controls_top_right(layout)
        assert "sliders" in layout
        assert layout["sliders"][0]["steps"][0]["label"] == "1.00"
        assert layout["sliders"][0]["steps"][-1]["label"] == "2.50"

        boundary_trace = next(t[0] for t in fig.traces if t[0]["type"] == "Scatter" and t[0]["mode"] == "lines")
        assert boundary_trace["name"] == "M-N Capacity"

        heatmap_trace = next(t[0] for t in fig.traces if t[0]["type"] == "Heatmap")
        heatmap_z = np.asarray(heatmap_trace["z"], dtype=float)
        assert np.isnan(heatmap_z).any()
        assert np.isfinite(heatmap_z).any()

    def test_plot_axial_moment_contour_plots_loadcases_with_binary_colors(self, monkeypatch):
        _install_fake_plotly(monkeypatch)
        viewer = ShearViewer(_FakeCheck())

        fig = viewer.plot_axial_moment_contour(
            V_Ed=150.0,
            loadcases=[
                {"M_Ed": 0.0, "N_Ed": 0.0, "name": "Inside"},
                {"M_Ed": 120.0, "N_Ed": 0.0, "name": "Outside"},
            ],
            n_moment=4,
            n_axial=4,
            n_cot=3,
            show=False,
        )

        marker_traces = [t[0] for t in fig.traces if t[0]["type"] == "Scatter" and t[0]["mode"] == "markers"]
        colors = {trace["name"]: trace["marker"]["color"] for trace in marker_traces}
        assert colors["Inside"] == "green"
        assert colors["Outside"] == "red"

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


class TestSliderValueAllocation:
    """_build_slider_values must not allocate an unbounded array for a tiny step."""

    def test_tiny_step_is_capped(self):
        from materials.reinforced_concrete.analysis.shear_viewer import _build_slider_values

        vals = _build_slider_values(value_min=0.0, value_max=100.0, n_points=10, step=1e-9)
        assert len(vals) <= 10000
        assert vals[0] == pytest.approx(0.0)
        assert vals[-1] == pytest.approx(100.0)

    def test_normal_step_unchanged(self):
        from materials.reinforced_concrete.analysis.shear_viewer import _build_slider_values

        vals = _build_slider_values(value_min=0.0, value_max=10.0, n_points=5, step=2.0)
        assert list(vals) == pytest.approx([0.0, 2.0, 4.0, 6.0, 8.0, 10.0])
