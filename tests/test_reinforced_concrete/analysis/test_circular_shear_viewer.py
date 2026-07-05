"""
Tests for circular shear viewer plotting helpers.
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

from section_design_checks.reinforced_concrete.analysis.circular_shear_viewer import CircularShearViewer


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


class _FakeShearCheck:
    """Minimal mock of ShearCheck used by CircularSectionCheck._shear_check."""

    def __init__(self, *, diagram: _FakeDiagram | None = None):
        self._diagram = diagram or _FakeDiagram()

    def _get_diagram(self, ignore_compression_steel: bool = False):
        return self._diagram

    def find_effective_depth(
        self,
        M_Ed: float,
        N_Ed: float,
        eps_top=None,
        eps_bottom=None,
        ignore_compression_steel: bool = False,
    ) -> float:
        return 500.0

    def find_lever_arm(
        self,
        M_Ed: float,
        N_Ed: float,
        d: float,
        eps_top=None,
        eps_bottom=None,
        ignore_compression_steel: bool = False,
    ):
        return (450.0, 430.0)  # (z_ec2, z_mech)


class _FakeSection:
    section_name = "FakeCircular"

    def get_area(self) -> float:
        return 250000.0  # mm²


class _FakeCircularCheck:
    """Minimal mock of CircularSectionCheck."""

    def __init__(self, *, with_rebar: bool = True, diagram: _FakeDiagram | None = None):
        self.shear_reinforcement = _FakeRebar() if with_rebar else None
        self.use_sigma_cp_for_alpha_cw = False
        self.section = _FakeSection()
        self.diameter = 600.0
        self.use_simplified_lambda_1 = True
        self._shear_check = _FakeShearCheck(diagram=diagram)
        self._concrete_uls = SimpleNamespace(f_ck=30.0, gamma_c=1.5)

    @property
    def _f_cd_design(self) -> float:
        return 20.0

    @property
    def _f_ywd_design(self) -> float:
        return 435.0

    def calculate_lambda_1(self, z_0: float, z: float, integration_points: int = 100) -> float:
        return 0.85

    def calculate_lambda_2(self) -> float:
        return 1.0

    def calculate_equivalent_web_width(self, d: float, z: float):
        return (250.0, 260.0, 240.0)  # (b_w, b_wc, b_wt)

    def _find_rho_l(
        self,
        My_Ed: float,
        N_Ed: float,
        b_w: float,
        d: float,
        eps_top=None,
        eps_bottom=None,
        ignore_compression_steel: bool = False,
    ) -> float:
        return 0.01

    def calculate_V_Rd_c_uncracked(self, sigma_cp: float) -> float:
        return 140.0 + 5.0 * sigma_cp

    def _get_cot_theta_limits(self, sigma_cp: float):
        return (1.0, 2.5)


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


def _patch_tension_shift(monkeypatch):
    """Patch calculate_tension_shift to avoid needing real shear_utils internals."""
    result = SimpleNamespace(M_design=12.0, M_add=2.0, shift_distance_a_l=50.0, cot_theta=1.5, capped_by_M_cap=False, z=430.0, d=500.0)
    monkeypatch.setattr(
        "section_design_checks.reinforced_concrete.analysis.circular_shear_viewer.calculate_tension_shift",
        lambda **kwargs: result,
    )


def _patch_V_Rd_c_cracked(monkeypatch):
    """Patch find_V_Rd_c_cracked to return a fixed value."""
    monkeypatch.setattr(
        "section_design_checks.reinforced_concrete.analysis.circular_shear_viewer.find_V_Rd_c_cracked",
        lambda **kwargs: 120.0,
    )


def _patch_sigma_cp(monkeypatch):
    """Patch sigma_cp helpers to return simple values."""
    monkeypatch.setattr(
        "section_design_checks.reinforced_concrete.analysis.circular_shear_viewer.sigma_cp_from_N_and_area",
        lambda N_Ed, area: N_Ed * 1000.0 / area,
    )
    monkeypatch.setattr(
        "section_design_checks.reinforced_concrete.analysis.circular_shear_viewer.cap_sigma_cp_upper",
        lambda sigma_cp, f_cd: min(sigma_cp, 0.2 * f_cd),
    )


def _apply_all_patches(monkeypatch):
    """Apply all patches needed for tests."""
    _install_fake_plotly(monkeypatch)
    _patch_tension_shift(monkeypatch)
    _patch_V_Rd_c_cracked(monkeypatch)
    _patch_sigma_cp(monkeypatch)


class TestCircularShearViewer:
    """Tests for CircularShearViewer plotting API."""

    def test_plot_cot_theta_study_builds_traces(self, monkeypatch):
        _apply_all_patches(monkeypatch)
        viewer = CircularShearViewer(_FakeCircularCheck())

        fig = viewer.plot_cot_theta_study(
            load_case={"V_Ed": 150.0, "M_Ed": 10.0, "N_Ed": 50.0},
            n_points=6,
            show=False,
        )

        assert isinstance(fig, _FakeFigure)
        trace_types = [t[0]["type"] for t in fig.traces]
        assert trace_types.count("Scatter") >= 6
        trace_names = [t[0].get("name") for t in fig.traces if t[0]["type"] == "Scatter"]
        assert "V_Ed" in trace_names
        assert "V_Rd,c" in trace_names
        assert "V_Rd,s(cot)" in trace_names
        assert "V_Rd,max(cot)" in trace_names
        assert "V_Rd,max design" in trace_names
        assert "V_Rd,s design" in trace_names
        assert "Cot(theta),min" not in trace_names
        assert "Cot(theta),max" not in trace_names
        assert fig.shown is False

    def test_plot_cot_theta_study_intersection_trace(self, monkeypatch):
        _apply_all_patches(monkeypatch)
        viewer = CircularShearViewer(_FakeCircularCheck())

        fig = viewer.plot_cot_theta_study(
            load_case={"V_Ed": 150.0, "M_Ed": 10.0, "N_Ed": 50.0},
            n_points=60,
            show=False,
        )

        trace_names = [t[0].get("name") for t in fig.traces if t[0]["type"] == "Scatter"]
        assert "V_Ed,max" in trace_names

    def test_plot_cot_theta_study_adds_cot_theta_min_intercept_trace(self, monkeypatch):
        _apply_all_patches(monkeypatch)
        viewer = CircularShearViewer(_FakeCircularCheck())

        fig = viewer.plot_cot_theta_study(
            load_case={"V_Ed": 220.0, "M_Ed": 10.0, "N_Ed": 50.0},
            n_points=60,
            show=False,
        )

        scatter_traces = [t[0] for t in fig.traces if t[0]["type"] == "Scatter"]
        names = [t.get("name") for t in scatter_traces]
        assert "Cot(theta),min" in names
        assert "Cot(theta),max" not in names

        intercept_trace = next(t for t in scatter_traces if t.get("name") == "Cot(theta),min")
        assert len({float(x) for x in intercept_trace["x"]}) == 1
        assert min(intercept_trace["y"]) == pytest.approx(0.0)
        assert max(intercept_trace["y"]) == pytest.approx(220.0)
        assert "cot(theta)" in intercept_trace.get("hovertemplate", "")
        assert "V_Ed" in intercept_trace.get("hovertemplate", "")

    def test_find_V_Rd_max_guards_nonpositive_cot(self):
        """cot(theta) <= 0 must return 0.0, not raise ZeroDivisionError on 1/cot."""
        viewer = CircularShearViewer(_FakeCircularCheck())
        # The guard returns before touching the context, so context can be None.
        assert viewer._find_V_Rd_max(0.0, None) == 0.0
        assert viewer._find_V_Rd_max(-1.5, None) == 0.0

    def test_cot_theta_bounds_must_be_positive(self, monkeypatch):
        """User-supplied cot(theta) bounds of 0 (or negative) raise a clear error."""
        _apply_all_patches(monkeypatch)
        viewer = CircularShearViewer(_FakeCircularCheck())
        with pytest.raises(ValueError, match="cot.*positive"):
            viewer.plot_cot_theta_study(
                load_case={"V_Ed": 150.0, "M_Ed": 10.0, "N_Ed": 50.0},
                cot_theta_min=0.0,
                n_points=6,
                show=False,
            )

    def test_cot_theta_study_util_uses_per_cot_vrdmax(self, monkeypatch):
        """Utilization in the cot(theta) study must use the per-cot V_Rd,max(theta),
        not the constant design value at cot_min (which is the curve's peak)."""
        _apply_all_patches(monkeypatch)
        viewer = CircularShearViewer(_FakeCircularCheck())
        series = viewer._compute_cot_theta_study_series(
            load_case={"V_Ed": 150.0, "M_Ed": 10.0, "N_Ed": 50.0},
            n_points=8,
            cot_theta_min=None,
            cot_theta_max=None,
            use_uncracked_V_Rd_c=False,
        )
        V_Ed = series.context.V_Ed
        for v_s, v_max, util in zip(series.V_Rd_s_vals, series.V_Rd_max_theta_vals, series.util_vals):
            governing = min(v_s, v_max)
            expected = V_Ed / governing if governing > 0.0 else float("inf")
            assert util == pytest.approx(expected, rel=1e-9)
        # The scenario must include cots where per-cot V_Rd,max < the design value,
        # else the bug (using the design constant) would be indistinguishable.
        assert any(vmax < series.V_Rd_max_design - 1e-9 for vmax in series.V_Rd_max_theta_vals)

    def test_plot_cot_theta_study_adds_cot_theta_max_intercept_trace(self, monkeypatch):
        _apply_all_patches(monkeypatch)
        viewer = CircularShearViewer(_FakeCircularCheck())

        fig = viewer.plot_cot_theta_study(
            load_case={"V_Ed": 450.0, "M_Ed": 10.0, "N_Ed": 50.0},
            n_points=60,
            show=False,
        )

        scatter_traces = [t[0] for t in fig.traces if t[0]["type"] == "Scatter"]
        names = [t.get("name") for t in scatter_traces]
        assert "Cot(theta),max" in names
        assert "Cot(theta),min" not in names

        intercept_trace = next(t for t in scatter_traces if t.get("name") == "Cot(theta),max")
        assert len({float(x) for x in intercept_trace["x"]}) == 1
        assert min(intercept_trace["y"]) == pytest.approx(0.0)
        assert max(intercept_trace["y"]) == pytest.approx(450.0)
        assert "cot(theta)" in intercept_trace.get("hovertemplate", "")
        assert "V_Ed" in intercept_trace.get("hovertemplate", "")

    def test_plot_cot_theta_study_title_contains_lambda(self, monkeypatch):
        _apply_all_patches(monkeypatch)
        viewer = CircularShearViewer(_FakeCircularCheck())

        fig = viewer.plot_cot_theta_study(
            load_case={"V_Ed": 150.0, "M_Ed": 10.0, "N_Ed": 50.0},
            n_points=6,
            show=False,
        )

        layout = fig.layout_updates[0]
        assert "lambda_1" in layout["title"]
        assert "lambda_2" in layout["title"]
        assert "b_w" in layout["title"]

    def test_plot_cot_theta_moment_shift_study_builds_traces(self, monkeypatch):
        _apply_all_patches(monkeypatch)
        viewer = CircularShearViewer(_FakeCircularCheck())

        fig = viewer.plot_cot_theta_moment_shift_study(
            load_case={"V_Ed": 150.0, "M_Ed": 10.0, "N_Ed": 50.0},
            n_points=6,
            show=False,
        )

        assert isinstance(fig, _FakeFigure)
        trace_types = [t[0]["type"] for t in fig.traces]
        assert trace_types.count("Scatter") >= 3
        trace_names = [t[0].get("name") for t in fig.traces if t[0]["type"] == "Scatter"]
        assert "Utilization" in trace_names
        assert "M_add (tension shift)" in trace_names
        assert "Utilization = 1.0" in trace_names

    def test_plot_force_cot_theta_contour_has_heatmap_and_contour(self, monkeypatch):
        _apply_all_patches(monkeypatch)
        viewer = CircularShearViewer(_FakeCircularCheck())

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
        assert max(upper_trace["y"]) < viewer.check._shear_check._diagram.n_limit
        assert min(lower_trace["y"]) > -viewer.check._shear_check._diagram.n_limit
        heatmap_trace = next(t[0] for t in fig.traces if t[0]["type"] == "Heatmap")
        heatmap_z = np.asarray(heatmap_trace["z"], dtype=float)
        assert np.isnan(heatmap_z).any()
        assert np.isfinite(heatmap_z).any()
        assert hasattr(fig, "frames")
        assert 30 <= len(fig.frames) <= 70
        layout = fig.layout_updates[-1]
        _assert_slider_animation_controls_top_right(layout)
        assert "sliders" in layout
        assert layout["sliders"][0]["currentvalue"]["prefix"] == "My_Ed (kN*m): "
        _assert_force_slider_has_nice_steps(layout)

    def test_plot_force_cot_theta_contour_moment_on_y_axis_uses_axial_slider(self, monkeypatch):
        _apply_all_patches(monkeypatch)
        viewer = CircularShearViewer(_FakeCircularCheck())

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
        assert layout.get("yaxis_title") == "Moment My_Ed (kN*m)"
        assert layout["sliders"][0]["currentvalue"]["prefix"] == "N_Ed (kN): "
        upper_trace = next(t[0] for t in fig.traces if t[0].get("name") == "Upper M-N limit")
        lower_trace = next(t[0] for t in fig.traces if t[0].get("name") == "Lower M-N limit")
        assert max(upper_trace["y"]) < viewer.check._shear_check._diagram.m_limit
        assert min(lower_trace["y"]) > -viewer.check._shear_check._diagram.m_limit
        _assert_force_slider_has_nice_steps(layout)

    def test_plot_axial_moment_contour_has_heatmap_contour_and_frames(self, monkeypatch):
        _apply_all_patches(monkeypatch)
        viewer = CircularShearViewer(_FakeCircularCheck())

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
        _apply_all_patches(monkeypatch)
        viewer = CircularShearViewer(_FakeCircularCheck())

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

    def test_plot_force_cot_theta_contour_capacity_metric(self, monkeypatch):
        _apply_all_patches(monkeypatch)
        viewer = CircularShearViewer(_FakeCircularCheck())

        fig = viewer.plot_force_cot_theta_contour(
            load_case={"V_Ed": 150.0, "M_Ed": 10.0, "N_Ed": 50.0},
            n_axial=3,
            n_cot=3,
            metric="capacity",
            show=False,
        )

        trace_types = [t[0]["type"] for t in fig.traces]
        assert "Heatmap" in trace_types
        # No contour trace for non-utilization metrics
        assert "Contour" not in trace_types

    def test_plot_force_cot_theta_contour_invalid_metric(self, monkeypatch):
        _apply_all_patches(monkeypatch)
        viewer = CircularShearViewer(_FakeCircularCheck())

        with pytest.raises(ValueError, match="metric must be one of"):
            viewer.plot_force_cot_theta_contour(
                load_case={"V_Ed": 150.0, "M_Ed": 10.0, "N_Ed": 50.0},
                metric="invalid",
                show=False,
            )

    def test_plot_methods_require_shear_reinforcement(self, monkeypatch):
        _apply_all_patches(monkeypatch)
        viewer = CircularShearViewer(_FakeCircularCheck(with_rebar=False))
        with pytest.raises(ValueError, match="Shear reinforcement is required"):
            viewer.plot_cot_theta_study(
                load_case={"V_Ed": 150.0, "M_Ed": 10.0, "N_Ed": 50.0},
                show=False,
            )

    def test_plotly_import_error_branch(self, monkeypatch):
        _patch_tension_shift(monkeypatch)
        _patch_V_Rd_c_cracked(monkeypatch)
        _patch_sigma_cp(monkeypatch)

        real_import = builtins.__import__

        def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name in {"plotly.graph_objects", "plotly.subplots"}:
                raise ImportError("plotly missing")
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.delitem(sys.modules, "plotly", raising=False)
        monkeypatch.delitem(sys.modules, "plotly.graph_objects", raising=False)
        monkeypatch.delitem(sys.modules, "plotly.subplots", raising=False)
        monkeypatch.setattr(builtins, "__import__", _fake_import)

        viewer = CircularShearViewer(_FakeCircularCheck())
        with pytest.raises(ImportError, match="Plotly is required"):
            viewer.plot_cot_theta_study(
                load_case={"V_Ed": 150.0, "M_Ed": 10.0, "N_Ed": 50.0},
                show=False,
            )

    def test_save_path_writes_html(self, monkeypatch, tmp_path):
        _apply_all_patches(monkeypatch)
        viewer = CircularShearViewer(_FakeCircularCheck())

        save_file = tmp_path / "test_output.html"
        fig = viewer.plot_cot_theta_study(
            load_case={"V_Ed": 150.0, "M_Ed": 10.0, "N_Ed": 50.0},
            n_points=4,
            save_path=str(save_file),
            show=False,
        )

        assert fig.saved_path == str(save_file)

    def test_custom_cot_theta_bounds(self, monkeypatch):
        _apply_all_patches(monkeypatch)
        viewer = CircularShearViewer(_FakeCircularCheck())

        fig = viewer.plot_cot_theta_study(
            load_case={"V_Ed": 150.0, "M_Ed": 10.0, "N_Ed": 50.0},
            n_points=6,
            cot_theta_min=1.2,
            cot_theta_max=2.0,
            show=False,
        )

        assert isinstance(fig, _FakeFigure)
        assert len(fig.traces) >= 6
