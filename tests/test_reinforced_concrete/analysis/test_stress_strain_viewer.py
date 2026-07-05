"""
Tests for StressStrainViewer helpers and plotting flow.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass

import numpy as np
import pytest

from materials.reinforced_concrete.analysis.stress_strain_viewer import (
    StressStrainViewer,
    _StressStrainPlotState,
)
from materials.reinforced_concrete.geometry import create_rectangular_section


@dataclass
class _CapacityResult:
    M_Rd: float | None
    N_Rd: float | None
    utilization: float | None


class _FakeConcreteModel:
    def get_stress_array(self, strains: np.ndarray) -> np.ndarray:
        # Simple linear stand-in for testing interpolation and plotting logic
        return 1000.0 * strains


class _FakeSection:
    def __init__(self):
        self._sec = create_rectangular_section(width=300.0, height=500.0)
        self.outline = self._sec.outline

    def get_centroid(self):
        return self._sec.get_centroid()

    def get_bounding_box(self):
        return self._sec.get_bounding_box()


class _FakeDiagram:
    def __init__(self):
        self.section = _FakeSection()
        self.concrete_model = _FakeConcreteModel()
        self.section_top = 500.0
        self.section_bottom = 0.0
        self.section_height = 500.0
        self._fibre_mat = np.array(["concrete", "concrete", "steel", "steel"])
        self._fibre_x = np.array([50.0, 250.0, 80.0, 220.0])
        self._fibre_i = np.array([0, 1, 0, 1])
        self._fibre_j = np.array([0, 0, 1, 1])

    def find_strains_for_MN(self, My_target: float, N_target: float):
        return 0.001, -0.001

    def get_fibre_forces_from_end_strains(self, eps_top: float, eps_bottom: float):
        forces = np.array([20000.0, -10000.0, 5000.0, -4000.0])  # N
        y = np.array([450.0, 50.0, 450.0, 50.0])  # mm
        areas = np.array([1000.0, 1000.0, 200.0, 200.0])  # mm2
        return forces, y, areas

    def _strain_field_from_end_strains(self, eps_top: float, eps_bottom: float):
        return np.array([0.0008, -0.0008, 0.0008, -0.0008])

    def get_capacity_vector(self, *, N_Ed: float, M_Ed: float):
        return _CapacityResult(M_Rd=120.0, N_Rd=N_Ed, utilization=0.8)


class _FakeDiagramWithConcreteOptions(_FakeDiagram):
    def _concrete_stress_with_options(self, strains: np.ndarray) -> np.ndarray:
        # Deliberately different from concrete_model.get_stress_array to verify
        # viewer uses option-aware path.
        arr = np.asarray(strains, dtype=float)
        return np.where(arr < 0.0, 0.0, 2000.0 * arr)


class _FakeFigure:
    def __init__(self):
        self.traces = []
        self.layout_updates = []
        self.xaxes_updates = []
        self.yaxes_updates = []
        self.annotations = []
        self.shown = False
        self.saved_path = None
        self.layout = types.SimpleNamespace(
            width=1200,
            height=800,
            margin=types.SimpleNamespace(l=60, r=120, t=80, b=200),
            xaxis=types.SimpleNamespace(domain=[0.0, 0.3]),
            yaxis=types.SimpleNamespace(domain=[0.0, 1.0]),
            xaxis2=types.SimpleNamespace(domain=[0.35, 0.65]),
            yaxis2=types.SimpleNamespace(domain=[0.0, 1.0]),
            xaxis3=types.SimpleNamespace(domain=[0.7, 1.0]),
            yaxis3=types.SimpleNamespace(domain=[0.0, 1.0]),
        )

    def add_trace(self, trace, row=None, col=None):
        self.traces.append((trace, row, col))

    def update_layout(self, **kwargs):
        self.layout_updates.append(kwargs)
        if "width" in kwargs:
            self.layout.width = kwargs["width"]
        if "height" in kwargs:
            self.layout.height = kwargs["height"]
        if "margin" in kwargs:
            m = kwargs["margin"]
            self.layout.margin = types.SimpleNamespace(
                l=m.get("l", self.layout.margin.l),
                r=m.get("r", self.layout.margin.r),
                t=m.get("t", self.layout.margin.t),
                b=m.get("b", self.layout.margin.b),
            )

    def update_xaxes(self, **kwargs):
        self.xaxes_updates.append(kwargs)

    def update_yaxes(self, **kwargs):
        self.yaxes_updates.append(kwargs)

    def add_annotation(self, **kwargs):
        self.annotations.append(kwargs)

    def show(self):
        self.shown = True

    def write_html(self, path: str):
        self.saved_path = path


class _FakeGo:
    Scatter = staticmethod(lambda **kwargs: {"type": "Scatter", **kwargs})
    Scattergl = staticmethod(lambda **kwargs: {"type": "Scattergl", **kwargs})
    Contour = staticmethod(lambda **kwargs: {"type": "Contour", **kwargs})


def _make_state(**overrides) -> _StressStrainPlotState:
    base = dict(
        My_Ed=80.0,
        N_Ed=150.0,
        eps_top=0.001,
        eps_bottom=-0.001,
        forces_N=np.array([20000.0, -10000.0, 5000.0, -4000.0]),
        areas=np.array([1000.0, 1000.0, 200.0, 200.0]),
        x=np.array([50.0, 250.0, 80.0, 220.0]),
        y=np.array([450.0, 50.0, 450.0, 50.0]),
        strains=np.array([0.0008, -0.0008, 0.0008, -0.0008]),
        stresses=np.array([20.0, -10.0, 25.0, -20.0]),
        conc_mask=np.array([True, True, False, False]),
        steel_mask=np.array([False, False, True, True]),
        y_top=500.0,
        y_bottom=0.0,
        h=500.0,
        y_na=250.0,
        na_in_section=True,
        F_c_comp=20.0,
        F_c_tens=-10.0,
        F_s_comp=5.0,
        F_s_tens=-4.0,
        y_c_comp=450.0,
        y_c_tens=50.0,
        y_s_comp=450.0,
        y_s_tens=50.0,
        y_C=400.0,
        y_T=80.0,
        z=320.0,
        max_stress_pos=20.0,
        min_stress_neg=-10.0,
        force_scale=0.2,
        fibre_i=np.array([0, 1, 0, 1]),
        fibre_j=np.array([0, 0, 1, 1]),
        bbox=(0.0, 0.0, 300.0, 500.0),
        section_failed=False,
        achieved_N=11.0,
        achieved_M=85.0,
        equilibrium_error_N=0.5,
        equilibrium_error_M=0.8,
        M_Rd_pos=120.0,
        M_Rd_neg=None,
        N_Rd=150.0,
        utilisation=0.8,
    )
    base.update(overrides)
    return _StressStrainPlotState(**base)


class TestStressStrainViewerHelpers:
    """Tests for TestStressStrainViewerHelpers."""
    def test_get_capacity_mapping(self):
        """Test get capacity mapping."""
        viewer = StressStrainViewer(_FakeDiagram())
        cap = viewer._get_capacity(My_Ed=50.0, N_Ed=100.0)
        assert cap["M_Rd_pos"] == pytest.approx(120.0, rel=1e-12)
        assert cap["N_Rd"] == pytest.approx(100.0, rel=1e-12)
        assert cap["utilisation"] == pytest.approx(0.8, rel=1e-12)

    def test_weighted_centroid_and_neutral_axis(self):
        """Test weighted centroid and neutral axis."""
        forces0 = np.array([10.0, 20.0])
        y0 = np.array([100.0, 200.0])
        mask0 = np.array([False, False])
        assert StressStrainViewer._weighted_centroid_y(forces0, y0, mask0) is None

        forces = np.array([10.0, -10.0])
        y = np.array([100.0, 200.0])
        mask = np.array([True, True])
        assert StressStrainViewer._weighted_centroid_y(forces, y, mask) is None

        forces2 = np.array([10.0, 20.0])
        assert StressStrainViewer._weighted_centroid_y(forces2, y, mask) == pytest.approx(
            (10.0 * 100.0 + 20.0 * 200.0) / 30.0,
            rel=1e-12,
        )

        y_na, in_section = StressStrainViewer._neutral_axis(
            eps_top=0.001,
            eps_bottom=-0.001,
            y_top=500.0,
            y_bottom=0.0,
            h=500.0,
        )
        assert y_na == pytest.approx(250.0, rel=1e-12)
        assert in_section is True

        y_na2, in_section2 = StressStrainViewer._neutral_axis(
            eps_top=0.001,
            eps_bottom=0.001,
            y_top=500.0,
            y_bottom=0.0,
            h=500.0,
        )
        assert y_na2 is None
        assert in_section2 is False

    def test_concrete_color_and_range_helpers(self):
        """Test concrete color and range helpers."""
        cmin0, cmax0, scale0 = StressStrainViewer._concrete_colorscale(np.array([]))
        assert (cmin0, cmax0) == pytest.approx((0.0, 1.0), rel=1e-12)
        assert scale0 == [[0, "white"], [1, "red"]]

        cmin, cmax, scale = StressStrainViewer._concrete_colorscale(np.array([5.0, 10.0]))
        assert cmin == pytest.approx(0.0, abs=1e-12)
        assert cmax == pytest.approx(10.0, abs=1e-12)
        assert scale == [[0, "white"], [1, "red"]]

        cmin2, cmax2, scale2 = StressStrainViewer._concrete_colorscale(np.array([-6.0, 10.0]))
        assert (cmin2, cmax2) == pytest.approx((-10.0, 10.0), rel=1e-12)
        assert scale2 == "RdBu_r"

        max_pos, min_neg = StressStrainViewer._concrete_stress_range(
            np.array([5.0, -3.0, 2.0]),
            np.array([True, True, False]),
        )
        assert max_pos == pytest.approx(5.0, rel=1e-12)
        assert min_neg == pytest.approx(-3.0, rel=1e-12)

        max_pos2, min_neg2 = StressStrainViewer._concrete_stress_range(
            np.array([5.0, -3.0, 2.0]),
            np.array([False, False, False]),
        )
        assert (max_pos2, min_neg2) == pytest.approx((1.0, -1.0), rel=1e-12)

    def test_force_scale_and_stress_x_range(self):
        """Test force scale and stress x range."""
        scale = StressStrainViewer._force_scale(
            max_stress_pos=20.0,
            min_stress_neg=-10.0,
            F_c_comp=30.0,
            F_c_tens=-5.0,
            F_s_comp=0.0,
            F_s_tens=-2.0,
        )
        assert scale == pytest.approx((20.0 * 0.5) / 30.0, rel=1e-12)

        viewer = StressStrainViewer(_FakeDiagram())
        s = _make_state()
        x_min, x_max = viewer._stress_x_range(s)
        assert x_min < 0.0
        assert x_max > 0.0

    def test_add_resultant_arrow_skip_and_add(self):
        """Test add resultant arrow skip and add."""
        fig = _FakeFigure()
        StressStrainViewer._add_resultant_arrow(
            fig=fig,
            go=_FakeGo,
            row=1,
            col=3,
            name="F",
            force=0.0,
            y=100.0,
            force_scale=0.1,
            line_color="red",
            tip_symbol="triangle-right",
            extra="X",
        )
        assert len(fig.traces) == 0

        StressStrainViewer._add_resultant_arrow(
            fig=fig,
            go=_FakeGo,
            row=1,
            col=3,
            name="F",
            force=10.0,
            y=100.0,
            force_scale=0.1,
            line_color="red",
            tip_symbol="triangle-right",
            extra="X",
        )
        assert len(fig.traces) == 2

    def test_build_annotation_text_contains_failure_details(self):
        """Test build annotation text contains failure details."""
        viewer = StressStrainViewer(_FakeDiagram())
        ok_text = viewer._build_annotation_text(_make_state(section_failed=False))
        assert "Load Case" in ok_text
        assert "x = 250.0 mm" in ok_text
        assert "SECTION FAILS" not in ok_text

        fail_text = viewer._build_annotation_text(
            _make_state(
                section_failed=True,
                equilibrium_error_N=12.0,
                equilibrium_error_M=8.0,
            )
        )
        assert "SECTION FAILS" in fail_text
        assert "Equilibrium Error" in fail_text

        na_out_text = viewer._build_annotation_text(
            _make_state(y_na=650.0, na_in_section=False)
        )
        assert "outside section" in na_out_text

    def test_interpolate_concrete_stress_profile(self):
        """Test interpolate concrete stress profile."""
        viewer = StressStrainViewer(_FakeDiagram())
        s = _make_state()
        y, strains, stresses = viewer._interpolate_concrete_stress_profile(s, n_points=5)
        assert len(y) == 5
        assert strains[0] == pytest.approx(s.eps_bottom, rel=1e-12)
        assert strains[-1] == pytest.approx(s.eps_top, rel=1e-12)
        assert np.allclose(stresses, 1000.0 * strains)

        s_zero_h = _make_state(h=0.0, eps_top=0.002, eps_bottom=0.001)
        _, strains2, _ = viewer._interpolate_concrete_stress_profile(s_zero_h, n_points=3)
        assert np.all(strains2 == pytest.approx(0.001, rel=1e-12))

    def test_interpolate_concrete_stress_profile_prefers_option_aware_path(self):
        """Interpolation should use diagram option-aware concrete stress when available."""
        viewer = StressStrainViewer(_FakeDiagramWithConcreteOptions())
        s = _make_state()
        _, strains, stresses = viewer._interpolate_concrete_stress_profile(s, n_points=5)
        assert np.allclose(stresses, np.where(strains < 0.0, 0.0, 2000.0 * strains))

    def test_outline_and_section_horizontal_segments(self):
        """Test outline and section horizontal segments."""
        viewer = StressStrainViewer(_FakeDiagram())
        outline_x, outline_y = viewer._get_outline_xy()
        assert len(outline_x) == len(outline_y)
        assert len(outline_x) >= 4

        segs_mid = viewer._section_horizontal_segments_at_y(250.0)
        assert len(segs_mid) >= 1
        xa, xb = segs_mid[0]
        assert xa == pytest.approx(0.0, abs=1e-6)
        assert xb == pytest.approx(300.0, abs=1e-6)

        segs_out = viewer._section_horizontal_segments_at_y(600.0)
        assert segs_out == []

    def test_section_horizontal_segments_collection_and_merge_branches(self):
        """Test section horizontal segments collection and merge branches."""
        viewer = StressStrainViewer(_FakeDiagram())

        class _LineNoCoords:
            geom_type = "LineString"
            coords = None

        class _LineA:
            geom_type = "LineString"
            coords = [(0.0, 0.0), (5.0, 0.0)]

        class _LineB:
            geom_type = "LineString"
            coords = [(4.0, 0.0), (10.0, 0.0)]

        class _LineC:
            geom_type = "LineString"
            coords = [(20.0, 0.0), (25.0, 0.0)]

        class _Point:
            geom_type = "Point"

        class _Collection:
            geom_type = "GeometryCollection"
            geoms = [_LineNoCoords(), _Point(), _LineA(), _LineB(), _LineC()]

        class _Outline:
            def intersection(self, cut):
                return _Collection()

        viewer.diagram.section.outline = _Outline()
        segs = viewer._section_horizontal_segments_at_y(250.0)
        assert segs == pytest.approx([(0.0, 10.0), (20.0, 25.0)], rel=1e-12)

    def test_section_horizontal_segments_shapely_import_error(self, monkeypatch):
        """Test section horizontal segments shapely import error."""
        import builtins
        real_import = builtins.__import__

        def _boom(name, *args, **kwargs):
            if name == "shapely.geometry":
                raise ImportError("no shapely")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _boom)
        viewer = StressStrainViewer(_FakeDiagram())
        with pytest.raises(ImportError, match="Shapely is required"):
            viewer._section_horizontal_segments_at_y(250.0)

    def test_add_concrete_filled_field_missing_indices_and_success(self):
        """Test add concrete filled field missing indices and success."""
        viewer = StressStrainViewer(_FakeDiagram())
        fig = _FakeFigure()
        s_missing = _make_state(fibre_i=None, fibre_j=None)
        with pytest.raises(ValueError, match="requires fibre i/j indices"):
            viewer._add_concrete_filled_field(fig, _FakeGo, s_missing, row=1, col=1)

        s = _make_state()
        viewer._add_concrete_filled_field(fig, _FakeGo, s, row=1, col=1)
        assert len(fig.traces) == 1
        assert fig.traces[0][0]["type"] == "Contour"

        # No concrete fibres -> early return
        fig2 = _FakeFigure()
        s_no_conc = _make_state(
            conc_mask=np.array([False, False, False, False]),
            steel_mask=np.array([True, True, True, True]),
        )
        viewer._add_concrete_filled_field(fig2, _FakeGo, s_no_conc, row=1, col=1)
        assert len(fig2.traces) == 0

        # Non-positive inferred grid size -> early return
        fig3 = _FakeFigure()
        s_bad_idx = _make_state(fibre_i=np.array([-1, -1, -1, -1]), fibre_j=np.array([0, 0, 0, 0]))
        viewer._add_concrete_filled_field(fig3, _FakeGo, s_bad_idx, row=1, col=1)
        assert len(fig3.traces) == 0

        # Degenerate bbox -> early return
        fig4 = _FakeFigure()
        s_bad_bbox = _make_state(bbox=(0.0, 0.0, 0.0, 500.0))
        viewer._add_concrete_filled_field(fig4, _FakeGo, s_bad_bbox, row=1, col=1)
        assert len(fig4.traces) == 0

    def test_add_concrete_filled_field_uses_option_aware_concrete_stress(self):
        """Filled field should use diagram option-aware concrete stress when available."""
        viewer = StressStrainViewer(_FakeDiagramWithConcreteOptions())
        fig = _FakeFigure()
        s = _make_state()
        viewer._add_concrete_filled_field(fig, _FakeGo, s, row=1, col=1)
        assert len(fig.traces) == 1
        contour = fig.traces[0][0]
        z = np.asarray(contour["z"], dtype=float)
        finite = np.isfinite(z)
        assert np.any(finite)
        assert np.all(z[finite] >= 0.0)

    def test_add_subplots_and_layout_methods(self):
        """Test add subplots and layout methods."""
        viewer = StressStrainViewer(_FakeDiagram())
        fig = _FakeFigure()
        s = _make_state()

        viewer._add_section_subplot(fig, _FakeGo, s, section_render="points")
        assert len(fig.traces) >= 3  # outline + concrete + steel (+NA segments)

        fig_filled = _FakeFigure()
        viewer._add_section_subplot(fig_filled, _FakeGo, s, section_render="filled")
        assert any(t[0].get("type") == "Contour" for t in fig_filled.traces)

        fig2 = _FakeFigure()
        viewer._add_strain_subplot(fig2, _FakeGo, s)
        assert len(fig2.traces) >= 3

        fig3 = _FakeFigure()
        viewer._add_strain_subplot(fig3, _FakeGo, _make_state(section_failed=True))
        assert len(fig3.traces) >= 3

        fig4 = _FakeFigure()
        viewer._add_stress_subplot(fig4, _FakeGo, s)
        assert len(fig4.traces) >= 4

        fig5 = _FakeFigure()
        viewer._add_stress_subplot(fig5, _FakeGo, _make_state(conc_mask=np.array([False, False, False, False])))
        assert len(fig5.traces) >= 1  # arrows/zero line still drawn

        viewer._apply_stress_strain_layout(fig4, s, title=None, width=900, height=700)
        assert fig4.annotations
        assert len(fig4.annotations) >= 2
        assert fig4.layout_updates
        assert fig4.layout.width == 900
        assert fig4.layout.height == 700

        fig6 = _FakeFigure()
        viewer._apply_stress_strain_layout(
            fig6,
            _make_state(section_failed=True),
            title=None,
            width=900,
            height=700,
        )
        assert fig6.layout.margin.b == 60

    def test_build_plot_state_success_and_solver_failure(self):
        """Test build plot state success and solver failure."""
        viewer = StressStrainViewer(_FakeDiagram())
        s = viewer._build_stress_strain_plot_state(My_Ed=80.0, N_Ed=150.0)
        assert isinstance(s, _StressStrainPlotState)
        assert s.My_Ed == pytest.approx(80.0, rel=1e-12)
        assert s.N_Ed == pytest.approx(150.0, rel=1e-12)
        assert s.M_Rd_pos == pytest.approx(120.0, rel=1e-12)

        class _FailingDiagram(_FakeDiagram):
            def find_strains_for_MN(self, My_target: float, N_target: float):
                raise ValueError("solver fail")

        viewer_fail = StressStrainViewer(_FailingDiagram())
        with pytest.raises(ValueError, match="Cannot find strain state"):
            viewer_fail._build_stress_strain_plot_state(My_Ed=80.0, N_Ed=150.0)


class TestStressStrainViewerPlotFlow:
    """Tests for TestStressStrainViewerPlotFlow."""
    def test_plot_import_error_when_plotly_missing(self, monkeypatch):
        """Test plot import error when plotly missing."""
        import builtins
        real_import = builtins.__import__

        def _boom(name, *args, **kwargs):
            if name.startswith("plotly"):
                raise ImportError("no plotly")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _boom)
        viewer = StressStrainViewer(_FakeDiagram())
        with pytest.raises(ImportError, match="Plotly is required for plotting"):
            viewer.plot(My_Ed=80.0, N_Ed=150.0, show=False)

    def test_plot_delegates_to_subplot_builders(self, monkeypatch):
        # Fake plotly modules
        """Test plot delegates to subplot builders."""
        go_mod = types.ModuleType("plotly.graph_objects")
        go_mod.Scatter = lambda **kwargs: kwargs
        go_mod.Scattergl = lambda **kwargs: kwargs
        go_mod.Contour = lambda **kwargs: kwargs

        subplots_mod = types.ModuleType("plotly.subplots")
        subplots_mod.make_subplots = lambda **kwargs: _FakeFigure()

        plotly_mod = types.ModuleType("plotly")
        plotly_mod.graph_objects = go_mod
        plotly_mod.subplots = subplots_mod

        monkeypatch.setitem(sys.modules, "plotly", plotly_mod)
        monkeypatch.setitem(sys.modules, "plotly.graph_objects", go_mod)
        monkeypatch.setitem(sys.modules, "plotly.subplots", subplots_mod)

        viewer = StressStrainViewer(_FakeDiagram())
        state = _make_state()

        called = {"section": 0, "strain": 0, "stress": 0, "layout": 0}
        monkeypatch.setattr(
            StressStrainViewer,
            "_build_stress_strain_plot_state",
            lambda self, **kw: state,
        )
        monkeypatch.setattr(
            StressStrainViewer,
            "_add_section_subplot",
            lambda self, fig, go, s, section_render="points", row=1, col=1: called.__setitem__("section", called["section"] + 1),
        )
        monkeypatch.setattr(
            StressStrainViewer,
            "_add_strain_subplot",
            lambda self, fig, go, s, row=2, col=1: called.__setitem__("strain", called["strain"] + 1),
        )
        monkeypatch.setattr(
            StressStrainViewer,
            "_add_stress_subplot",
            lambda self, fig, go, s, row=2, col=2: called.__setitem__("stress", called["stress"] + 1),
        )
        monkeypatch.setattr(
            StressStrainViewer,
            "_apply_stress_strain_layout",
            lambda self, fig, s, title, width, height: called.__setitem__("layout", called["layout"] + 1),
        )

        fig = viewer.plot(My_Ed=80.0, N_Ed=150.0, show=True)
        assert isinstance(fig, _FakeFigure)
        assert fig.shown is True
        assert called == {"section": 1, "strain": 1, "stress": 1, "layout": 1}

    def test_plot_save_path_writes_html(self, monkeypatch, tmp_path):
        """Test plot save path writes html."""
        go_mod = types.ModuleType("plotly.graph_objects")
        go_mod.Scatter = lambda **kwargs: kwargs
        go_mod.Scattergl = lambda **kwargs: kwargs
        go_mod.Contour = lambda **kwargs: kwargs

        subplots_mod = types.ModuleType("plotly.subplots")
        subplots_mod.make_subplots = lambda **kwargs: _FakeFigure()

        plotly_mod = types.ModuleType("plotly")
        plotly_mod.graph_objects = go_mod
        plotly_mod.subplots = subplots_mod

        monkeypatch.setitem(sys.modules, "plotly", plotly_mod)
        monkeypatch.setitem(sys.modules, "plotly.graph_objects", go_mod)
        monkeypatch.setitem(sys.modules, "plotly.subplots", subplots_mod)

        viewer = StressStrainViewer(_FakeDiagram())
        out = tmp_path / "viewer.html"
        fig = viewer.plot(My_Ed=80.0, N_Ed=150.0, show=False, save_path=out)
        assert isinstance(fig, _FakeFigure)
        assert fig.saved_path == str(out)
