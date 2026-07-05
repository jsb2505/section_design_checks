"""
Tests for section plotting utilities.
"""

from __future__ import annotations

import sys
import types

from materials.reinforced_concrete.geometry import (
    create_box_section,
    create_linear_rebar_layer,
    create_rectangular_section,
)
from materials.reinforced_concrete.geometry.section_viewer import SectionViewer
from materials.reinforced_concrete.materials import ConcreteMaterial, Rebar


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
    go_mod.Scatter = lambda **kwargs: kwargs

    plotly_mod = types.ModuleType("plotly")
    plotly_mod.graph_objects = go_mod

    monkeypatch.setitem(sys.modules, "plotly", plotly_mod)
    monkeypatch.setitem(sys.modules, "plotly.graph_objects", go_mod)


def _make_reinforced_section():
    section = create_rectangular_section(width=300.0, height=500.0, section_name="S1")
    layer = create_linear_rebar_layer(
        rebar=Rebar(diameter=16, grade="B500B"),
        n_bars=2,
        start_point=(60.0, 60.0),
        end_point=(240.0, 60.0),
        layer_name="bottom",
    )
    section.add_rebar_group(layer)
    return section


class TestSectionViewer:
    """Tests for TestSectionViewer."""
    def test_plot_basic_without_rebar_or_concrete(self, monkeypatch):
        """Test plot basic without rebar or concrete."""
        _install_fake_plotly(monkeypatch)
        section = create_rectangular_section(width=300.0, height=500.0, section_name="Basic")
        viewer = SectionViewer(section)

        fig = viewer.plot(show=False, title="Custom", width=800, height=600)

        assert isinstance(fig, _FakeFigure)
        assert fig.shown is False
        assert len(fig.traces) >= 2  # concrete + gross centroid

        names = [t.get("name") for t in fig.traces]
        assert "Concrete" in names
        assert "Gross Centroid" in names
        assert "Transformed Centroid" not in names

        assert fig.layout_updates
        title_text = fig.layout_updates[-1]["title"]["text"]
        assert title_text == "Custom"

    def test_plot_with_rebars_and_concrete_adds_transformed_centroid(self, monkeypatch):
        """Test plot with rebars and concrete adds transformed centroid."""
        _install_fake_plotly(monkeypatch)
        section = _make_reinforced_section()
        viewer = SectionViewer(section)
        concrete = ConcreteMaterial(grade="C30/37")

        fig = viewer.plot(concrete=concrete, show=False)

        names = [t.get("name") for t in fig.traces]
        assert "Concrete" in names
        assert "Gross Centroid" in names
        assert "Transformed Centroid" in names
        assert any(name and "ϕ16" in name for name in names)

    def test_plot_show_true_calls_show(self, monkeypatch):
        """Test plot show true calls show."""
        _install_fake_plotly(monkeypatch)
        section = create_rectangular_section(width=300.0, height=500.0)
        viewer = SectionViewer(section)

        fig = viewer.plot(show=True)
        assert fig.shown is True

    def test_plot_with_void_adds_void_trace(self, monkeypatch):
        """Test plot with void adds void trace."""
        _install_fake_plotly(monkeypatch)
        section = create_box_section(width=400.0, height=300.0, t_web=40.0, t_flange_top=40.0, t_flange_bot=40.0)
        viewer = SectionViewer(section)
        fig = viewer.plot(show=False)
        names = [t.get("name") for t in fig.traces]
        assert "Void 1" in names

    def test_plot_save_path_calls_write_html(self, monkeypatch, tmp_path):
        """Test plot save path calls write html."""
        _install_fake_plotly(monkeypatch)
        section = create_rectangular_section(width=300.0, height=500.0)
        viewer = SectionViewer(section)
        save_path = tmp_path / "section_plot.html"
        fig = viewer.plot(show=False, save_path=save_path)
        assert fig.saved_path == str(save_path)


class TestSectionPlotWrapper:
    """Tests for TestSectionPlotWrapper."""
    def test_section_plot_delegates_to_section_viewer(self, monkeypatch):
        """Test section plot delegates to section viewer."""
        class _FakeViewer:
            def __init__(self, section):
                self.section = section

            def plot(self, **kwargs):
                return {"section_name": self.section.section_name, "kwargs": kwargs}

        fake_mod = types.SimpleNamespace(SectionViewer=_FakeViewer)
        monkeypatch.setitem(
            sys.modules,
            "materials.reinforced_concrete.geometry.section_viewer",
            fake_mod,
        )

        section = create_rectangular_section(width=300.0, height=500.0, section_name="Wrapped")
        concrete = ConcreteMaterial(grade="C30/37")

        result = section.plot(concrete=concrete, show=False, title="Wrapped title")
        assert result["section_name"] == "Wrapped"
        assert result["kwargs"]["show"] is False
        assert result["kwargs"]["title"] == "Wrapped title"
