"""
Tests for M-N interaction diagram plot viewer.
"""

from __future__ import annotations

import builtins
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from materials.reinforced_concrete.analysis.mn_diagram_viewer import MNDiagramViewer


class _FakeFigure:
    def __init__(self):
        self.traces = []
        self.layout_updates = []
        self.shown = False
        self.saved_paths = []

    def add_trace(self, trace):
        self.traces.append(trace)

    def update_layout(self, **kwargs):
        self.layout_updates.append(kwargs)

    def show(self):
        self.shown = True

    def write_html(self, path):
        self.saved_paths.append(path)


def _install_fake_plotly(monkeypatch):
    go_mod = types.ModuleType("plotly.graph_objects")
    go_mod.Figure = _FakeFigure
    go_mod.Scatter = lambda **kwargs: kwargs

    plotly_mod = types.ModuleType("plotly")
    plotly_mod.graph_objects = go_mod

    monkeypatch.setitem(sys.modules, "plotly", plotly_mod)
    monkeypatch.setitem(sys.modules, "plotly.graph_objects", go_mod)


class _FakeDiagram:
    def __init__(self, capacities):
        self.capacities = capacities
        self.generate_calls = []
        self.capacity_calls = []

    def generate_diagram_points(self, n_points):
        self.generate_calls.append(n_points)
        return [
            SimpleNamespace(M=-100.0, N=200.0),
            SimpleNamespace(M=100.0, N=-200.0),
        ]

    def get_capacity_vector(self, *, N_Ed, M_Ed, n_points, return_details=False):
        self.capacity_calls.append((N_Ed, M_Ed, n_points, return_details))
        return self.capacities[(M_Ed, N_Ed)]


class TestMNDiagramViewer:
    """Tests for TestMNDiagramViewer."""
    def test_plot_raises_clean_error_if_plotly_missing(self, monkeypatch):
        """Test plot raises clean error if plotly missing."""
        monkeypatch.delitem(sys.modules, "plotly", raising=False)
        monkeypatch.delitem(sys.modules, "plotly.graph_objects", raising=False)
        real_import = builtins.__import__

        def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "plotly.graph_objects":
                raise ImportError("plotly not installed")
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", _fake_import)
        viewer = MNDiagramViewer(_FakeDiagram(capacities={}))

        with pytest.raises(ImportError, match="Plotly is required"):
            viewer.plot(show=False)

    def test_plot_without_load_points_adds_curve_and_origin(self, monkeypatch):
        """Test plot without load points adds curve and origin."""
        _install_fake_plotly(monkeypatch)
        diagram = _FakeDiagram(capacities={})
        viewer = MNDiagramViewer(diagram)

        fig = viewer.plot(show=False, title="Custom M-N", width=700, height=500)

        assert isinstance(fig, _FakeFigure)
        assert fig.shown is False
        assert len(fig.traces) == 2
        assert fig.traces[0]["name"] == "M-N Capacity"
        assert fig.traces[1]["name"] == "Origin"
        assert diagram.generate_calls == [120]

        layout = fig.layout_updates[-1]
        assert layout["title"]["text"] == "Custom M-N"
        assert layout["width"] == 700
        assert layout["height"] == 500

    def test_plot_load_points_with_vectors_and_metadata(self, monkeypatch):
        """Test plot load points with vectors and metadata."""
        _install_fake_plotly(monkeypatch)
        capacities = {
            (10.0, 100.0): SimpleNamespace(M_Rd=20.0, N_Rd=180.0, utilization=0.6, is_safe=True),
            (20.0, 120.0): SimpleNamespace(M_Rd=30.0, N_Rd=200.0, utilization=0.95, is_safe=True),
            (30.0, 140.0): SimpleNamespace(M_Rd=35.0, N_Rd=210.0, utilization=1.1, is_safe=False),
        }
        diagram = _FakeDiagram(capacities=capacities)
        viewer = MNDiagramViewer(diagram)

        fig = viewer.plot(
            load_points=[
                {"M_Ed": 10.0, "N_Ed": 100.0, "name": "LC1"},
                {"M_Ed": 20.0, "N_Ed": 120.0, "name": "LC2"},
                {"M_Ed": 30.0, "N_Ed": 140.0, "name": "LC3"},
            ],
            show_vectors=True,
            show_metadata=True,
            show=False,
            n_points=40,
        )

        # 2 base traces + 3*(2 vectors + 1 marker)
        assert len(fig.traces) == 11
        assert len(diagram.capacity_calls) == 3

        marker_by_name = {t["name"]: t for t in fig.traces if t.get("mode") == "markers" and t.get("name", "").startswith("LC")}
        assert marker_by_name["LC1"]["marker"]["color"] == "green"
        assert marker_by_name["LC2"]["marker"]["color"] == "orange"
        assert marker_by_name["LC3"]["marker"]["color"] == "red"

        assert "Utilization" in marker_by_name["LC1"]["hovertemplate"]
        assert "PASS" in marker_by_name["LC2"]["hovertemplate"]
        assert "FAIL" in marker_by_name["LC3"]["hovertemplate"]

    def test_plot_load_points_without_metadata_and_without_capacity_vectors(self, monkeypatch):
        """Test plot load points without metadata and without capacity vectors."""
        _install_fake_plotly(monkeypatch)
        capacities = {
            (15.0, 80.0): SimpleNamespace(M_Rd=None, N_Rd=None, utilization=0.7, is_safe=True),
        }
        diagram = _FakeDiagram(capacities=capacities)
        viewer = MNDiagramViewer(diagram)

        fig = viewer.plot(
            load_points=[{"M_Ed": 15.0, "N_Ed": 80.0, "name": "Case A"}],
            show_vectors=True,
            show_metadata=False,
            show=False,
        )

        # No vector traces because capacity point is unavailable.
        assert len(fig.traces) == 3
        marker = [t for t in fig.traces if t.get("name") == "Case A"][0]
        assert marker["hovertemplate"] == "Case A<extra></extra>"

    def test_plot_show_and_save_path(self, monkeypatch, tmp_path: Path):
        """Test plot show and save path."""
        _install_fake_plotly(monkeypatch)
        capacities = {
            (5.0, 40.0): SimpleNamespace(M_Rd=10.0, N_Rd=60.0, utilization=0.5, is_safe=True),
        }
        diagram = _FakeDiagram(capacities=capacities)
        viewer = MNDiagramViewer(diagram)
        save_path = tmp_path / "mn_plot.html"

        fig = viewer.plot(
            load_points=[{"M_Ed": 5.0, "N_Ed": 40.0, "name": "LC"}],
            show=True,
            save_path=save_path,
            width=777,
            height=555,
        )

        assert fig.shown is True
        assert fig.saved_paths == [str(save_path)]
        layout = fig.layout_updates[-1]
        assert layout["width"] == 777
        assert layout["height"] == 555
