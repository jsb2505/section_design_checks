"""
Tests for helper functions in geometry/rebar_layer.py.
"""

from __future__ import annotations

import math

import pytest

from materials.reinforced_concrete.geometry import (
    create_circular_perimeter_rebars,
    create_custom_rebar_layer,
    create_linear_rebar_layer,
    create_rectangular_perimeter_rebars,
)
from materials.reinforced_concrete.materials import Rebar


def _bar() -> Rebar:
    return Rebar(diameter=16, grade="B500B")


class TestCreateLinearRebarLayer:
    """Tests for TestCreateLinearRebarLayer."""
    def test_invalid_n_bars_and_omit_all(self):
        """Test invalid n bars and omit all."""
        with pytest.raises(ValueError, match="at least 1"):
            create_linear_rebar_layer(_bar(), 0, (0.0, 0.0), (100.0, 0.0))

        with pytest.raises(ValueError, match="All bars were omitted"):
            create_linear_rebar_layer(_bar(), 1, (0.0, 0.0), (100.0, 0.0), omit_start=True, omit_end=True)

    def test_single_bar_midpoint_and_multi_bar_spacing(self):
        """Test single bar midpoint and multi bar spacing."""
        one = create_linear_rebar_layer(_bar(), 1, (0.0, 0.0), (100.0, 20.0), layer_name="L1")
        assert len(one.positions) == 1
        assert one.positions[0].x == pytest.approx(50.0, rel=1e-12)
        assert one.positions[0].y == pytest.approx(10.0, rel=1e-12)
        assert one.layer_name == "L1"

        many = create_linear_rebar_layer(_bar(), 4, (0.0, 0.0), (300.0, 0.0))
        xs = [p.x for p in many.positions]
        assert xs == pytest.approx([0.0, 100.0, 200.0, 300.0], rel=1e-12)

    def test_omit_flags_trim_without_recomputing_spacing(self):
        """Test omit flags trim without recomputing spacing."""
        layer = create_linear_rebar_layer(_bar(), 4, (0.0, 0.0), (300.0, 0.0), omit_start=True, omit_end=True)
        xs = [p.x for p in layer.positions]
        assert xs == pytest.approx([100.0, 200.0], rel=1e-12)


class TestCreateRectangularPerimeterRebars:
    """Tests for TestCreateRectangularPerimeterRebars."""
    def test_input_validation_and_geometry_feasibility(self):
        """Test input validation and geometry feasibility."""
        with pytest.raises(ValueError, match="width and height must be > 0"):
            create_rectangular_perimeter_rebars(_bar(), width=0.0, height=500.0, cover=30.0, n_bars_width=2, n_bars_height=2)
        with pytest.raises(ValueError, match="cover must be >= 0"):
            create_rectangular_perimeter_rebars(_bar(), width=300.0, height=500.0, cover=-1.0, n_bars_width=2, n_bars_height=2)
        with pytest.raises(ValueError, match="must be >= 0"):
            create_rectangular_perimeter_rebars(_bar(), width=300.0, height=500.0, cover=30.0, n_bars_width=-1, n_bars_height=2)
        with pytest.raises(ValueError, match="hook_ref must be"):
            create_rectangular_perimeter_rebars(_bar(), width=300.0, height=500.0, cover=30.0, n_bars_width=2, n_bars_height=2, hook_ref=9)
        with pytest.raises(ValueError, match="cover \\+ bar radius is too large"):
            create_rectangular_perimeter_rebars(_bar(), width=40.0, height=40.0, cover=20.0, n_bars_width=2, n_bars_height=2)

    def test_full_perimeter_groups_and_layer_names(self):
        """Test full perimeter groups and layer names."""
        groups = create_rectangular_perimeter_rebars(
            _bar(),
            width=300.0,
            height=500.0,
            cover=30.0,
            n_bars_width=3,
            n_bars_height=2,
        )
        assert [g.layer_name for g in groups] == ["bottom", "top", "left", "right"]
        assert len(groups[0].positions) == 3
        assert len(groups[1].positions) == 3
        assert len(groups[2].positions) == 2
        assert len(groups[3].positions) == 2

    def test_only_side_groups_when_n_bars_width_is_zero(self):
        """Test only side groups when n bars width is zero."""
        groups = create_rectangular_perimeter_rebars(
            _bar(),
            width=300.0,
            height=500.0,
            cover=30.0,
            n_bars_width=0,
            n_bars_height=1,
        )
        assert [g.layer_name for g in groups] == ["left", "right"]
        assert len(groups[0].positions) == 1
        assert len(groups[1].positions) == 1

    @pytest.mark.parametrize(
        ("hook_ref", "origin", "expected_center"),
        [
            (0, (100.0, 200.0), (100.0, 200.0)),
            (2, (0.0, 0.0), (-150.0, 250.0)),
            (3, (0.0, 0.0), (-150.0, -250.0)),
            (4, (0.0, 0.0), (150.0, -250.0)),
        ],
    )
    def test_hook_ref_variants_place_layers_relative_to_expected_center(
        self, hook_ref, origin, expected_center
    ):
        """Test hook ref variants place layers relative to expected center."""
        bar = _bar()
        width = 300.0
        height = 500.0
        cover = 30.0
        offset = cover + bar.diameter / 2.0
        cx, cy = expected_center

        groups = create_rectangular_perimeter_rebars(
            bar,
            width=width,
            height=height,
            cover=cover,
            n_bars_width=1,
            n_bars_height=0,
            origin=origin,
            hook_ref=hook_ref,
        )

        assert [g.layer_name for g in groups] == ["bottom", "top"]
        bottom = groups[0].positions[0]
        top = groups[1].positions[0]
        assert bottom.x == pytest.approx(cx, rel=1e-12)
        assert bottom.y == pytest.approx(cy - height / 2.0 + offset, rel=1e-12)
        assert top.x == pytest.approx(cx, rel=1e-12)
        assert top.y == pytest.approx(cy + height / 2.0 - offset, rel=1e-12)


class TestCreateCircularPerimeterRebars:
    """Tests for TestCreateCircularPerimeterRebars."""
    def test_input_validation(self):
        """Test input validation."""
        with pytest.raises(ValueError, match="diameter must be > 0"):
            create_circular_perimeter_rebars(_bar(), diameter=0.0, cover=30.0, n_bars=8)
        with pytest.raises(ValueError, match="cover must be >= 0"):
            create_circular_perimeter_rebars(_bar(), diameter=400.0, cover=-1.0, n_bars=8)
        with pytest.raises(ValueError, match="at least 3 bars"):
            create_circular_perimeter_rebars(_bar(), diameter=400.0, cover=30.0, n_bars=2)
        with pytest.raises(ValueError, match="cover \\+ bar radius is too large"):
            create_circular_perimeter_rebars(_bar(), diameter=40.0, cover=20.0, n_bars=8)

    def test_geometry_and_start_angle(self):
        """Test geometry and start angle."""
        group = create_circular_perimeter_rebars(
            _bar(),
            diameter=400.0,
            cover=30.0,
            n_bars=4,
            origin=(10.0, -20.0),
            start_angle=90.0,
        )
        assert group.layer_name == "perimeter"
        assert len(group.positions) == 4

        radius = 400.0 / 2.0 - 30.0 - 16.0 / 2.0
        # first point at 90° from +x (i.e., straight up from origin)
        p0 = group.positions[0]
        assert p0.x == pytest.approx(10.0, rel=1e-12)
        assert p0.y == pytest.approx(-20.0 + radius, rel=1e-12)

        # all points should lie on the same radius
        for p in group.positions:
            r = math.hypot(p.x - 10.0, p.y + 20.0)
            assert r == pytest.approx(radius, rel=1e-12)


class TestCreateCustomRebarLayer:
    """Tests for TestCreateCustomRebarLayer."""
    def test_custom_positions_and_layer_name(self):
        """Test custom positions and layer name."""
        layer = create_custom_rebar_layer(
            _bar(),
            positions=[(0, 0), (100.5, 50), (200, -20)],
            layer_name="custom",
        )
        assert layer.layer_name == "custom"
        assert len(layer.positions) == 3
        assert layer.positions[1].x == pytest.approx(100.5, rel=1e-12)
        assert layer.positions[2].y == pytest.approx(-20.0, rel=1e-12)
