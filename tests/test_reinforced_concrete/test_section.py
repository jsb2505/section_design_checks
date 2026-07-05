"""
Tests for reinforced_concrete.geometry.section module.
"""

import pytest
import math
from shapely.geometry import Polygon
from pydantic import ValidationError
from materials.core.geometry import Point2D
from materials.reinforced_concrete.geometry import (
    RebarGroup,
    RCSection,
    create_rectangular_section,
    create_circular_section,
)


class TestRebarGroup:
    """Tests for RebarGroup class."""

    def test_create_rebar_group(self, rebar_20):
        """Test creating a rebar group."""
        positions = [Point2D(x=50, y=50), Point2D(x=150, y=50), Point2D(x=250, y=50)]
        group = RebarGroup(
            rebar=rebar_20,
            positions=positions,
            layer_name="bottom",
        )

        assert group.rebar.diameter == 20
        assert len(group.positions) == 3
        assert group.layer_name == "bottom"

    def test_n_bars(self, rebar_20):
        """Test n_bars property."""
        positions = [Point2D(x=50, y=50), Point2D(x=150, y=50)]
        group = RebarGroup(rebar=rebar_20, positions=positions)

        assert group.n_bars == 2

    def test_total_area(self, rebar_20):
        """Test total_area calculation."""
        positions = [Point2D(x=50, y=50), Point2D(x=150, y=50), Point2D(x=250, y=50)]
        group = RebarGroup(rebar=rebar_20, positions=positions)

        expected = 3 * rebar_20.area
        assert group.total_area == pytest.approx(expected)

    def test_get_centroid(self, rebar_20):
        """Test centroid calculation."""
        positions = [Point2D(x=100, y=50), Point2D(x=200, y=50)]
        group = RebarGroup(rebar=rebar_20, positions=positions)

        centroid = group.get_centroid()
        assert centroid.x == pytest.approx(150.0)  # (100 + 200) / 2
        assert centroid.y == pytest.approx(50.0)

    def test_positions_too_close(self, rebar_20):
        """Test that bars too close together are rejected."""
        positions = [Point2D(x=50, y=50), Point2D(x=50.5, y=50)]  # < 1mm apart
        with pytest.raises(ValidationError, match="too close"):
            RebarGroup(rebar=rebar_20, positions=positions)

    def test_single_bar_group(self, rebar_20):
        """Test group with single bar."""
        group = RebarGroup(rebar=rebar_20, positions=[Point2D(x=100, y=50)])
        assert group.n_bars == 1
        assert group.total_area == rebar_20.area

    def test_repr(self, rebar_20):
        """Test __repr__ method."""
        positions = [Point2D(x=50, y=50), Point2D(x=150, y=50)]
        group = RebarGroup(rebar=rebar_20, positions=positions, layer_name="test")

        r = repr(group)
        assert "2×" in r or "2" in r
        assert "test" in r


class TestRCSection:
    """Tests for RCSection class."""

    def test_create_section(self, rectangular_beam):
        """Test creating RC section."""
        assert rectangular_beam.section_name == "Test Beam"
        assert rectangular_beam.outline.is_valid

    def test_invalid_polygon(self):
        """Test that invalid polygon is rejected."""
        # Shapely will raise ValueError for too few points before Pydantic validates
        with pytest.raises(ValueError):
            invalid_poly = Polygon([(0, 0), (100, 0)])  # Not enough points
            RCSection(outline=invalid_poly)

    def test_get_area(self, rectangular_beam):
        """Test area calculation."""
        # 300 × 500 = 150,000 mm²
        assert rectangular_beam.get_area() == pytest.approx(150_000.0)

    def test_get_centroid(self, rectangular_beam):
        """Test centroid calculation."""
        cx, cy = rectangular_beam.get_centroid()
        # Should be at center: (150, 250)
        assert cx == pytest.approx(150.0, abs=1.0)
        assert cy == pytest.approx(250.0, abs=1.0)

    def test_get_bounding_box(self, rectangular_beam):
        """Test bounding box."""
        min_x, min_y, max_x, max_y = rectangular_beam.get_bounding_box()
        assert min_x == pytest.approx(0.0)
        assert min_y == pytest.approx(0.0)
        assert max_x == pytest.approx(300.0)
        assert max_y == pytest.approx(500.0)

    def test_add_rebar_group(self, rectangular_beam, rebar_20):
        """Test adding rebar group."""
        positions = [Point2D(x=50, y=50), Point2D(x=150, y=50), Point2D(x=250, y=50)]
        group = RebarGroup(rebar=rebar_20, positions=positions)

        rectangular_beam.add_rebar_group(group)

        assert len(rectangular_beam.rebar_groups) == 1
        assert rectangular_beam.total_steel_area == group.total_area

    def test_add_rebar_outside_section(self, rectangular_beam, rebar_20):
        """Test that rebars outside section are rejected."""
        # Position outside section
        positions = [Point2D(x=400, y=50)]  # x > 300
        group = RebarGroup(rebar=rebar_20, positions=positions)

        with pytest.raises(ValueError, match="outside section"):
            rectangular_beam.add_rebar_group(group)

    def test_total_steel_area_empty(self, rectangular_beam):
        """Test total steel area with no rebars."""
        assert rectangular_beam.total_steel_area == 0.0

    def test_total_steel_area_multiple_groups(self, rectangular_beam, rebar_16, rebar_20):
        """Test total steel area with multiple groups."""
        group1 = RebarGroup(
            rebar=rebar_20,
            positions=[Point2D(x=50, y=50), Point2D(x=250, y=50)]
        )
        group2 = RebarGroup(
            rebar=rebar_16,
            positions=[Point2D(x=150, y=450)]
        )

        rectangular_beam.add_rebar_group(group1)
        rectangular_beam.add_rebar_group(group2)

        expected = 2 * rebar_20.area + rebar_16.area
        assert rectangular_beam.total_steel_area == pytest.approx(expected)

    def test_reinforcement_ratio(self, rectangular_beam_with_rebars):
        """Test reinforcement ratio calculation."""
        rho = rectangular_beam_with_rebars.reinforcement_ratio
        expected = rectangular_beam_with_rebars.total_steel_area / 150_000.0
        assert rho == pytest.approx(expected)

    def test_get_rebar_positions(self, rectangular_beam_with_rebars):
        """Test getting all rebar positions."""
        positions = rectangular_beam_with_rebars.get_rebar_positions()

        assert len(positions) == 3  # 3 bars
        for x, y, area in positions:
            assert isinstance(x, float)
            assert isinstance(y, float)
            assert area > 0

    def test_get_steel_centroid(self, rectangular_beam_with_rebars):
        """Test steel centroid calculation."""
        cx, cy = rectangular_beam_with_rebars.get_steel_centroid()

        # 3 bars at (50,50), (150,50), (250,50)
        expected_x = (50 + 150 + 250) / 3
        expected_y = 50.0

        assert cx == pytest.approx(expected_x)
        assert cy == pytest.approx(expected_y)

    def test_get_steel_centroid_empty(self, rectangular_beam):
        """Test steel centroid with no rebars."""
        cx, cy = rectangular_beam.get_steel_centroid()
        assert cx == 0.0
        assert cy == 0.0

    def test_get_effective_depth_top(self, rectangular_beam_with_rebars):
        """Test effective depth from top."""
        d = rectangular_beam_with_rebars.get_effective_depth("top")
        # Section height = 500, steel at y = 50
        # d = 500 - 50 = 450
        assert d == pytest.approx(450.0)

    def test_get_effective_depth_bottom(self, rectangular_beam_with_rebars):
        """Test effective depth from bottom."""
        d = rectangular_beam_with_rebars.get_effective_depth("bottom")
        # Steel at y = 50
        assert d == pytest.approx(50.0)

    def test_repr(self, rectangular_beam):
        """Test __repr__ method."""
        r = repr(rectangular_beam)
        assert "Test Beam" in r
        assert "150000" in r  # Area


class TestCreateRectangularSection:
    """Tests for create_rectangular_section factory."""

    def test_create_basic(self):
        """Test creating basic rectangular section."""
        section = create_rectangular_section(300, 500)

        assert section.get_area() == pytest.approx(150_000.0)
        assert section.section_name == "Rect 300×500" or "Rect 300x500" in section.section_name

    def test_create_with_origin(self):
        """Test creating with custom origin."""
        section = create_rectangular_section(200, 400, origin=(100, 50))

        min_x, min_y, max_x, max_y = section.get_bounding_box()
        assert min_x == pytest.approx(100.0)
        assert min_y == pytest.approx(50.0)
        assert max_x == pytest.approx(300.0)
        assert max_y == pytest.approx(450.0)

    def test_create_with_name(self):
        """Test creating with custom name."""
        section = create_rectangular_section(300, 500, section_name="My Beam")
        assert section.section_name == "My Beam"


class TestCreateCircularSection:
    """Tests for create_circular_section factory."""

    def test_create_basic(self):
        """Test creating circular section."""
        section = create_circular_section(400)

        # Area = π · r² = π · 200²
        expected_area = math.pi * 200**2
        assert section.get_area() == pytest.approx(expected_area, rel=0.01)

    def test_create_with_origin(self):
        """Test creating with custom origin."""
        section = create_circular_section(400, origin=(100, 100))

        cx, cy = section.get_centroid()
        assert cx == pytest.approx(100.0, abs=1.0)
        assert cy == pytest.approx(100.0, abs=1.0)

    def test_create_with_custom_points(self):
        """Test creating with custom number of points."""
        section = create_circular_section(400, n_points=64)

        # Should still have correct area (within 0.2% due to polygon approximation)
        expected_area = math.pi * 200**2
        assert section.get_area() == pytest.approx(expected_area, rel=0.002)

    def test_create_with_name(self):
        """Test creating with custom name."""
        section = create_circular_section(400, section_name="Column C1")
        assert section.section_name == "Column C1"
