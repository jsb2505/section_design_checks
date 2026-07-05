"""
Tests for core.geometry module.
"""

import pytest
from materials.core.geometry import Point2D


class TestPoint2D:
    """Tests for Point2D class."""

    def test_create_point(self):
        """Test creating a point."""
        p = Point2D(x=10.0, y=20.0)
        assert p.x == 10.0
        assert p.y == 20.0

    def test_point_immutable(self):
        """Test that Point2D is immutable."""
        p = Point2D(x=10.0, y=20.0)
        with pytest.raises(Exception):  # Pydantic frozen model raises error
            p.x = 15.0

    def test_point_repr(self):
        """Test __repr__ method."""
        p = Point2D(x=10.5, y=20.7)
        assert "10.50" in repr(p)
        assert "20.70" in repr(p)

    def test_point_str(self):
        """Test __str__ method."""
        p = Point2D(x=10.5, y=20.7)
        assert "(10.50, 20.70)" == str(p)

    def test_point_equality(self):
        """Test point equality."""
        p1 = Point2D(x=10.0, y=20.0)
        p2 = Point2D(x=10.0, y=20.0)
        p3 = Point2D(x=10.0, y=20.1)

        assert p1 == p2
        assert p1 != p3

    def test_point_negative_coordinates(self):
        """Test points with negative coordinates."""
        p = Point2D(x=-10.0, y=-20.0)
        assert p.x == -10.0
        assert p.y == -20.0
