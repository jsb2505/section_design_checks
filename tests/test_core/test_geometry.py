"""
Tests for core.geometry module.
"""

import pytest
from pydantic import ValidationError

from materials.core.geometry import BaseGeometry, Point2D


class ConcreteGeometry(BaseGeometry):
    """Minimal concrete implementation for testing."""

    def get_area(self) -> float:
        return 100.0

    def get_centroid(self):
        return (0.0, 0.0)

    def get_second_moment_area(self):
        return (100.0, 100.0, 0.0)

    def get_bounding_box(self):
        return (0.0, 0.0, 10.0, 10.0)


class TestBaseGeometry:
    """Tests for the abstract base class contract."""

    def test_cannot_instantiate_abstract(self):
        """Ensure BaseGeometry cannot be used directly."""
        with pytest.raises(TypeError):
            BaseGeometry()

    def test_concrete_implementation_works(self):
        """Ensure a valid subclass works."""
        geo = ConcreteGeometry()
        assert geo.get_area() == 100.0

    def test_extra_fields_forbidden(self):
        """Test that extra fields raise validation error."""
        with pytest.raises(ValidationError):
            ConcreteGeometry(random_field=500)


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
        with pytest.raises(ValidationError, match="frozen"):
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

    def test_point_distance(self):
        """Test Euclidean distance calculation (3-4-5 triangle)."""
        p1 = Point2D(x=0, y=0)
        p2 = Point2D(x=3, y=4)
        assert p1.distance_to(p2) == 5.0

    def test_point_distance_symmetric(self):
        """Test that distance is symmetric."""
        p1 = Point2D(x=10, y=20)
        p2 = Point2D(x=30, y=50)
        assert p1.distance_to(p2) == p2.distance_to(p1)

    def test_point_distance_to_self(self):
        """Test distance to self is zero."""
        p = Point2D(x=10, y=20)
        assert p.distance_to(p) == 0.0

    def test_point_addition(self):
        """Test vector addition."""
        p1 = Point2D(x=10, y=20)
        p2 = Point2D(x=5, y=5)

        p_sum = p1 + p2
        assert p_sum.x == 15
        assert p_sum.y == 25

    def test_point_subtraction(self):
        """Test vector subtraction."""
        p1 = Point2D(x=10, y=20)
        p2 = Point2D(x=5, y=5)

        p_diff = p1 - p2
        assert p_diff.x == 5
        assert p_diff.y == 15

    def test_point_arithmetic_returns_new_point(self):
        """Test that arithmetic operations return new Point2D instances."""
        p1 = Point2D(x=10, y=20)
        p2 = Point2D(x=5, y=5)

        p_sum = p1 + p2
        p_diff = p1 - p2

        # Original points unchanged (they're frozen anyway)
        assert p1.x == 10 and p1.y == 20
        assert p2.x == 5 and p2.y == 5

        # Results are Point2D instances
        assert isinstance(p_sum, Point2D)
        assert isinstance(p_diff, Point2D)

    def test_float_precision_handling(self):
        """Demonstrate how to handle float comparisons in geometry."""
        val = 1.0 / 3.0
        p1 = Point2D(x=val, y=val)
        p2 = Point2D(x=0.3333333333333333, y=0.3333333333333333)

        # Use pytest.approx for floating-point comparisons
        assert p1.x == pytest.approx(p2.x)
        assert p1.y == pytest.approx(p2.y)

    def test_point_hashable(self):
        """Test that Point2D is hashable (can be used in sets/dicts)."""
        p1 = Point2D(x=10, y=20)
        p2 = Point2D(x=10, y=20)
        p3 = Point2D(x=30, y=40)

        # Can be added to set
        point_set = {p1, p2, p3}
        assert len(point_set) == 2  # p1 and p2 are equal, so only 2 unique

        # Can be used as dict key
        point_dict = {p1: "first", p3: "third"}
        assert point_dict[p2] == "first"  # p2 == p1, so same key
