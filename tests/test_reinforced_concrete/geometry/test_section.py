"""
Tests for reinforced_concrete.geometry.section module.
"""

import math

import pytest
from pydantic import ValidationError

from section_design_checks.core.geometry import Point2D
from section_design_checks.reinforced_concrete.geometry import (
    RCSection,
    RebarGroup,
    create_circular_section,
    create_rectangular_section,
)
from section_design_checks.reinforced_concrete.geometry.reinforcement_reconcile import (
    ReinforcementInvalidPolicy,
    find_invalid_rebars,
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

    def test_centroid(self, rebar_20):
        """Test centroid calculation."""
        positions = [Point2D(x=100, y=50), Point2D(x=200, y=50)]
        group = RebarGroup(rebar=rebar_20, positions=positions)

        centroid = group.centroid  # cached_property, not get_centroid()
        assert centroid.x == pytest.approx(150.0)  # (100 + 200) / 2
        assert centroid.y == pytest.approx(50.0)

    def test_positions_too_close(self, rebar_20):
        """Test that bars too close together are rejected."""
        positions = [Point2D(x=50, y=50), Point2D(x=50.5, y=50)]  # < 1mm apart
        with pytest.raises(ValidationError, match="Rebars overlap"):
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
        # Too few points for a valid polygon
        with pytest.raises(ValueError):
            RCSection(outline_coords=(Point2D(x=0, y=0), Point2D(x=100, y=0)))  # Only 2 points

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

        with pytest.raises(ValueError, match="not fully within the section outline"):
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
        """Test effective depth from bottom.

        When compression is from bottom, we need to explicitly specify tension_zone
        since rebars are only at bottom (would be compression zone by default).
        """
        d = rectangular_beam_with_rebars.get_effective_depth("bottom", tension_zone="bottom")
        # Steel at y = 50, measured from bottom (y=0)
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
        # With hook_ref=0 (centered), origin is the center
        section = create_circular_section(400, origin=(100, 100), hook_ref=0)

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


class TestReinforcementReconciliation:
    """Tests for reinforcement reconciliation when section boundaries change."""

    def test_default_policy_is_error(self):
        """Test that the default policy is ERROR."""
        section = create_rectangular_section(300, 500)
        assert section.reinforcement_policy == ReinforcementInvalidPolicy.ERROR

    def test_find_invalid_rebars_none_invalid(self, rectangular_beam_with_rebars):
        """Test find_invalid_rebars returns empty when all bars are valid."""
        details, indices = find_invalid_rebars(rectangular_beam_with_rebars)
        assert len(details) == 0
        assert len(indices) == 0

    def test_invalid_rebars_method(self, rectangular_beam_with_rebars):
        """Test the invalid_rebars() method on RCSection."""
        details, indices = rectangular_beam_with_rebars.invalid_rebars()
        assert len(details) == 0
        assert len(indices) == 0

    def test_shrink_section_error_policy_rejects(self, rebar_20):
        """Test that shrinking section with ERROR policy rejects change when rebars become invalid."""
        # Create section and add rebars near the edge
        section = create_rectangular_section(300, 500)
        section.reinforcement_policy = ReinforcementInvalidPolicy.ERROR

        # Add bar near right edge
        group = RebarGroup(
            rebar=rebar_20,
            positions=(Point2D(x=280, y=250),),  # 280mm from left, near 300mm edge
            layer_name="edge_bar",
        )
        section.add_rebar_group(group)

        # Shrink section width to 200mm - bar at x=280 is now outside
        new_coords = (
            Point2D(x=0, y=0),
            Point2D(x=200, y=0),
            Point2D(x=200, y=500),
            Point2D(x=0, y=500),
        )

        with pytest.raises(ValueError, match="reinforcement invalid"):
            section.update_outline(outline_coords=new_coords)

        # Section should be unchanged (rollback)
        min_x, _, max_x, _ = section.get_bounding_box()
        assert max_x == pytest.approx(300.0)  # Original width retained

    def test_shrink_section_drop_bars_policy(self, rebar_20):
        """Test that DROP_INVALID_BARS removes only the offending bars."""
        section = create_rectangular_section(300, 500)
        section.reinforcement_policy = ReinforcementInvalidPolicy.DROP_INVALID_BARS

        # Add 3 bars: 2 will remain valid, 1 will become invalid
        group = RebarGroup(
            rebar=rebar_20,
            positions=(
                Point2D(x=50, y=250),   # Will remain valid
                Point2D(x=150, y=250),  # Will remain valid
                Point2D(x=280, y=250),  # Will become invalid
            ),
            layer_name="test",
        )
        section.add_rebar_group(group)

        initial_bar_count = sum(len(g.positions) for g in section.rebar_groups)
        assert initial_bar_count == 3

        # Shrink section
        new_coords = (
            Point2D(x=0, y=0),
            Point2D(x=200, y=0),
            Point2D(x=200, y=500),
            Point2D(x=0, y=500),
        )
        report = section.update_outline(outline_coords=new_coords)

        # Check report
        assert report.invalid_bars == 1
        assert report.removed_bars == 1
        assert report.removed_groups == 0

        # Check section state
        final_bar_count = sum(len(g.positions) for g in section.rebar_groups)
        assert final_bar_count == 2
        assert len(section.rebar_groups) == 1  # Group still exists with 2 bars

    def test_shrink_section_drop_groups_policy(self, rebar_20):
        """Test that DROP_INVALID_GROUPS removes entire groups with any invalid bar."""
        section = create_rectangular_section(300, 500)
        section.reinforcement_policy = ReinforcementInvalidPolicy.DROP_INVALID_GROUPS

        # Add 2 groups: one will have an invalid bar, one will remain valid
        group1 = RebarGroup(
            rebar=rebar_20,
            positions=(Point2D(x=50, y=50),),
            layer_name="valid_group",
        )
        group2 = RebarGroup(
            rebar=rebar_20,
            positions=(
                Point2D(x=50, y=250),   # Valid
                Point2D(x=280, y=250),  # Invalid after shrink
            ),
            layer_name="invalid_group",
        )
        section.add_rebar_group(group1)
        section.add_rebar_group(group2)

        assert len(section.rebar_groups) == 2

        # Shrink section
        new_coords = (
            Point2D(x=0, y=0),
            Point2D(x=200, y=0),
            Point2D(x=200, y=500),
            Point2D(x=0, y=500),
        )
        report = section.update_outline(outline_coords=new_coords)

        # Check report
        assert report.invalid_groups == 1
        assert report.removed_groups == 1

        # Check section state - only valid_group should remain
        assert len(section.rebar_groups) == 1
        assert section.rebar_groups[0].layer_name == "valid_group"

    def test_shrink_section_allow_invalid_policy(self, rebar_20):
        """Test that ALLOW_INVALID keeps bars even when outside boundary."""
        section = create_rectangular_section(300, 500)
        section.reinforcement_policy = ReinforcementInvalidPolicy.ALLOW_INVALID

        # Add bar near edge
        group = RebarGroup(
            rebar=rebar_20,
            positions=(Point2D(x=280, y=250),),
            layer_name="edge_bar",
        )
        section.add_rebar_group(group)

        # Shrink section
        new_coords = (
            Point2D(x=0, y=0),
            Point2D(x=200, y=0),
            Point2D(x=200, y=500),
            Point2D(x=0, y=500),
        )
        report = section.update_outline(outline_coords=new_coords)

        # Check report - bars are invalid but not removed
        assert report.invalid_bars == 1
        assert report.removed_bars == 0

        # Bar should still exist
        assert len(section.rebar_groups) == 1
        assert len(section.rebar_groups[0].positions) == 1

        # invalid_rebars() should report it
        details, indices = section.invalid_rebars()
        assert len(indices) == 1

    def test_expand_section_no_issues(self, rectangular_beam_with_rebars):
        """Test that expanding section doesn't affect valid rebars."""
        initial_bar_count = sum(len(g.positions) for g in rectangular_beam_with_rebars.rebar_groups)

        # Expand section
        new_coords = (
            Point2D(x=0, y=0),
            Point2D(x=400, y=0),  # Wider
            Point2D(x=400, y=600),  # Taller
            Point2D(x=0, y=600),
        )
        report = rectangular_beam_with_rebars.update_outline(outline_coords=new_coords)

        assert report.invalid_bars == 0
        assert report.removed_bars == 0

        final_bar_count = sum(len(g.positions) for g in rectangular_beam_with_rebars.rebar_groups)
        assert final_bar_count == initial_bar_count

    def test_atomic_update_with_voids(self, rebar_20):
        """Test atomic update of outline and voids together."""
        section = create_rectangular_section(300, 500)
        section.reinforcement_policy = ReinforcementInvalidPolicy.ERROR

        # Add bar in the middle
        group = RebarGroup(
            rebar=rebar_20,
            positions=(Point2D(x=150, y=250),),
            layer_name="center",
        )
        section.add_rebar_group(group)

        # Update to add a void that encompasses the bar - should fail
        new_outline = (
            Point2D(x=0, y=0),
            Point2D(x=300, y=0),
            Point2D(x=300, y=500),
            Point2D(x=0, y=500),
        )
        void_around_bar = (
            Point2D(x=100, y=200),
            Point2D(x=200, y=200),
            Point2D(x=200, y=300),
            Point2D(x=100, y=300),
        )

        with pytest.raises(ValueError, match="reinforcement invalid"):
            section.update_outline(
                outline_coords=new_outline,
                voids_coords=(void_around_bar,),
            )

        # Section should have no voids (rolled back)
        assert len(section.outline.interiors) == 0

    def test_update_outline_with_temporary_policy_override(self, rebar_20):
        """Test that policy can be temporarily overridden in update_outline."""
        section = create_rectangular_section(300, 500)
        section.reinforcement_policy = ReinforcementInvalidPolicy.ERROR

        group = RebarGroup(
            rebar=rebar_20,
            positions=(Point2D(x=280, y=250),),
            layer_name="edge_bar",
        )
        section.add_rebar_group(group)

        # Shrink with temporary ALLOW_INVALID policy
        new_coords = (
            Point2D(x=0, y=0),
            Point2D(x=200, y=0),
            Point2D(x=200, y=500),
            Point2D(x=0, y=500),
        )
        report = section.update_outline(
            outline_coords=new_coords,
            reinforcement_policy=ReinforcementInvalidPolicy.ALLOW_INVALID,
        )

        # Bar should still exist (ALLOW_INVALID used)
        assert report.invalid_bars == 1
        assert report.removed_bars == 0
        assert len(section.rebar_groups) == 1

        # Original policy should be restored
        assert section.reinforcement_policy == ReinforcementInvalidPolicy.ERROR

    def test_direct_outline_coords_assignment_triggers_reconcile(self, rebar_20):
        """Test that directly assigning outline_coords triggers reconciliation.

        Note: Direct assignment does NOT support rollback - use update_outline() for
        atomic operations with rollback on failure. Direct assignment will raise but
        the coords will still be changed.
        """
        section = create_rectangular_section(300, 500)
        section.reinforcement_policy = ReinforcementInvalidPolicy.ERROR

        group = RebarGroup(
            rebar=rebar_20,
            positions=(Point2D(x=280, y=250),),
            layer_name="edge_bar",
        )
        section.add_rebar_group(group)

        # Direct assignment should trigger reconcile and fail with ERROR policy
        new_coords = (
            Point2D(x=0, y=0),
            Point2D(x=200, y=0),
            Point2D(x=200, y=500),
            Point2D(x=0, y=500),
        )

        # Note: ValidationError is raised (Pydantic wraps ValueError)
        with pytest.raises((ValueError, ValidationError)):
            section.outline_coords = new_coords

        # Note: With Pydantic's validate_assignment, the value IS changed before
        # validation runs, so there's no automatic rollback. Use update_outline()
        # for atomic operations with rollback.

    def test_move_section_rebars_become_invalid(self, rebar_20):
        """Test that moving section makes rebars invalid when they fall outside new boundary."""
        section = create_rectangular_section(300, 500)
        section.reinforcement_policy = ReinforcementInvalidPolicy.ERROR

        # Bar near bottom-left corner
        group = RebarGroup(
            rebar=rebar_20,
            positions=(Point2D(x=30, y=30),),  # Near (0,0) corner
            layer_name="corner",
        )
        section.add_rebar_group(group)

        # Move section so bar at (30, 30) is now outside
        # New section from (100,100) to (400,600) - bar at (30,30) is outside
        new_coords = (
            Point2D(x=100, y=100),
            Point2D(x=400, y=100),
            Point2D(x=400, y=600),
            Point2D(x=100, y=600),
        )

        with pytest.raises(ValueError, match="reinforcement invalid"):
            section.update_outline(outline_coords=new_coords)

    def test_rebar_group_positions_are_tuples(self, rebar_20):
        """Test that RebarGroup positions field is a tuple (immutable)."""
        positions = (Point2D(x=50, y=50), Point2D(x=150, y=50))
        group = RebarGroup(rebar=rebar_20, positions=positions)

        assert isinstance(group.positions, tuple)

    def test_section_with_void_rebar_in_void_rejected(self, rebar_20):
        """Test that adding rebar inside a void is rejected."""
        # Create section with void
        outline = (
            Point2D(x=0, y=0),
            Point2D(x=300, y=0),
            Point2D(x=300, y=500),
            Point2D(x=0, y=500),
        )
        void = (
            Point2D(x=100, y=200),
            Point2D(x=200, y=200),
            Point2D(x=200, y=300),
            Point2D(x=100, y=300),
        )
        section = RCSection(
            outline_coords=outline,
            voids_coords=(void,),
        )

        # Try to add bar inside the void
        group = RebarGroup(
            rebar=rebar_20,
            positions=(Point2D(x=150, y=250),),  # Inside void
            layer_name="in_void",
        )

        with pytest.raises(ValueError, match="not fully within the section outline"):
            section.add_rebar_group(group)


class TestRebarGroupImmutability:
    """Tests for RebarGroup immutability characteristics."""

    def test_positions_is_tuple(self, rebar_20):
        """Test that positions is stored as tuple."""
        positions = [Point2D(x=50, y=50), Point2D(x=150, y=50)]  # Pass as list
        group = RebarGroup(rebar=rebar_20, positions=positions)

        # Should be converted to tuple
        assert isinstance(group.positions, tuple)

    def test_point2d_is_frozen(self):
        """Test that Point2D is immutable."""
        point = Point2D(x=100, y=200)

        # Attempting to modify should raise
        with pytest.raises((ValidationError, TypeError, AttributeError)):
            point.x = 300

    def test_rebar_group_is_frozen(self, rebar_20):
        """Test that RebarGroup is immutable (frozen=True)."""
        group = RebarGroup(
            rebar=rebar_20,
            positions=(Point2D(x=50, y=50),),
            layer_name="test",
        )

        # Attempting to modify should raise
        with pytest.raises((ValidationError, TypeError, AttributeError)):
            group.layer_name = "modified"

    def test_rebar_group_frozen_but_not_hashable(self, rebar_20):
        """Test that RebarGroup is frozen but not hashable (Rebar field is unhashable)."""
        group = RebarGroup(
            rebar=rebar_20,
            positions=(Point2D(x=50, y=50),),
            layer_name="test",
        )

        # RebarGroup is frozen (immutable), but cannot be hashed because
        # the Rebar field itself is not hashable (Pydantic model not frozen)
        with pytest.raises(TypeError, match="unhashable"):
            hash(group)


class TestTransformedSectionProperties:
    """Tests for transformed section calculations (modular ratio method)."""

    def test_transformed_area_no_steel(self):
        """Test transformed area equals gross area when no steel present."""
        section = create_rectangular_section(300, 500)
        E_cm = 30000.0  # MPa

        A_tr = section.get_transformed_area(E_cm)
        A_gross = section.get_area()

        assert A_tr == pytest.approx(A_gross)

    def test_transformed_area_with_steel(self, rebar_20):
        """
        Test transformed area calculation with steel.

        Section 300x500 (150,000 mm²)
        1 bar ϕ20 (~314.16 mm²)
        E_s = 200,000 MPa, E_cm = 30,000 MPa
        α_e = 200000/30000 ≈ 6.667
        A_tr = 150,000 + 314.16 * (6.667 - 1) = 150,000 + 1,780.2 ≈ 151,780
        """
        section = create_rectangular_section(300, 500)
        group = RebarGroup(
            rebar=rebar_20,
            positions=(Point2D(x=150, y=250),),
        )
        section.add_rebar_group(group)

        E_cm = 30000.0
        E_s = rebar_20.E_s  # Should be 200000
        alpha_e = E_s / E_cm
        A_bar = rebar_20.area

        A_tr = section.get_transformed_area(E_cm)
        expected = (300 * 500) + A_bar * (alpha_e - 1)

        assert A_tr == pytest.approx(expected, rel=1e-4)

    def test_transformed_centroid_shifts_toward_steel(self, rebar_20):
        """Test that transformed centroid shifts toward the steel location."""
        section = create_rectangular_section(300, 500)

        # Add steel at bottom (y=50)
        group = RebarGroup(
            rebar=rebar_20,
            positions=(Point2D(x=150, y=50),),
        )
        section.add_rebar_group(group)

        E_cm = 30000.0

        # Gross centroid is at (150, 250) for 300x500 rectangle
        cx_gross, cy_gross = section.get_centroid()
        assert cy_gross == pytest.approx(250.0)

        # Transformed centroid should shift down toward steel
        A_tr, cx_tr, cy_tr = section.get_transformed_centroid(E_cm)

        assert cy_tr < cy_gross  # Should be lower than gross centroid
        assert cx_tr == pytest.approx(cx_gross)  # x unchanged (steel centered)

    def test_transformed_area_invalid_modulus(self):
        """Test that invalid E_cm raises error."""
        section = create_rectangular_section(300, 500)

        with pytest.raises(ValueError, match="positive"):
            section.get_transformed_area(0.0)

        with pytest.raises(ValueError, match="positive"):
            section.get_transformed_area(-1000.0)


class TestEffectiveDepthMultipleLayers:
    """Tests for effective depth with multiple rebar layers."""

    def test_effective_depth_two_layers(self, rebar_20):
        """
        Test effective depth calculation with two layers.

        Section: 300x500 (y from 0 to 500)
        Bottom layer at y=50
        Second layer at y=100
        Steel centroid = (50 + 100) / 2 = 75
        Effective depth from top = 500 - 75 = 425
        """
        section = create_rectangular_section(300, 500)

        # Add two layers
        layer1 = RebarGroup(
            rebar=rebar_20,
            positions=(Point2D(x=150, y=50),),
            layer_name="bottom",
        )
        layer2 = RebarGroup(
            rebar=rebar_20,
            positions=(Point2D(x=150, y=100),),
            layer_name="second",
        )
        section.add_rebar_group(layer1)
        section.add_rebar_group(layer2)

        # Compression from top, tension at bottom
        d = section.get_effective_depth(compression_face="top")

        # Steel centroid = (50 + 100) / 2 = 75 (equal areas)
        # d = 500 - 75 = 425
        assert d == pytest.approx(425.0)

    def test_effective_depth_weighted_by_area(self, rebar_16, rebar_20):
        """
        Test that effective depth uses area-weighted steel centroid.

        Section: 300x500
        Layer 1: 2×ϕ20 at y=50  (area = 2 × 314.16 = 628.32)
        Layer 2: 1×ϕ16 at y=100 (area = 1 × 201.06 = 201.06)

        Weighted centroid y = (628.32×50 + 201.06×100) / (628.32 + 201.06)
                            = (31416 + 20106) / 829.38
                            ≈ 62.1 mm
        d = 500 - 62.1 ≈ 437.9 mm
        """
        section = create_rectangular_section(300, 500)

        # 2 bars at bottom
        layer1 = RebarGroup(
            rebar=rebar_20,
            positions=(Point2D(x=100, y=50), Point2D(x=200, y=50)),
            layer_name="bottom",
        )
        # 1 bar at second layer
        layer2 = RebarGroup(
            rebar=rebar_16,
            positions=(Point2D(x=150, y=100),),
            layer_name="second",
        )
        section.add_rebar_group(layer1)
        section.add_rebar_group(layer2)

        # Calculate expected weighted centroid
        A1 = 2 * rebar_20.area
        A2 = 1 * rebar_16.area
        y_centroid = (A1 * 50 + A2 * 100) / (A1 + A2)
        expected_d = 500 - y_centroid

        d = section.get_effective_depth(compression_face="top")
        assert d == pytest.approx(expected_d, rel=1e-3)


class TestHollowSections:
    """Tests for hollow sections (sections with voids)."""

    @pytest.fixture
    def hollow_box(self):
        """
        Creates a 400x400 hollow box with a 200x200 void in the center.
        Outer: (0,0) to (400,400) -> Area = 160,000
        Inner: (100,100) to (300,300) -> Area = 40,000
        Net Concrete Area = 120,000 mm²
        """
        outer = (
            Point2D(x=0, y=0),
            Point2D(x=400, y=0),
            Point2D(x=400, y=400),
            Point2D(x=0, y=400),
        )
        inner = (
            Point2D(x=100, y=100),
            Point2D(x=300, y=100),
            Point2D(x=300, y=300),
            Point2D(x=100, y=300),
        )
        return RCSection(
            outline_coords=outer,
            voids_coords=(inner,),
            section_name="Hollow Box",
        )

    def test_hollow_area(self, hollow_box):
        """Verify that voids correctly subtract from area."""
        # Net Area: 400^2 - 200^2 = 160,000 - 40,000 = 120,000
        assert hollow_box.get_area() == pytest.approx(120_000)

    def test_hollow_inertia(self, hollow_box):
        """Verify that voids correctly subtract from second moment of area."""
        # Ixx for a centered hollow square about centroid:
        # (B*H^3 / 12) - (b*h^3 / 12)
        # (400^4 / 12) - (200^4 / 12) = 2,133,333,333 - 133,333,333 = 2,000,000,000
        I_xx, I_yy, _ = hollow_box.get_second_moment_area()
        assert I_xx == pytest.approx(2.0e9, rel=1e-3)
        assert I_yy == pytest.approx(2.0e9, rel=1e-3)  # Square, so symmetric

    def test_rebar_in_void_fails(self, hollow_box, rebar_16):
        """Ensure bars cannot be placed inside the hollow void."""
        # Center of void is (200, 200). Placing a bar there should fail.
        invalid_group = RebarGroup(
            rebar=rebar_16,
            positions=(Point2D(x=200, y=200),),
        )

        with pytest.raises(ValueError, match="not fully within"):
            hollow_box.add_rebar_group(invalid_group)

    def test_rebar_in_concrete_wall_passes(self, hollow_box, rebar_16):
        """Ensure bars can be placed in the actual concrete 'walls'."""
        # (50, 50) is in the bottom-left concrete wall
        valid_group = RebarGroup(
            rebar=rebar_16,
            positions=(Point2D(x=50, y=50),),
        )
        hollow_box.add_rebar_group(valid_group)
        assert len(hollow_box.rebar_groups) == 1

    def test_void_expansion_prunes_rebar(self, hollow_box, rebar_16):
        """Test that expanding a void prunes bars that the void 'swallows'."""
        # Place a bar at (50, 50) - currently inside the concrete wall
        pos = Point2D(x=50, y=50)
        hollow_box.add_rebar_group(RebarGroup(rebar=rebar_16, positions=(pos,)))
        hollow_box.reinforcement_policy = ReinforcementInvalidPolicy.DROP_INVALID_BARS

        # Expand the void so it covers (50, 50)
        # New void: (25, 25) to (375, 375)
        new_void = (
            Point2D(x=25, y=25),
            Point2D(x=375, y=25),
            Point2D(x=375, y=375),
            Point2D(x=25, y=375),
        )

        report = hollow_box.update_outline(
            outline_coords=hollow_box.outline_coords,
            voids_coords=(new_void,),
        )

        assert report.removed_bars == 1
        assert len(hollow_box.rebar_groups) == 0

    def test_hollow_centroid(self, hollow_box):
        """Verify centroid of symmetric hollow section is at center."""
        cx, cy = hollow_box.get_centroid()
        assert cx == pytest.approx(200.0)
        assert cy == pytest.approx(200.0)


class TestCircularVoids:
    """Tests for sections with circular voids (polygonal approximations)."""

    @pytest.fixture
    def circular_void_section(self):
        """
        400x400 Square section with a Circular Void (Ø200) at center (200, 200).
        The void is approximated with 32 points.
        """
        outer = (
            Point2D(x=0, y=0),
            Point2D(x=400, y=0),
            Point2D(x=400, y=400),
            Point2D(x=0, y=400),
        )

        # Create circular void using the factory function
        void_template = create_circular_section(
            diameter=200, n_points=32, origin=(200, 200), hook_ref=0
        )

        return RCSection(
            outline_coords=outer,
            voids_coords=(void_template.outline_coords,),
            section_name="Square with Circular Void",
        )

    def test_circular_void_area(self, circular_void_section):
        """Verify area subtraction with n_points approximation."""
        # True circle area = π * 100^2 ≈ 31,415.9
        # Polygon area (32 sides) will be slightly less than a true circle.
        # For inscribed polygon: A = (n/2) * R^2 * sin(2π/n)
        n_points = 32
        R = 100  # radius
        expected_void = (n_points / 2) * (R**2) * math.sin(2 * math.pi / n_points)
        expected_net = 160_000 - expected_void

        assert circular_void_section.get_area() == pytest.approx(expected_net, rel=1e-5)

    def test_rebar_in_circular_void_fails(self, circular_void_section, rebar_16):
        """Ensure bars cannot be placed inside the circular void."""
        # Center of void is (200, 200)
        invalid_group = RebarGroup(
            rebar=rebar_16,
            positions=(Point2D(x=200, y=200),),
        )

        with pytest.raises(ValueError, match="not fully within"):
            circular_void_section.add_rebar_group(invalid_group)

    def test_rebar_outside_circular_void_passes(self, circular_void_section, rebar_16):
        """Ensure bars can be placed in the concrete outside the circular void."""
        # (50, 50) is far from the circular void centered at (200,200)
        valid_group = RebarGroup(
            rebar=rebar_16,
            positions=(Point2D(x=50, y=50),),
        )
        circular_void_section.add_rebar_group(valid_group)
        assert len(circular_void_section.rebar_groups) == 1

    def test_rebar_near_void_tangent(self, circular_void_section, rebar_20):
        """
        Test a bar placed tangent to the void's nominal radius.

        Nominal radius = 100. Void center = (200, 200).
        Bar radius = 10 (ϕ20).
        Bar center at y = 200 - 100 - 10 = 90 should be just outside.
        """
        # Place bar below the void, with some margin for polygon approximation
        valid_pos = Point2D(x=200, y=85)  # Safe margin below void

        group = RebarGroup(rebar=rebar_20, positions=(valid_pos,))
        circular_void_section.add_rebar_group(group)
        assert len(circular_void_section.rebar_groups) == 1

    def test_low_resolution_void_polygon_approximation(self, rebar_16):
        """
        Demonstrate how low n_points affects the void shape.

        An 8-point 'circle' (octagon) has an apothem = R * cos(π/8) ≈ 0.924 * R.
        For R=100, the octagon's "flat" edges are only ~92.4mm from center.

        Important: The rebar validation checks if the ENTIRE bar disc fits, not just
        the center point. A ϕ16 bar has radius 8mm, so the bar center must be at
        least 8mm away from the void boundary.
        """
        outer = (
            Point2D(x=0, y=0),
            Point2D(x=400, y=0),
            Point2D(x=400, y=400),
            Point2D(x=0, y=400),
        )

        # Create octagonal void (8 points)
        void_template = create_circular_section(
            diameter=200, n_points=8, origin=(200, 200), hook_ref=0
        )

        section = RCSection(
            outline_coords=outer,
            voids_coords=(void_template.outline_coords,),
        )

        # Octagon apothem ≈ 92.4mm from center. Bar radius = 8mm.
        # Bar center must be at least 92.4 + 8 = 100.4mm from void center.
        # Place bar at (310, 200) = 110mm from center - safely outside
        safe_pos = Point2D(x=310, y=200)
        group = RebarGroup(rebar=rebar_16, positions=(safe_pos,))
        section.add_rebar_group(group)
        assert len(section.rebar_groups) == 1

        # Now try a bar that would fit in a true circle (r=100) but NOT in octagon
        # Bar center at (295, 200) = 95mm from center
        # Bar edge extends to 87mm from center (inside the r=100 true circle)
        # But octagon edge is at ~92.4mm, so 87mm is INSIDE the void
        risky_pos = Point2D(x=295, y=200)
        with pytest.raises(ValueError, match="not fully within"):
            section.add_rebar_group(RebarGroup(rebar=rebar_16, positions=(risky_pos,)))

    def test_polygonal_approximation_area_comparison(self):
        """Compare polygon approximation accuracy for different n_points."""
        R = 100  # radius
        true_circle_area = math.pi * R**2

        for n_points in [8, 16, 32, 64]:
            section = create_circular_section(diameter=200, n_points=n_points)
            polygon_area = section.get_area()

            # Inscribed polygon area = (n/2) * R^2 * sin(2π/n)
            expected_polygon = (n_points / 2) * (R**2) * math.sin(2 * math.pi / n_points)

            # Verify our section matches the formula
            assert polygon_area == pytest.approx(expected_polygon, rel=1e-6)

            # Calculate error vs true circle
            error_pct = 100 * (true_circle_area - polygon_area) / true_circle_area

            # Higher n_points should have lower error
            # n=32: error ≈ 0.64%, n=64: error ≈ 0.16%
            if n_points >= 32:
                assert error_pct < 1.0  # Less than 1% error for n>=32
            if n_points >= 64:
                assert error_pct < 0.2  # Less than 0.2% error for n>=64
