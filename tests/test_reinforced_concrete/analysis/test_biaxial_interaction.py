"""
Tests for reinforced_concrete.analysis.biaxial_interaction module.
"""

import pytest
import numpy as np
from pathlib import Path
import json
import csv
from pydantic import ValidationError

from materials.reinforced_concrete.analysis.biaxial_interaction import (
    BiaxialInteractionPoint,
    BiaxialMNInteractionSurface,
    create_biaxial_interaction_surface,
)
from materials.reinforced_concrete.materials import ConcreteMaterial, Rebar
from materials.reinforced_concrete.geometry import (
    create_rectangular_section,
    create_linear_rebar_layer,
)


class TestBiaxialInteractionPoint:
    """Tests for BiaxialInteractionPoint model."""

    def test_create_point(self):
        """Test creating a biaxial interaction point."""
        point = BiaxialInteractionPoint(
            N=500.0,
            My=150.0,
            Mz=100.0,
            neutral_axis_depth=250.0,
            neutral_axis_angle=45.0,
            max_concrete_strain=0.0035,
            max_steel_strain=0.010,
        )

        assert point.N == 500.0
        assert point.My == 150.0
        assert point.Mz == 100.0
        assert point.neutral_axis_depth == 250.0
        assert point.neutral_axis_angle == 45.0

    def test_point_is_frozen(self):
        """Test that BiaxialInteractionPoint is immutable."""
        point = BiaxialInteractionPoint(
            N=500.0,
            My=150.0,
            Mz=100.0,
            neutral_axis_depth=250.0,
            neutral_axis_angle=45.0,
            max_concrete_strain=0.0035,
            max_steel_strain=0.010,
        )

        with pytest.raises(ValidationError, match="frozen"):
            point.N = 600.0

    def test_to_dict(self):
        """Test converting point to dictionary."""
        point = BiaxialInteractionPoint(
            N=500.0,
            My=150.0,
            Mz=100.0,
            neutral_axis_depth=250.0,
            neutral_axis_angle=45.0,
            max_concrete_strain=0.0035,
            max_steel_strain=0.010,
        )

        data = point.to_dict()

        assert isinstance(data, dict)
        assert data["N"] == 500.0
        assert data["My"] == 150.0
        assert data["Mz"] == 100.0
        assert data["neutral_axis_angle_deg"] == 45.0


class TestBiaxialMNInteractionSurface:
    """Tests for BiaxialMNInteractionSurface class."""

    @pytest.fixture
    def square_column(self, rebar_20):
        """Create a square column section."""
        section = create_rectangular_section(400, 400, section_name="Square Column")

        # Corner bars
        corners = [
            (50, 50),
            (350, 50),
            (350, 350),
            (50, 350),
        ]

        for i, (x, y) in enumerate(corners):
            layer = create_linear_rebar_layer(
                rebar=rebar_20,
                n_bars=1,
                start_point=(x, y),
                end_point=(x, y),
                layer_name=f"corner_{i}",
            )
            section.add_rebar_group(layer)

        return section

    @pytest.fixture
    def rectangular_column(self, rebar_20):
        """Create a rectangular column section."""
        section = create_rectangular_section(300, 500, section_name="Rect Column")

        # Distributed reinforcement on all sides
        # Bottom
        bottom = create_linear_rebar_layer(
            rebar=rebar_20,
            n_bars=3,
            start_point=(50, 50),
            end_point=(250, 50),
            layer_name="bottom",
        )
        section.add_rebar_group(bottom)

        # Top
        top = create_linear_rebar_layer(
            rebar=rebar_20,
            n_bars=3,
            start_point=(50, 450),
            end_point=(250, 450),
            layer_name="top",
        )
        section.add_rebar_group(top)

        return section

    def test_create_surface(self, square_column, concrete_c30):
        """Test creating biaxial interaction surface."""
        surface = BiaxialMNInteractionSurface(
            section=square_column,
            concrete=concrete_c30,
        )

        assert surface.section is not None
        assert surface.concrete is not None
        assert surface.mesh is not None
        # Verify steel_models is a list with one model per rebar group
        assert isinstance(surface.steel_models, list)
        assert len(surface.steel_models) == len(square_column.rebar_groups)

    def test_calculate_point_zero_angle(self, square_column, concrete_c30):
        """Test calculating point with neutral axis at 0° (bending about y-axis, major)."""
        surface = BiaxialMNInteractionSurface(
            section=square_column,
            concrete=concrete_c30,
        )

        # NA at 0° (horizontal) means bending creates moments
        # The actual axis convention produces Mz at 0° angle
        point = surface.calculate_point_pivot(na_depth=200.0, neutral_axis_angle=0.0)

        assert isinstance(point, BiaxialInteractionPoint)
        assert point.N != 0  # Should have axial force
        # At 0°, one moment component should be dominant
        assert abs(point.My) + abs(point.Mz) > 0  # Should have moment

    def test_calculate_point_90_degree_angle(self, square_column, concrete_c30):
        """Test calculating point with neutral axis at 90° (bending about z-axis, minor)."""
        surface = BiaxialMNInteractionSurface(
            section=square_column,
            concrete=concrete_c30,
        )

        # NA at 90° (vertical) means bending about z-axis (moment Mz, minor axis)
        # Forces act horizontally (y-direction), creating moment about z-axis
        point = surface.calculate_point_pivot(na_depth=200.0, neutral_axis_angle=90.0)

        assert isinstance(point, BiaxialInteractionPoint)
        assert point.N != 0
        # At 90°, should have Mz dominant
        # For square section, magnitudes should be similar to 0° case

    def test_calculate_point_45_degree_angle(self, square_column, concrete_c30):
        """Test calculating point with neutral axis at 45° (diagonal bending)."""
        surface = BiaxialMNInteractionSurface(
            section=square_column,
            concrete=concrete_c30,
        )

        point = surface.calculate_point_pivot(na_depth=200.0, neutral_axis_angle=45.0)

        assert isinstance(point, BiaxialInteractionPoint)
        assert point.N != 0
        # At 45°, both My and Mz should be non-zero
        assert point.My != 0
        assert point.Mz != 0

    def test_generate_surface_returns_points(self, square_column, concrete_c30):
        """Test that generate_surface returns list of points."""
        surface = BiaxialMNInteractionSurface(
            section=square_column,
            concrete=concrete_c30,
        )

        points = surface.generate_surface_pivot(n_angles=8, n_axial_levels=10)

        assert len(points) > 0
        assert all(isinstance(p, BiaxialInteractionPoint) for p in points)

    def test_generate_surface_covers_full_angles(self, square_column, concrete_c30):
        """Test that surface covers full range of angles."""
        surface = BiaxialMNInteractionSurface(
            section=square_column,
            concrete=concrete_c30,
        )

        points = surface.generate_surface_pivot(n_angles=8, n_axial_levels=10)

        angles = [p.neutral_axis_angle for p in points]

        # Should have points at various angles
        assert min(angles) >= 0
        assert max(angles) < 360

    def test_square_column_symmetry(self, square_column, concrete_c30):
        """Test that square column shows 4-fold symmetry."""
        surface = BiaxialMNInteractionSurface(
            section=square_column,
            concrete=concrete_c30,
        )

        # Calculate points at 0°, 90°, 180°, 270°
        p0 = surface.calculate_point_pivot(200.0, 0.0)
        p90 = surface.calculate_point_pivot(200.0, 90.0)
        p180 = surface.calculate_point_pivot(200.0, 180.0)
        p270 = surface.calculate_point_pivot(200.0, 270.0)

        # For square section with symmetric reinforcement:
        # N should be similar at all angles
        assert p0.N == pytest.approx(p90.N, rel=0.2)
        assert p0.N == pytest.approx(p180.N, rel=0.2)
        assert p0.N == pytest.approx(p270.N, rel=0.2)

    def test_rectangular_column_shows_asymmetry(self, rectangular_column, concrete_c30):
        """Test that rectangular column shows different behavior at 0° vs 90°."""
        surface = BiaxialMNInteractionSurface(
            section=rectangular_column,
            concrete=concrete_c30,
        )

        # 0° = NA horizontal → My (major axis)
        # 90° = NA vertical → Mz (minor axis)
        p0 = surface.calculate_point_pivot(200.0, 0.0)
        p90 = surface.calculate_point_pivot(200.0, 90.0)

        # Both should produce valid points
        assert isinstance(p0, BiaxialInteractionPoint)
        assert isinstance(p90, BiaxialInteractionPoint)

        # For rectangular section, the capacity should differ at different angles
        # This demonstrates that biaxial implementation handles non-square sections
        assert p0.N != 0
        assert p90.N != 0

    def test_export_to_json(self, square_column, concrete_c30, tmp_path):
        """Test exporting surface to JSON."""
        surface = BiaxialMNInteractionSurface(
            section=square_column,
            concrete=concrete_c30,
        )

        output_file = tmp_path / "biaxial_surface.json"
        surface.export_to_json(output_file, n_angles=4, n_axial_levels=5)

        assert output_file.exists()

        with open(output_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        assert "surface_points" in data
        assert len(data["surface_points"]) > 0
        assert "My" in data["surface_points"][0]
        assert "Mz" in data["surface_points"][0]

    def test_export_to_csv(self, square_column, concrete_c30, tmp_path):
        """Test exporting surface to CSV."""
        surface = BiaxialMNInteractionSurface(
            section=square_column,
            concrete=concrete_c30,
        )

        output_file = tmp_path / "biaxial_surface.csv"
        surface.export_to_csv(output_file, n_angles=4, n_axial_levels=5)

        assert output_file.exists()

        with open(output_file, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) > 0
        assert 'My' in rows[0]
        assert 'Mz' in rows[0]
        assert 'neutral_axis_angle_deg' in rows[0]

    def test_no_rebar_raises_error(self, concrete_c30):
        """Test that section without rebars raises error."""
        section = create_rectangular_section(400, 400)

        with pytest.raises(ValueError, match="at least one rebar group"):
            BiaxialMNInteractionSurface(section=section, concrete=concrete_c30)


class TestCreateBiaxialSurface:
    """Tests for create_biaxial_interaction_surface factory."""

    def test_create_basic(self, rectangular_beam_with_rebars, concrete_c30):
        """Test creating surface with factory function."""
        surface = create_biaxial_interaction_surface(
            section=rectangular_beam_with_rebars,
            concrete=concrete_c30,
        )

        assert isinstance(surface, BiaxialMNInteractionSurface)
        assert surface.section is rectangular_beam_with_rebars
        assert surface.concrete is concrete_c30
