"""
Tests for reinforced_concrete.geometry.fibre_mesh module.
"""

import pytest
import numpy as np
from shapely.geometry import Point
from materials.reinforced_concrete.geometry import FibreMesh, create_linear_rebar_layer
from materials.core.geometry import Point2D


class TestFibreMesh:
    """Tests for FibreMesh class."""

    @pytest.fixture
    def mesh_simple(self, rectangular_beam_with_rebars):
        """Simple fibre mesh."""
        return FibreMesh(
            section=rectangular_beam_with_rebars,
            n_fibres_width=10,
            n_fibres_height=20,
        )

    def test_create_mesh(self, mesh_simple, rectangular_beam_with_rebars):
        """Test creating fibre mesh."""
        expected_steel = sum(
            len(g.positions) for g in rectangular_beam_with_rebars.rebar_groups
        )
        assert mesh_simple.n_concrete_fibres > 0
        assert mesh_simple.n_steel_fibres == expected_steel
        assert mesh_simple.total_fibres == mesh_simple.n_concrete_fibres + expected_steel

    def test_concrete_fibres_generated(self, mesh_simple, rectangular_beam_with_rebars):
        """Test that concrete fibres lie within section bounding box and outline."""
        assert len(mesh_simple.concrete_fibres) > 0

        min_x, min_y, max_x, max_y = rectangular_beam_with_rebars.get_bounding_box()
        outline_buffered = rectangular_beam_with_rebars.outline.buffer(1e-6)

        for fibre in mesh_simple.concrete_fibres:
            assert fibre.material_type == "concrete"
            assert fibre.area > 0
            assert min_x - 1e-9 <= fibre.x <= max_x + 1e-9
            assert min_y - 1e-9 <= fibre.y <= max_y + 1e-9
            assert outline_buffered.covers(Point(fibre.x, fibre.y))

    def test_steel_fibres_generated(self, mesh_simple):
        """Test that steel fibres are generated."""
        assert len(mesh_simple.steel_fibres) == 3

        for fibre in mesh_simple.steel_fibres:
            assert fibre.material_type == "steel"
            assert fibre.area > 0

    def test_all_fibres(self, mesh_simple):
        """Test all_fibres property."""
        all_f = mesh_simple.all_fibres
        assert len(all_f) == mesh_simple.total_fibres

    def test_fibre_resolution(self, rectangular_beam_with_rebars):
        """Test that higher resolution creates more fibres."""
        mesh_coarse = FibreMesh(
            section=rectangular_beam_with_rebars,
            n_fibres_width=5,
            n_fibres_height=10,
        )
        mesh_fine = FibreMesh(
            section=rectangular_beam_with_rebars,
            n_fibres_width=20,
            n_fibres_height=40,
        )

        assert mesh_fine.n_concrete_fibres > mesh_coarse.n_concrete_fibres

    def test_exclude_steel_area(self, rectangular_beam_with_rebars):
        """Test excluding steel area from concrete fibres."""
        mesh_excluded = FibreMesh(
            section=rectangular_beam_with_rebars,
            n_fibres_width=10,
            n_fibres_height=20,
            exclude_steel_area=True,
        )
        mesh_included = FibreMesh(
            section=rectangular_beam_with_rebars,
            n_fibres_width=10,
            n_fibres_height=20,
            exclude_steel_area=False,
        )

        # Total concrete area should be less when steel is excluded
        concrete_area_excluded = sum(f.area for f in mesh_excluded.concrete_fibres)
        concrete_area_included = sum(f.area for f in mesh_included.concrete_fibres)

        assert concrete_area_excluded <= concrete_area_included

    def test_get_fibre_arrays(self, mesh_simple):
        """Test getting fibre data as arrays."""
        x, y, area, material_type, material_index, i, j = mesh_simple.get_fibre_arrays()

        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert isinstance(area, np.ndarray)
        assert isinstance(material_type, np.ndarray)
        assert isinstance(material_index, np.ndarray)
        assert isinstance(i, np.ndarray)
        assert isinstance(j, np.ndarray)

        assert len(x) == mesh_simple.total_fibres
        assert len(y) == mesh_simple.total_fibres
        assert len(area) == mesh_simple.total_fibres
        assert len(i) == mesh_simple.total_fibres
        assert len(j) == mesh_simple.total_fibres

    def test_fibre_arrays_properties(self, mesh_simple):
        """Test fibre array properties."""
        x, y, area, material_type, material_index, i, j = mesh_simple.get_fibre_arrays()

        # All areas should be positive
        assert np.all(area > 0)

        # Material types should be 'concrete' or 'steel'
        assert np.all((material_type == 'concrete') | (material_type == 'steel'))

        # Count material types
        n_concrete = np.sum(material_type == 'concrete')
        n_steel = np.sum(material_type == 'steel')

        assert n_concrete == mesh_simple.n_concrete_fibres
        assert n_steel == mesh_simple.n_steel_fibres

    def test_mesh_conservation_of_area(self, rectangular_beam):
        """Test that total fibre area approximately equals section area."""
        mesh = FibreMesh(
            section=rectangular_beam,
            n_fibres_width=20,
            n_fibres_height=30,
            exclude_steel_area=False,
        )

        total_fibre_area = sum(f.area for f in mesh.concrete_fibres)

        # Should be close to section area (within 2% due to discretization)
        assert total_fibre_area == pytest.approx(
            rectangular_beam.get_area(),
            rel=0.02
        )

    def test_steel_fibre_positions(self, rectangular_beam_with_rebars):
        """Test that steel fibres are at correct positions."""
        mesh = FibreMesh(
            section=rectangular_beam_with_rebars,
            n_fibres_width=10,
            n_fibres_height=20,
        )

        # Get steel fibre positions
        steel_positions = [(f.x, f.y) for f in mesh.steel_fibres]

        # Should have 3 steel fibres
        assert len(steel_positions) == 3

        # Check they're at expected y-coordinate (50mm from bottom)
        for x, y in steel_positions:
            assert y == pytest.approx(50.0)

    def test_concrete_fibre_grid_indices(self, mesh_simple):
        """Test that concrete fibres have valid grid indices."""
        for f in mesh_simple.concrete_fibres:
            assert 0 <= f.i < mesh_simple.n_fibres_width
            assert 0 <= f.j < mesh_simple.n_fibres_height

    def test_steel_fibre_grid_indices(self, mesh_simple):
        """Test that steel fibres have sentinel grid indices (-1)."""
        for f in mesh_simple.steel_fibres:
            assert f.i == -1
            assert f.j == -1

    def test_exclude_steel_area_approximates_bar_area(self, rectangular_beam_with_rebars):
        """Test that excluded area is approximately equal to total bar area."""
        mesh_excluded = FibreMesh(
            section=rectangular_beam_with_rebars,
            n_fibres_width=60,
            n_fibres_height=60,
            exclude_steel_area=True,
        )
        mesh_included = FibreMesh(
            section=rectangular_beam_with_rebars,
            n_fibres_width=60,
            n_fibres_height=60,
            exclude_steel_area=False,
        )

        area_excluded = sum(f.area for f in mesh_excluded.concrete_fibres)
        area_included = sum(f.area for f in mesh_included.concrete_fibres)
        removed = area_included - area_excluded

        steel_area = sum(
            len(g.positions) * g.rebar.area
            for g in rectangular_beam_with_rebars.rebar_groups
        )
        # Loose tolerance: circle approximation + discretization
        assert removed == pytest.approx(steel_area, rel=0.05)

    def test_n_fibres_rejects_float(self, rectangular_beam):
        """Test that float fibre counts are rejected."""
        with pytest.raises(TypeError):
            FibreMesh(section=rectangular_beam, n_fibres_width=9.9, n_fibres_height=20)
        with pytest.raises(TypeError):
            FibreMesh(section=rectangular_beam, n_fibres_width=10, n_fibres_height=20.5)

    def test_n_fibres_rejects_zero(self, rectangular_beam):
        """Test that zero fibre counts are rejected."""
        with pytest.raises(ValueError):
            FibreMesh(section=rectangular_beam, n_fibres_width=0, n_fibres_height=20)
        with pytest.raises(ValueError):
            FibreMesh(section=rectangular_beam, n_fibres_width=10, n_fibres_height=0)

    def test_n_fibres_rejects_negative(self, rectangular_beam):
        """Test that negative fibre counts are rejected."""
        with pytest.raises(ValueError):
            FibreMesh(section=rectangular_beam, n_fibres_width=-5, n_fibres_height=20)

    def test_repr(self, mesh_simple):
        """Test __repr__ method."""
        r = repr(mesh_simple)
        assert "FibreMesh" in r
        assert "concrete" in r
        assert "steel" in r
        assert str(mesh_simple.total_fibres) in r
