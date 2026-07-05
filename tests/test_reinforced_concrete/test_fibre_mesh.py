"""
Tests for reinforced_concrete.geometry.fibre_mesh module.
"""

import pytest
import numpy as np
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

    def test_create_mesh(self, mesh_simple):
        """Test creating fibre mesh."""
        assert mesh_simple.n_concrete_fibres > 0
        assert mesh_simple.n_steel_fibres == 3  # 3 rebars
        assert mesh_simple.total_fibres == mesh_simple.n_concrete_fibres + 3

    def test_concrete_fibres_generated(self, mesh_simple):
        """Test that concrete fibres are generated."""
        assert len(mesh_simple.concrete_fibres) > 0

        for fibre in mesh_simple.concrete_fibres:
            assert fibre.material_type == "concrete"
            assert fibre.area > 0
            assert fibre.x >= 0
            assert fibre.y >= 0

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

    def test_repr(self, mesh_simple):
        """Test __repr__ method."""
        r = repr(mesh_simple)
        assert "FibreMesh" in r
        assert "concrete" in r
        assert "steel" in r
        assert str(mesh_simple.total_fibres) in r
