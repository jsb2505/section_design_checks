"""
Tests for reinforced_concrete.geometry.fiber_mesh module.
"""

import pytest
import numpy as np
from materials.reinforced_concrete.geometry import FiberMesh, create_linear_rebar_layer
from materials.core.geometry import Point2D


class TestFiberMesh:
    """Tests for FiberMesh class."""

    @pytest.fixture
    def mesh_simple(self, rectangular_beam_with_rebars):
        """Simple fiber mesh."""
        return FiberMesh(
            section=rectangular_beam_with_rebars,
            n_fibers_width=10,
            n_fibers_height=20,
        )

    def test_create_mesh(self, mesh_simple):
        """Test creating fiber mesh."""
        assert mesh_simple.n_concrete_fibers > 0
        assert mesh_simple.n_steel_fibers == 3  # 3 rebars
        assert mesh_simple.total_fibers == mesh_simple.n_concrete_fibers + 3

    def test_concrete_fibers_generated(self, mesh_simple):
        """Test that concrete fibers are generated."""
        assert len(mesh_simple.concrete_fibers) > 0

        for fiber in mesh_simple.concrete_fibers:
            assert fiber.material_type == "concrete"
            assert fiber.area > 0
            assert fiber.x >= 0
            assert fiber.y >= 0

    def test_steel_fibers_generated(self, mesh_simple):
        """Test that steel fibers are generated."""
        assert len(mesh_simple.steel_fibers) == 3

        for fiber in mesh_simple.steel_fibers:
            assert fiber.material_type == "steel"
            assert fiber.area > 0

    def test_all_fibers(self, mesh_simple):
        """Test all_fibers property."""
        all_f = mesh_simple.all_fibers
        assert len(all_f) == mesh_simple.total_fibers

    def test_fiber_resolution(self, rectangular_beam_with_rebars):
        """Test that higher resolution creates more fibers."""
        mesh_coarse = FiberMesh(
            section=rectangular_beam_with_rebars,
            n_fibers_width=5,
            n_fibers_height=10,
        )
        mesh_fine = FiberMesh(
            section=rectangular_beam_with_rebars,
            n_fibers_width=20,
            n_fibers_height=40,
        )

        assert mesh_fine.n_concrete_fibers > mesh_coarse.n_concrete_fibers

    def test_exclude_steel_area(self, rectangular_beam_with_rebars):
        """Test excluding steel area from concrete fibers."""
        mesh_excluded = FiberMesh(
            section=rectangular_beam_with_rebars,
            n_fibers_width=10,
            n_fibers_height=20,
            exclude_steel_area=True,
        )
        mesh_included = FiberMesh(
            section=rectangular_beam_with_rebars,
            n_fibers_width=10,
            n_fibers_height=20,
            exclude_steel_area=False,
        )

        # Total concrete area should be less when steel is excluded
        concrete_area_excluded = sum(f.area for f in mesh_excluded.concrete_fibers)
        concrete_area_included = sum(f.area for f in mesh_included.concrete_fibers)

        assert concrete_area_excluded <= concrete_area_included

    def test_get_fiber_arrays(self, mesh_simple):
        """Test getting fiber data as arrays."""
        x, y, area, material_type, material_index = mesh_simple.get_fiber_arrays()

        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert isinstance(area, np.ndarray)
        assert isinstance(material_type, np.ndarray)
        assert isinstance(material_index, np.ndarray)

        assert len(x) == mesh_simple.total_fibers
        assert len(y) == mesh_simple.total_fibers
        assert len(area) == mesh_simple.total_fibers

    def test_fiber_arrays_properties(self, mesh_simple):
        """Test fiber array properties."""
        x, y, area, material_type, material_index = mesh_simple.get_fiber_arrays()

        # All areas should be positive
        assert np.all(area > 0)

        # Material types should be 'concrete' or 'steel'
        assert np.all((material_type == 'concrete') | (material_type == 'steel'))

        # Count material types
        n_concrete = np.sum(material_type == 'concrete')
        n_steel = np.sum(material_type == 'steel')

        assert n_concrete == mesh_simple.n_concrete_fibers
        assert n_steel == mesh_simple.n_steel_fibers

    def test_calculate_section_forces(self, mesh_simple):
        """Test calculating section forces from fiber stresses."""
        # Create dummy strains and stresses
        n_fibers = mesh_simple.total_fibers
        strains = np.ones(n_fibers) * 0.001
        stresses = np.ones(n_fibers) * 20.0  # 20 MPa constant

        N, M = mesh_simple.calculate_section_forces(strains, stresses)

        # N should be positive (compression)
        assert N > 0

        # M depends on distribution, but should be calculable
        assert isinstance(M, float)

    def test_calculate_forces_uniform_stress(self, rectangular_beam):
        """Test force calculation with uniform stress."""
        # Simple section with no rebars
        mesh = FiberMesh(
            section=rectangular_beam,
            n_fibers_width=10,
            n_fibers_height=10,
        )

        # Uniform compression stress of 10 MPa
        n_fibers = mesh.total_fibers
        strains = np.ones(n_fibers) * 0.001
        stresses = np.ones(n_fibers) * 10.0

        N, M = mesh.calculate_section_forces(strains, stresses)

        # N ≈ σ · A = 10 MPa · 150,000 mm² = 1,500,000 N = 1,500 kN
        # (approximate due to mesh discretization)
        expected_N = 10.0 * rectangular_beam.get_area() / 1000  # Convert to kN
        assert N == pytest.approx(expected_N, rel=0.05)

        # M should be ~0 for uniform stress about centroid
        assert abs(M) < 10.0  # Small compared to N

    def test_mesh_conservation_of_area(self, rectangular_beam):
        """Test that total fiber area approximately equals section area."""
        mesh = FiberMesh(
            section=rectangular_beam,
            n_fibers_width=20,
            n_fibers_height=30,
            exclude_steel_area=False,
        )

        total_fiber_area = sum(f.area for f in mesh.concrete_fibers)

        # Should be close to section area (within 2% due to discretization)
        assert total_fiber_area == pytest.approx(
            rectangular_beam.get_area(),
            rel=0.02
        )

    def test_steel_fiber_positions(self, rectangular_beam_with_rebars):
        """Test that steel fibers are at correct positions."""
        mesh = FiberMesh(
            section=rectangular_beam_with_rebars,
            n_fibers_width=10,
            n_fibers_height=20,
        )

        # Get steel fiber positions
        steel_positions = [(f.x, f.y) for f in mesh.steel_fibers]

        # Should have 3 steel fibers
        assert len(steel_positions) == 3

        # Check they're at expected y-coordinate (50mm from bottom)
        for x, y in steel_positions:
            assert y == pytest.approx(50.0)

    def test_repr(self, mesh_simple):
        """Test __repr__ method."""
        r = repr(mesh_simple)
        assert "FiberMesh" in r
        assert "concrete" in r
        assert "steel" in r
        assert str(mesh_simple.total_fibers) in r
