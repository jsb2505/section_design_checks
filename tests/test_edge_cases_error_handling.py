"""
Edge case and error handling tests for M-N interaction diagram.

Tests cover:
1. Zero loads (M=0, N=0) handling
2. Very small forces near zero
3. Extreme load magnitudes
4. Invalid section configurations
5. Boundary condition handling
6. Material limit violations
7. Numerical edge cases
"""

import numpy as np
import pytest

from materials.core.geometry import Point2D
from materials.reinforced_concrete.analysis.interaction_diagram import (
    MNInteractionDiagram,
    create_interaction_diagram,
)
from materials.reinforced_concrete.geometry import RebarGroup, create_rectangular_section
from materials.reinforced_concrete.materials import ConcreteMaterial, Rebar


@pytest.fixture
def standard_section():
    """Create a standard rectangular section with reinforcement."""
    section = create_rectangular_section(width=300, height=500, hook_ref=0)
    rebar_20 = Rebar(diameter=20, grade="B500B")

    bottom_positions = [Point2D(x=-50, y=-200), Point2D(x=50, y=-200)]
    bottom_group = RebarGroup(rebar=rebar_20, positions=bottom_positions)
    section.add_rebar_group(bottom_group)

    top_positions = [Point2D(x=-50, y=200), Point2D(x=50, y=200)]
    top_group = RebarGroup(rebar=rebar_20, positions=top_positions)
    section.add_rebar_group(top_group)

    return section


@pytest.fixture
def standard_diagram(standard_section):
    """Create a standard interaction diagram."""
    concrete = ConcreteMaterial(grade="C30/37")
    return MNInteractionDiagram(
        section=standard_section,
        concrete=concrete,
        use_characteristic=False,
        use_accidental=False,
    )


class TestZeroAndSmallLoads:
    """Test handling of zero and very small loads."""

    def test_zero_strain_gives_zero_forces(self, standard_diagram):
        """Test that zero strain gives zero (or near-zero) forces."""
        point = standard_diagram.calculate_point_from_end_strains(0.0, 0.0)

        # Should have essentially zero forces
        assert abs(point.N) < 0.1, f"Expected near-zero N, got {point.N}"
        assert abs(point.M) < 0.1, f"Expected near-zero M, got {point.M}"

    def test_very_small_positive_strains(self, standard_diagram):
        """Test very small positive (compression) strains."""
        eps_small = 1e-6  # Very small compression

        point = standard_diagram.calculate_point_from_end_strains(eps_small, eps_small)

        # Should have small but positive compression force
        assert point.N > 0, "Small compression strain should give positive N"
        assert abs(point.M) < 1.0, "Uniform small strain should give near-zero moment"

    def test_very_small_negative_strains(self, standard_diagram):
        """Test very small negative (tension) strains."""
        eps_small = -1e-6  # Very small tension

        point = standard_diagram.calculate_point_from_end_strains(eps_small, eps_small)

        # Should have small tension force (negative N)
        # Concrete has no tension capacity, so only steel contributes
        assert abs(point.N) < 100.0, "Small tension strain should give small forces"

    def test_solve_for_zero_target_loads(self, standard_diagram):
        """Test inverse solver with zero target loads."""
        M_target = 0.0
        N_target = 0.0

        eps_top, eps_bottom = standard_diagram.find_strains_for_MN(M_target, N_target)

        # Should find near-zero strains
        assert abs(eps_top) < 1e-3, f"Expected small eps_top for zero loads, got {eps_top}"
        assert abs(eps_bottom) < 1e-3, f"Expected small eps_bottom for zero loads, got {eps_bottom}"

        # Verify round-trip
        point = standard_diagram.calculate_point_from_end_strains(eps_top, eps_bottom)
        assert abs(point.N) < 1.0
        assert abs(point.M) < 1.0


class TestInvalidSectionConfiguration:
    """Test error handling for invalid section configurations."""

    def test_section_without_rebar_raises_error(self):
        """Test that section without rebars raises ValueError."""
        section = create_rectangular_section(width=300, height=500)
        concrete = ConcreteMaterial(grade="C30/37")

        with pytest.raises(ValueError, match="at least one rebar group"):
            MNInteractionDiagram(
                section=section,
                concrete=concrete,
                use_characteristic=False,
                use_accidental=False,
            )

    def test_confined_concrete_without_parameters_raises_error(self, standard_section):
        """Test that confined concrete without required params raises error."""
        concrete = ConcreteMaterial(grade="C30/37")

        with pytest.raises(ValueError, match="confinement_rho_s must be provided"):
            MNInteractionDiagram(
                section=standard_section,
                concrete=concrete,
                confined_concrete=True,
                confinement_rho_s=None,  # Missing required parameter
            )

    def test_invalid_confinement_rho_s_raises_error(self, standard_section):
        """Test that invalid confinement ratio raises error."""
        concrete = ConcreteMaterial(grade="C30/37")

        # Test rho_s > 0.1 (too high)
        with pytest.raises(ValueError, match="confinement_rho_s must be in"):
            MNInteractionDiagram(
                section=standard_section,
                concrete=concrete,
                confined_concrete=True,
                confinement_rho_s=0.15,  # > 0.1
                confinement_f_yh=500.0,
            )

        # Test rho_s <= 0 (non-positive)
        with pytest.raises(ValueError, match="confinement_rho_s must be in"):
            MNInteractionDiagram(
                section=standard_section,
                concrete=concrete,
                confined_concrete=True,
                confinement_rho_s=0.0,
                confinement_f_yh=500.0,
            )

    def test_invalid_confinement_f_yh_raises_error(self, standard_section):
        """Test that invalid confinement yield strength raises error."""
        concrete = ConcreteMaterial(grade="C30/37")

        with pytest.raises(ValueError, match="confinement_f_yh must be > 0"):
            MNInteractionDiagram(
                section=standard_section,
                concrete=concrete,
                confined_concrete=True,
                confinement_rho_s=0.02,
                confinement_f_yh=-100.0,  # Negative
            )


class TestMaterialLimitBehavior:
    """Test behavior at and beyond material strain limits."""

    def test_strain_beyond_ultimate_compression(self, standard_diagram):
        """Test calculation with strain beyond ultimate compression."""
        eps_cu = standard_diagram.concrete_model.get_ultimate_strain()

        # Strain beyond ultimate (should still compute, possibly with reduced stress)
        eps_beyond = eps_cu * 1.5

        point = standard_diagram.calculate_point_from_end_strains(eps_beyond, eps_beyond)

        # Should complete without error
        assert point.N is not None
        assert not np.isnan(point.N)
        assert not np.isinf(point.N)

    def test_strain_at_exact_ultimate(self, standard_diagram):
        """Test calculation at exactly ultimate strain."""
        eps_cu = standard_diagram.concrete_model.get_ultimate_strain()

        point = standard_diagram.calculate_point_from_end_strains(eps_cu, eps_cu)

        # Should be at or near maximum compression capacity
        assert point.N > 0, "Ultimate compression should give positive N"
        assert abs(point.M) < 10.0, "Uniform ultimate strain should give small moment"

    def test_steel_yielding_both_sides(self, standard_diagram):
        """Test when steel yields on both tension and compression sides."""
        eps_y = standard_diagram.steel_models[0].epsilon_y

        # One side in compression yield, other in tension yield
        eps_top = eps_y * 1.5  # Compression yield
        eps_bottom = -eps_y * 1.5  # Tension yield

        point = standard_diagram.calculate_point_from_end_strains(eps_top, eps_bottom)

        assert point.N is not None
        assert point.M is not None
        assert not np.isnan(point.N)
        assert not np.isnan(point.M)


class TestBoundaryConditions:
    """Test behavior at boundary conditions."""

    def test_pure_compression_boundary(self, standard_diagram):
        """Test pure compression at material limit."""
        eps_cu = standard_diagram.concrete_model.get_ultimate_strain()

        # Maximum compression, no bending
        point = standard_diagram.calculate_point_from_end_strains(eps_cu, eps_cu)

        assert point.N > 0, "Should have maximum compression"
        assert abs(point.M) < 1.0, "Should have minimal moment"

    def test_balanced_failure_boundary(self, standard_diagram):
        """Test at balanced failure condition."""
        eps_cu = standard_diagram.concrete_model.get_ultimate_strain()
        eps_y = standard_diagram.steel_models[0].epsilon_y

        # Balanced: concrete crushes, steel yields
        point = standard_diagram.calculate_point_from_end_strains(eps_cu, -eps_y)

        assert point.N >= 0, "Balanced failure should have compression or near-zero N"
        assert point.M != 0, "Balanced failure should have significant moment"

    def test_pure_bending_boundary(self, standard_diagram):
        """Test near pure bending (N ≈ 0)."""
        # Find strains that give N ≈ 0
        M_target = 50.0
        N_target = 0.0

        eps_top, eps_bottom = standard_diagram.find_strains_for_MN(M_target, N_target)
        point = standard_diagram.calculate_point_from_end_strains(eps_top, eps_bottom)

        assert abs(point.N) < 10.0, "Should have near-zero axial force"
        assert abs(point.M - M_target) < 1.0


class TestDiagramGeneration:
    """Test edge cases in diagram generation."""

    def test_generate_with_minimum_points(self, standard_diagram):
        """Test diagram generation with minimum number of points."""
        points = standard_diagram.generate_diagram_points(n_points=5)

        assert len(points) >= 5, "Should generate at least requested number of points"

        # All points should be valid
        for point in points:
            assert not np.isnan(point.N)
            assert not np.isnan(point.M)
            assert not np.isinf(point.N)
            assert not np.isinf(point.M)

    def test_generate_with_large_number_points(self, standard_diagram):
        """Test diagram generation with large number of points."""
        # This tests performance and memory handling
        points = standard_diagram.generate_diagram_points(n_points=200)

        assert len(points) > 0, "Should generate points"

        # Check envelope is closed
        assert abs(points[0].N - points[-1].N) < 10.0, "Envelope should be closed (N)"
        assert abs(points[0].M - points[-1].M) < 10.0, "Envelope should be closed (M)"

    def test_generate_diagram_is_ordered(self, standard_diagram):
        """Test that generated diagram points are properly ordered."""
        points = standard_diagram.generate_diagram_points(n_points=30)

        # Points should trace a continuous path (no jumps)
        for i in range(len(points) - 1):
            N_diff = abs(points[i+1].N - points[i].N)
            M_diff = abs(points[i+1].M - points[i].M)

            # Should not have huge jumps between consecutive points
            assert N_diff < 500.0, f"Large N jump at index {i}"
            assert M_diff < 200.0, f"Large M jump at index {i}"


class TestCapacityChecks:
    """Test capacity checking edge cases."""

    def test_get_capacity_with_zero_axial(self, standard_diagram):
        """Test get_capacity_fixed_n with N_Ed = 0."""
        N_cap, M_Rd_pos, M_Rd_neg = standard_diagram.get_capacity_fixed_n(N_Ed=0.0)

        assert M_Rd_pos > 0, "Should have positive moment capacity"
        assert M_Rd_neg < 0, "Should have negative moment capacity"
        assert abs(M_Rd_pos) > 10.0, "Moment capacity should be significant"
        assert abs(M_Rd_neg) > 10.0, "Moment capacity should be significant"

    def test_get_capacity_with_high_compression(self, standard_diagram):
        """Test get_capacity_fixed_n with high compression."""
        # High compression reduces moment capacity
        N_Ed = 800.0

        N_cap, M_Rd_pos, M_Rd_neg = standard_diagram.get_capacity_fixed_n(N_Ed=N_Ed)

        assert N_cap > 0, "Should find capacity level"
        # Moment capacity should exist (values can be positive or negative)
        assert M_Rd_pos >= 0, "Positive moment capacity should be non-negative"
        assert M_Rd_neg <= 0, "Negative moment capacity should be non-positive"
        # At high compression, moment capacity should be reduced
        assert abs(M_Rd_pos) < 500.0  # Reasonable upper bound
        assert abs(M_Rd_neg) < 500.0

    def test_capacity_check_inside_envelope(self, standard_diagram):
        """Test capacity check for point inside envelope."""
        # Get a point on the diagram
        points = standard_diagram.generate_diagram_points(n_points=20)
        test_point = points[len(points) // 2]  # Middle point

        # Scale down to get a point inside envelope
        N_Ed = test_point.N * 0.5
        M_Ed = test_point.M * 0.5

        # Get capacity at this N level
        N_cap, M_Rd_pos, M_Rd_neg = standard_diagram.get_capacity_fixed_n(N_Ed=N_Ed)

        # Our moment should be within capacity
        if M_Ed >= 0:
            assert abs(M_Ed) <= abs(M_Rd_pos) * 1.01, "Load should be within positive capacity"
        else:
            assert abs(M_Ed) <= abs(M_Rd_neg) * 1.01, "Load should be within negative capacity"

    def test_capacity_at_zero_loads(self, standard_diagram):
        """Test capacity check at zero loads."""
        N_cap, M_Rd_pos, M_Rd_neg = standard_diagram.get_capacity_fixed_n(N_Ed=0.0)

        # Should have full bending capacity at zero axial load
        assert abs(M_Rd_pos) > 50.0, "Should have significant positive moment capacity"
        assert abs(M_Rd_neg) > 50.0, "Should have significant negative moment capacity"


class TestNumericalEdgeCases:
    """Test numerical edge cases."""

    def test_identical_strain_inputs(self, standard_diagram):
        """Test with many calls using identical strains."""
        # Should give consistent results
        eps_top = 0.001
        eps_bottom = -0.002

        results = []
        for _ in range(10):
            point = standard_diagram.calculate_point_from_end_strains(eps_top, eps_bottom)
            results.append((point.N, point.M))

        # All results should be identical
        for i in range(1, len(results)):
            assert abs(results[i][0] - results[0][0]) < 1e-10, "N should be identical"
            assert abs(results[i][1] - results[0][1]) < 1e-10, "M should be identical"

    def test_alternating_positive_negative_strains(self, standard_diagram):
        """Test rapidly alternating between positive and negative strains."""
        test_cases = [
            (0.002, -0.002),
            (-0.002, 0.002),
            (0.001, -0.001),
            (-0.001, 0.001),
        ]

        for eps_top, eps_bottom in test_cases:
            point = standard_diagram.calculate_point_from_end_strains(eps_top, eps_bottom)

            assert not np.isnan(point.N), f"NaN result for strains ({eps_top}, {eps_bottom})"
            assert not np.isnan(point.M)
            assert not np.isinf(point.N)
            assert not np.isinf(point.M)

    def test_strain_order_independence_for_solver(self, standard_diagram):
        """Test that solver works regardless of which strain is larger."""
        M_target = 60.0
        N_target = 200.0

        # Solve (should work regardless of internal strain ordering)
        eps_top, eps_bottom = standard_diagram.find_strains_for_MN(M_target, N_target)

        point = standard_diagram.calculate_point_from_end_strains(eps_top, eps_bottom)

        assert abs(point.M - M_target) < 1.0
        assert abs(point.N - N_target) < 1.0


class TestFactoryFunction:
    """Test the create_interaction_diagram factory function."""

    def test_factory_creates_valid_diagram(self, standard_section):
        """Test factory function creates valid diagram."""
        concrete = ConcreteMaterial(grade="C30/37")

        diagram = create_interaction_diagram(
            section=standard_section,
            concrete=concrete,
        )

        assert isinstance(diagram, MNInteractionDiagram)
        assert diagram.section is standard_section
        assert diagram.concrete is concrete

    def test_factory_with_all_options(self, standard_section):
        """Test factory function with optional parameters."""
        concrete = ConcreteMaterial(grade="C30/37")

        # Test with characteristic (cannot combine with accidental)
        diagram = create_interaction_diagram(
            section=standard_section,
            concrete=concrete,
            use_characteristic=True,
            use_accidental=False,  # Cannot be both True
            confined_concrete=True,
            confinement_rho_s=0.02,
            confinement_f_yh=500.0,
            tension_stiffening=True,
        )

        # Verify diagram was created successfully
        assert isinstance(diagram, MNInteractionDiagram)
        # Parameters are passed to internal models, verify diagram is functional
        points = diagram.generate_diagram_points(n_points=10)
        assert len(points) > 0, "Should generate points with all options enabled"

    def test_factory_with_invalid_section_raises_error(self):
        """Test factory raises error for invalid section."""
        section = create_rectangular_section(300, 500)  # No rebars
        concrete = ConcreteMaterial(grade="C30/37")

        with pytest.raises(ValueError, match="at least one rebar group"):
            create_interaction_diagram(section=section, concrete=concrete)
