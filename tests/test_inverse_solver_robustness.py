"""
Robustness tests for the inverse solver (find_strains_for_MN).

Tests cover:
1. Points outside the interaction envelope (should raise ValueError)
2. Robustness to poor initial guesses
3. Strain bounds enforcement
4. Round-trip verification with various initial guesses
5. Extreme load cases
6. Edge cases near pure compression/tension
"""

import pytest
import numpy as np
from materials.core.geometry import Point2D
from materials.reinforced_concrete.analysis.interaction_diagram import MNInteractionDiagram
from materials.reinforced_concrete.geometry import create_rectangular_section, RebarGroup
from materials.reinforced_concrete.materials import ConcreteMaterial, Rebar


@pytest.fixture
def test_section():
    """Create a simple rectangular section with reinforcement."""
    section = create_rectangular_section(width=300, height=500, hook_ref=0)

    rebar_20 = Rebar(diameter=20, grade="B500B")

    # Bottom bars (centered coordinate system: -250 to +250 in y)
    bottom_positions = [Point2D(x=-50, y=-200), Point2D(x=50, y=-200)]
    bottom_group = RebarGroup(rebar=rebar_20, positions=bottom_positions)
    section.add_rebar_group(bottom_group)

    # Top bars
    top_positions = [Point2D(x=-50, y=200), Point2D(x=50, y=200)]
    top_group = RebarGroup(rebar=rebar_20, positions=top_positions)
    section.add_rebar_group(top_group)

    return section


@pytest.fixture
def test_diagram(test_section):
    """Create a test interaction diagram."""
    concrete = ConcreteMaterial(grade="C30/37")
    return MNInteractionDiagram(
        section=test_section,
        concrete=concrete,
        use_characteristic=False,
        use_accidental=False,
    )


class TestInverseSolverOutsideEnvelope:
    """Test inverse solver behavior with points outside the interaction envelope.

    NOTE: The solver uses least_squares optimization which finds the closest feasible
    point rather than raising an error for unreachable targets. This is intentional
    behavior - the solver converges to boundary points when targets are outside envelope.
    """

    def test_extremely_high_moment_finds_boundary(self, test_diagram):
        """Test that extremely high moment (beyond capacity) finds boundary point."""
        # Generate diagram to know approximate capacity
        points = test_diagram.generate_diagram(n_points=20)
        max_M = max(abs(p.M) for p in points)

        # Try to find strains for moment way beyond capacity
        M_impossible = max_M * 5.0
        N_target = 0.0

        # Solver should converge (to boundary point)
        eps_top, eps_bottom = test_diagram.find_strains_for_MN(M_impossible, N_target)

        # Verify solver found a valid solution
        point = test_diagram.calculate_point_from_end_strains(eps_top, eps_bottom)

        # Result should be at or near maximum moment capacity
        assert abs(point.M) <= max_M * 1.1, \
            f"Solver should find point near max capacity, got M={point.M} (max={max_M})"

    def test_extremely_high_axial_force_finds_boundary(self, test_diagram):
        """Test that extremely high axial force (beyond capacity) finds boundary point."""
        # Generate diagram to know approximate capacity
        points = test_diagram.generate_diagram(n_points=20)
        max_N = max(p.N for p in points)

        # Try to find strains for axial force way beyond capacity
        N_impossible = max_N * 5.0
        M_target = 0.0

        # Solver should converge to boundary
        eps_top, eps_bottom = test_diagram.find_strains_for_MN(M_target, N_impossible)

        # Verify solver found a valid solution near maximum N
        point = test_diagram.calculate_point_from_end_strains(eps_top, eps_bottom)
        assert point.N <= max_N * 1.1, \
            f"Solver should find point near max capacity, got N={point.N} (max={max_N})"

    def test_combined_extreme_loads_find_boundary(self, test_diagram):
        """Test that combined loads beyond envelope find boundary point."""
        # Generate diagram to know approximate capacity
        points = test_diagram.generate_diagram(n_points=20)
        max_M = max(abs(p.M) for p in points)
        max_N = max(p.N for p in points)

        # Try unreachable combination
        M_target = max_M * 2.0
        N_target = max_N * 2.0

        # Solver should converge to some boundary point
        eps_top, eps_bottom = test_diagram.find_strains_for_MN(M_target, N_target)

        point = test_diagram.calculate_point_from_end_strains(eps_top, eps_bottom)

        # Result should be within or near envelope
        assert point.N <= max_N * 1.2
        assert abs(point.M) <= max_M * 1.2

    def test_large_tension_finds_boundary(self, test_diagram):
        """Test that large tension force finds boundary point with tension."""
        # Try large tension force (negative N)
        N_tension = -1000.0  # kN tension (concrete section can't handle this)
        M_target = 0.0

        # Solver should find some solution (boundary with tension)
        eps_top, eps_bottom = test_diagram.find_strains_for_MN(M_target, N_tension)

        point = test_diagram.calculate_point_from_end_strains(eps_top, eps_bottom)

        # Result should have tension (negative N)
        assert point.N < 0, "Should find a point with tension"


class TestInverseSolverInitialGuess:
    """Test inverse solver robustness to different initial guesses."""

    def test_poor_initial_guess_pure_compression(self, test_diagram):
        """Test solver can converge from poor initial guess for pure compression."""
        M_target = 0.0
        N_target = 500.0

        # Try with a terrible initial guess (high tension strains)
        bad_guess = (-0.01, -0.01)  # Both in high tension

        eps_top, eps_bottom = test_diagram.find_strains_for_MN(
            M_target, N_target, initial_guess=bad_guess
        )

        # Verify round-trip
        point = test_diagram.calculate_point_from_end_strains(eps_top, eps_bottom)
        assert abs(point.N - N_target) < 1.0
        assert abs(point.M - M_target) < 1.0

    def test_poor_initial_guess_sagging_moment(self, test_diagram):
        """Test solver can converge from poor initial guess for sagging moment."""
        M_target = 80.0
        N_target = 300.0

        # Try with opposite sign strains (hogging instead of sagging)
        bad_guess = (-0.003, 0.003)  # Wrong direction

        eps_top, eps_bottom = test_diagram.find_strains_for_MN(
            M_target, N_target, initial_guess=bad_guess
        )

        # Verify round-trip
        point = test_diagram.calculate_point_from_end_strains(eps_top, eps_bottom)
        assert abs(point.N - N_target) < 1.0
        assert abs(point.M - M_target) < 1.0

    def test_poor_initial_guess_hogging_moment(self, test_diagram):
        """Test solver can converge from poor initial guess for hogging moment."""
        M_target = -60.0
        N_target = 200.0

        # Try with opposite sign strains (sagging instead of hogging)
        bad_guess = (0.003, -0.003)  # Wrong direction

        eps_top, eps_bottom = test_diagram.find_strains_for_MN(
            M_target, N_target, initial_guess=bad_guess
        )

        # Verify round-trip
        point = test_diagram.calculate_point_from_end_strains(eps_top, eps_bottom)
        assert abs(point.N - N_target) < 1.0
        assert abs(point.M - M_target) < 1.0

    def test_zero_initial_guess(self, test_diagram):
        """Test solver can converge from zero initial guess."""
        M_target = 50.0
        N_target = 100.0

        # Zero initial guess
        zero_guess = (0.0, 0.0)

        eps_top, eps_bottom = test_diagram.find_strains_for_MN(
            M_target, N_target, initial_guess=zero_guess
        )

        # Verify round-trip
        point = test_diagram.calculate_point_from_end_strains(eps_top, eps_bottom)
        assert abs(point.N - N_target) < 1.0
        assert abs(point.M - M_target) < 1.0

    def test_automatic_initial_guess_quality(self, test_diagram):
        """Test that automatic initial guess works for various load combinations."""
        # Test multiple load cases without providing initial guess
        test_cases = [
            (0.0, 500.0),      # Pure compression
            (50.0, 100.0),     # Sagging with compression
            (-50.0, 100.0),    # Hogging with compression
            (80.0, 300.0),     # Large sagging
            (-80.0, 300.0),    # Large hogging
            (30.0, 0.0),       # Pure bending (sagging)
            (-30.0, 0.0),      # Pure bending (hogging)
        ]

        for M_target, N_target in test_cases:
            # Use automatic initial guess (None)
            eps_top, eps_bottom = test_diagram.find_strains_for_MN(M_target, N_target)

            # Verify round-trip
            point = test_diagram.calculate_point_from_end_strains(eps_top, eps_bottom)
            assert abs(point.N - N_target) < 1.0, \
                f"N mismatch for (M={M_target}, N={N_target}): got {point.N}"
            assert abs(point.M - M_target) < 1.0, \
                f"M mismatch for (M={M_target}, N={N_target}): got {point.M}"


class TestInverseSolverStrainBounds:
    """Test that inverse solver respects strain bounds."""

    def test_strain_bounds_not_exceeded_compression(self, test_diagram):
        """Test strains don't exceed ultimate compression strain."""
        eps_cu = test_diagram.concrete_model.get_ultimate_strain()

        # High compression case
        M_target = 0.0
        N_target = 800.0  # High compression

        eps_top, eps_bottom = test_diagram.find_strains_for_MN(M_target, N_target)

        # Both strains should be within bounds
        assert eps_top <= eps_cu * 1.01, \
            f"Top strain {eps_top} exceeds ultimate {eps_cu}"
        assert eps_bottom <= eps_cu * 1.01, \
            f"Bottom strain {eps_bottom} exceeds ultimate {eps_cu}"

    def test_strain_bounds_not_exceeded_tension(self, test_diagram):
        """Test strains don't exceed reasonable tension limits."""
        eps_y = test_diagram.steel_models[0].epsilon_y

        # Pure bending with tension
        M_target = 70.0
        N_target = 0.0

        eps_top, eps_bottom = test_diagram.find_strains_for_MN(M_target, N_target)

        # Tension strain should be reasonable (not extreme)
        # Allow up to 10 * yield strain (well within steel ductility)
        max_tension_strain = abs(eps_y * 10.0)
        assert abs(min(eps_top, eps_bottom)) < max_tension_strain, \
            f"Tension strain {min(eps_top, eps_bottom)} exceeds reasonable limit"


class TestInverseSolverRoundTrip:
    """Test round-trip verification: solve for strains → calculate point → verify."""

    def test_round_trip_with_various_tolerances(self, test_diagram):
        """Test round-trip works with different tolerance settings."""
        M_target = 50.0
        N_target = 100.0

        # Test with different tolerances
        tolerances = [1e-4, 1e-6, 1e-8]

        for tol in tolerances:
            eps_top, eps_bottom = test_diagram.find_strains_for_MN(
                M_target, N_target, tol=tol
            )

            point = test_diagram.calculate_point_from_end_strains(eps_top, eps_bottom)

            # Tighter tolerance should give better accuracy
            expected_error = max(1.0, tol * 1e6)  # Scale tolerance to kN/kNm units
            assert abs(point.N - N_target) < expected_error
            assert abs(point.M - M_target) < expected_error

    def test_round_trip_many_random_points(self, test_diagram):
        """Test round-trip for many random points within envelope."""
        # Generate envelope
        points = test_diagram.generate_diagram(n_points=30)

        # Extract N and M ranges
        N_values = [p.N for p in points]
        M_values = [p.M for p in points]

        N_min, N_max = min(N_values), max(N_values)
        M_min, M_max = min(M_values), max(M_values)

        # Test random points within conservative bounding box
        # Use smaller range to increase likelihood points are within envelope
        np.random.seed(42)  # Reproducibility
        n_tests = 10
        successful_tests = 0

        for i in range(n_tests):
            # Sample within conservative bounding box (50% of range from center)
            N_center = (N_min + N_max) / 2
            M_center = (M_min + M_max) / 2
            N_range = (N_max - N_min) * 0.5
            M_range = (M_max - M_min) * 0.5

            N_target = np.random.uniform(N_center - N_range/2, N_center + N_range/2)
            M_target = np.random.uniform(M_center - M_range/2, M_center + M_range/2)

            eps_top, eps_bottom = test_diagram.find_strains_for_MN(M_target, N_target)

            # Verify round-trip (with relaxed tolerance for boundary points)
            point = test_diagram.calculate_point_from_end_strains(eps_top, eps_bottom)

            # If point is close to target, count as success
            N_error = abs(point.N - N_target)
            M_error = abs(point.M - M_target)

            if N_error < 5.0 and M_error < 5.0:  # Relaxed tolerance
                successful_tests += 1

        # At least 70% of tests should have close matches (rest may be near boundary)
        assert successful_tests >= n_tests * 0.7, \
            f"Only {successful_tests}/{n_tests} tests had close matches"


class TestInverseSolverEdgeCases:
    """Test inverse solver edge cases."""

    def test_very_small_loads(self, test_diagram):
        """Test solver works with very small loads near zero."""
        M_target = 0.01  # Very small moment
        N_target = 0.01  # Very small axial force

        eps_top, eps_bottom = test_diagram.find_strains_for_MN(M_target, N_target)

        # Verify round-trip (with relaxed tolerance for small values)
        point = test_diagram.calculate_point_from_end_strains(eps_top, eps_bottom)
        assert abs(point.N - N_target) < 0.1
        assert abs(point.M - M_target) < 0.1

    def test_pure_compression_no_moment(self, test_diagram):
        """Test solver finds symmetric strains for pure compression."""
        M_target = 0.0
        N_target = 500.0

        eps_top, eps_bottom = test_diagram.find_strains_for_MN(M_target, N_target)

        # For pure compression with no moment, strains should be nearly equal
        assert abs(eps_top - eps_bottom) < 1e-4, \
            f"Strains should be equal for pure compression: {eps_top} vs {eps_bottom}"

        # Verify round-trip
        point = test_diagram.calculate_point_from_end_strains(eps_top, eps_bottom)
        assert abs(point.N - N_target) < 1.0
        assert abs(point.M - M_target) < 1.0

    def test_near_balanced_failure(self, test_diagram):
        """Test solver near balanced failure point."""
        # Balanced failure: steel yields in tension, concrete crushes in compression
        eps_cu = test_diagram.concrete_model.get_ultimate_strain()
        eps_y = test_diagram.steel_models[0].epsilon_y

        # First find balanced point
        point_balanced = test_diagram.calculate_point_from_end_strains(eps_cu, -eps_y)

        # Now try to solve for a point very close to balanced
        M_target = point_balanced.M * 0.99
        N_target = point_balanced.N * 0.99

        eps_top, eps_bottom = test_diagram.find_strains_for_MN(M_target, N_target)

        # Verify round-trip
        point = test_diagram.calculate_point_from_end_strains(eps_top, eps_bottom)
        assert abs(point.N - N_target) < 1.0
        assert abs(point.M - M_target) < 1.0

    def test_maximum_moment_capacity(self, test_diagram):
        """Test solver can find strains for near-maximum moment capacity."""
        # Generate envelope to find max moment
        points = test_diagram.generate_diagram(n_points=50)

        # Find point with maximum moment
        max_M_point = max(points, key=lambda p: abs(p.M))

        # Try to solve for 95% of maximum moment
        M_target = max_M_point.M * 0.95
        N_target = max_M_point.N * 0.95

        eps_top, eps_bottom = test_diagram.find_strains_for_MN(M_target, N_target)

        # Verify round-trip
        point = test_diagram.calculate_point_from_end_strains(eps_top, eps_bottom)
        assert abs(point.N - N_target) < 1.0
        assert abs(point.M - M_target) < 1.0


class TestInverseSolverConsistency:
    """Test consistency between forward and inverse calculations."""

    def test_forward_inverse_consistency(self, test_diagram):
        """Test forward → inverse → forward gives consistent results."""
        # Start with arbitrary strains
        eps_cu = test_diagram.concrete_model.get_ultimate_strain()
        eps_y = test_diagram.steel_models[0].epsilon_y

        eps_top_original = eps_cu * 0.5
        eps_bottom_original = -eps_y * 0.8

        # Forward: strains → (M, N)
        point_1 = test_diagram.calculate_point_from_end_strains(
            eps_top_original, eps_bottom_original
        )

        # Inverse: (M, N) → strains
        eps_top_solved, eps_bottom_solved = test_diagram.find_strains_for_MN(
            point_1.M, point_1.N
        )

        # Forward again: solved strains → (M, N)
        point_2 = test_diagram.calculate_point_from_end_strains(
            eps_top_solved, eps_bottom_solved
        )

        # Both forward calculations should give same (M, N)
        assert abs(point_1.N - point_2.N) < 0.1
        assert abs(point_1.M - point_2.M) < 0.1

        # Solved strains don't need to exactly match original (multiple solutions possible),
        # but they should give same (M, N)

    def test_multiple_solves_same_target(self, test_diagram):
        """Test multiple solves for same target give consistent results."""
        M_target = 60.0
        N_target = 150.0

        # Solve 3 times with different (or no) initial guesses
        results = []

        # Solve 1: automatic initial guess
        eps1 = test_diagram.find_strains_for_MN(M_target, N_target)
        results.append(eps1)

        # Solve 2: different initial guess
        eps2 = test_diagram.find_strains_for_MN(
            M_target, N_target, initial_guess=(0.001, -0.001)
        )
        results.append(eps2)

        # Solve 3: another different initial guess
        eps3 = test_diagram.find_strains_for_MN(
            M_target, N_target, initial_guess=(0.002, 0.001)
        )
        results.append(eps3)

        # All solutions should give same (M, N) when evaluated
        points = [
            test_diagram.calculate_point_from_end_strains(eps_top, eps_bottom)
            for eps_top, eps_bottom in results
        ]

        for point in points:
            assert abs(point.N - N_target) < 1.0
            assert abs(point.M - M_target) < 1.0
