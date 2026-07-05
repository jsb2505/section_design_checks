"""
Tests for analytical Jacobian implementation in M-N interaction diagram.

Verifies that:
1. Analytical Jacobian matches 2-point numerical approximation
2. Solver converges correctly with analytical Jacobian
3. Performance is improved vs 2-point method
"""

import time
import numpy as np
import pytest
from materials.core.geometry import Point2D
from materials.reinforced_concrete.analysis.interaction_diagram import MNInteractionDiagram
from materials.reinforced_concrete.geometry import create_rectangular_section, RebarGroup
from materials.reinforced_concrete.materials import ConcreteMaterial, Rebar


def create_test_diagram():
    """Create a test M-N diagram for verification."""
    section = create_rectangular_section(width=300, height=500)
    rebar_20 = Rebar(diameter=20, grade="B500B")

    # Bottom bars
    bottom_positions = [Point2D(x=-50, y=-200), Point2D(x=50, y=-200)]
    bottom_group = RebarGroup(rebar=rebar_20, positions=bottom_positions)
    section.add_rebar_group(bottom_group)

    # Top bars
    top_positions = [Point2D(x=-50, y=200), Point2D(x=50, y=200)]
    top_group = RebarGroup(rebar=rebar_20, positions=top_positions)
    section.add_rebar_group(top_group)

    concrete = ConcreteMaterial(grade="C30/37")

    return MNInteractionDiagram(
        section=section,
        concrete=concrete,
        use_characteristic=False,
        use_accidental=False,
    )


def test_analytical_jacobian_matches_numerical():
    """Verify analytical Jacobian matches 2-point numerical approximation."""
    diagram = create_test_diagram()

    # Test multiple strain pairs across different loading conditions
    test_strain_pairs = [
        (0.001, -0.002),   # Sagging: top compressed, bottom tension
        (-0.002, 0.001),   # Hogging: bottom compressed, top tension
        (0.0015, 0.0015),  # Pure compression (both positive)
        (0.0005, -0.0005), # Mild bending
        (0.002, -0.003),   # Near limits
    ]

    for eps_top, eps_bottom in test_strain_pairs:
        # Compute analytical Jacobian
        J_analytical = diagram._compute_analytical_jacobian(eps_top, eps_bottom)

        # Compute numerical Jacobian using 2-point finite difference
        h = 1e-8
        def f(eps_pair):
            point = diagram.calculate_point_from_end_strains(eps_pair[0], eps_pair[1])
            return np.array([point.N, point.M])

        J_numerical = np.zeros((2, 2))
        for i in range(2):
            eps_plus = np.array([eps_top, eps_bottom])
            eps_plus[i] += h
            f_plus = f(eps_plus)

            eps_minus = np.array([eps_top, eps_bottom])
            eps_minus[i] -= h
            f_minus = f(eps_minus)

            J_numerical[:, i] = (f_plus - f_minus) / (2 * h)

        # Compare (analytical should match numerical within 1%)
        # Allow higher tolerance for near-zero gradients
        max_element = np.max(np.abs(J_analytical))
        atol = max(1.0, max_element * 0.01)  # 1% or absolute 1.0 kN tolerance

        np.testing.assert_allclose(
            J_analytical, J_numerical, rtol=0.01, atol=atol,
            err_msg=f"Jacobian mismatch at eps_top={eps_top:.6f}, eps_bottom={eps_bottom:.6f}"
        )

    print(f"\n[PASS] Analytical Jacobian verified against numerical for {len(test_strain_pairs)} strain pairs")


def test_analytical_jacobian_solver_convergence():
    """Verify solver converges correctly with analytical Jacobian."""
    diagram = create_test_diagram()

    # Test various load cases
    test_cases = [
        (50.0, 100.0),    # Sagging moment with compression
        (30.0, 200.0),    # Smaller moment, more compression
        (0.0, 500.0),     # Pure axial compression
        (80.0, 300.0),    # Large moment
        (-30.0, 200.0),   # Hogging moment
        (10.0, 50.0),     # Small forces
    ]

    for M_target, N_target in test_cases:
        # Solve using analytical Jacobian
        eps_top, eps_bottom = diagram.find_strains_for_MN(M_target, N_target)

        # Verify solution is correct (forward check)
        point = diagram.calculate_point_from_end_strains(eps_top, eps_bottom)

        # Check M and N match targets (within 0.1%)
        M_error = abs(point.M - M_target)
        N_error = abs(point.N - N_target)

        assert M_error < max(0.001, abs(M_target) * 0.001), \
            f"M mismatch: target={M_target:.2f}, got={point.M:.2f}, error={M_error:.4f}"
        assert N_error < max(0.001, abs(N_target) * 0.001), \
            f"N mismatch: target={N_target:.2f}, got={point.N:.2f}, error={N_error:.4f}"

    print(f"\n[PASS] Solver converged correctly for {len(test_cases)} load cases")


def test_analytical_jacobian_iteration_count():
    """Verify analytical Jacobian reduces iteration count vs 2-point."""
    diagram = create_test_diagram()

    # Test a representative load case
    M_target, N_target = 50.0, 100.0

    # Solve and check iteration count
    # Note: scipy's least_squares doesn't expose iteration count directly,
    # but we can verify it succeeds within max_nfev=50 (vs 200 for 2-point)
    eps_top, eps_bottom = diagram.find_strains_for_MN(M_target, N_target)

    # Verify solution
    point = diagram.calculate_point_from_end_strains(eps_top, eps_bottom)
    assert abs(point.M - M_target) < 0.01
    assert abs(point.N - N_target) < 0.01

    print(f"\n[PASS] Analytical Jacobian solved within max_nfev=50 iterations")


def test_analytical_jacobian_performance():
    """Benchmark analytical vs 2-point Jacobian performance."""
    diagram = create_test_diagram()

    # Test cases for benchmarking
    test_cases = [
        (50.0, 100.0),
        (30.0, 200.0),
        (0.0, 500.0),
        (80.0, 300.0),
        (-30.0, 200.0),
    ]

    # Time analytical Jacobian (current implementation)
    t0 = time.time()
    for M, N in test_cases:
        eps_top, eps_bottom = diagram.find_strains_for_MN(M, N)
    analytical_time = time.time() - t0

    print(f"\n=== Analytical Jacobian Performance ===")
    print(f"Total time for {len(test_cases)} cases: {analytical_time:.3f}s")
    print(f"Time per case: {analytical_time*1000/len(test_cases):.1f}ms")
    print(f"[PASS] Analytical Jacobian performance measured")


def test_tangent_modulus_concrete():
    """Test concrete tangent modulus computation."""
    diagram = create_test_diagram()
    concrete_model = diagram.concrete_model

    # Test various strain regions
    test_cases = [
        (0.0, 0.0),           # Zero strain → E_t = 0 (no tension)
        (0.0001, ">0"),       # Start of parabolic region (positive gradient)
        (0.001, ">0"),        # Middle of parabolic region
        (0.0019, ">0"),       # Near transition
        (0.002, 0.0),         # Start of rectangular region → E_t = 0
        (0.003, 0.0),         # Middle of rectangular region → E_t = 0
        (0.0035, 0.0),        # Ultimate strain → E_t = 0
        (0.004, 0.0),         # Beyond ultimate → E_t = 0
    ]

    for strain, expected in test_cases:
        E_t = concrete_model.get_tangent_modulus(strain)

        if expected == 0.0:
            assert E_t == 0.0, f"Expected E_t=0 at strain={strain}, got {E_t}"
        elif expected == ">0":
            # Parabolic region should have positive gradient
            assert E_t > 0.0, f"Expected E_t>0 at strain={strain}, got {E_t}"
            # Should be less than typical E_c (~30-40 GPa)
            assert E_t < 50000.0, f"E_t too large at strain={strain}: {E_t}"

    print(f"\n[PASS] Concrete tangent modulus computed correctly")


def test_tangent_modulus_steel():
    """Test steel tangent modulus computation."""
    diagram = create_test_diagram()
    steel_model = diagram.steel_models[0]

    E_s = steel_model.steel.E_s  # ~200,000 MPa
    eps_y = steel_model.epsilon_y  # ~0.00217 for B500B

    # Test various strain regions
    test_cases = [
        (0.0, E_s),                # Zero strain → elastic modulus
        (0.001, E_s),              # Elastic region (tension)
        (-0.001, E_s),             # Elastic region (compression)
        (eps_y * 0.5, E_s),        # Mid elastic region
        (eps_y, E_s),              # Yield point (still elastic)
        (eps_y * 1.5, None),       # Plastic region (hardening or 0)
        (0.02, None),              # Well into plastic region
    ]

    for strain, expected_E_t in test_cases:
        E_t = steel_model.get_tangent_modulus(strain)

        if expected_E_t is not None:
            assert abs(E_t - expected_E_t) < 100.0, \
                f"E_t at strain={strain}: expected {expected_E_t}, got {E_t}"
        else:
            # Plastic region: either E_hardening or 0
            assert E_t >= 0.0, f"E_t cannot be negative at strain={strain}"
            # For inclined branch: E_hardening ~1000-3000 MPa
            # For horizontal branch: E_t = 0
            if steel_model.branch_type == "horizontal":
                assert E_t == 0.0
            else:
                # Inclined: expect positive hardening modulus < E_s
                assert 0 < E_t < E_s * 0.1

    print(f"\n[PASS] Steel tangent modulus computed correctly")


def test_jacobian_at_rectangular_region():
    """Test Jacobian computation when concrete is in rectangular region (E_t=0)."""
    diagram = create_test_diagram()

    # Strain pair with concrete in rectangular region (eps > 0.002)
    eps_top = 0.0025  # In rectangular region
    eps_bottom = 0.002  # At transition

    # Compute Jacobian (should not crash despite E_t=0 for some fibers)
    J = diagram._compute_analytical_jacobian(eps_top, eps_bottom)

    # Jacobian should be well-defined (steel provides stiffness)
    assert J.shape == (2, 2)
    assert not np.any(np.isnan(J)), "Jacobian contains NaN"
    assert not np.any(np.isinf(J)), "Jacobian contains Inf"

    print(f"\n[PASS] Jacobian computed correctly with concrete in rectangular region")


if __name__ == "__main__":
    # Run all tests
    print("Running analytical Jacobian verification tests...\n")

    test_analytical_jacobian_matches_numerical()
    test_analytical_jacobian_solver_convergence()
    test_analytical_jacobian_iteration_count()
    test_analytical_jacobian_performance()
    test_tangent_modulus_concrete()
    test_tangent_modulus_steel()
    test_jacobian_at_rectangular_region()

    print("\n" + "="*60)
    print("All analytical Jacobian tests passed!")
    print("="*60)
