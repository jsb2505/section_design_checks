"""
Test that initial guess quality is good after sign convention fix.

Verifies that _estimate_initial_strains() provides guesses in the correct
quadrant that are reasonably close to the final solution.
"""

from section_design_checks.core.geometry import Point2D
from section_design_checks.reinforced_concrete.analysis.interaction_diagram import MNInteractionDiagram
from section_design_checks.reinforced_concrete.geometry import RebarGroup, create_rectangular_section
from section_design_checks.reinforced_concrete.materials import ConcreteMaterial, Rebar


def create_test_section():
    """Create a simple rectangular section with reinforcement."""
    section = create_rectangular_section(width=300, height=500)
    rebar_20 = Rebar(diameter=20, grade="B500B")

    # Bottom bars (50mm cover from edges)
    bottom_positions = [Point2D(x=50, y=50), Point2D(x=250, y=50)]
    bottom_group = RebarGroup(rebar=rebar_20, positions=bottom_positions)
    section.add_rebar_group(bottom_group)

    # Top bars (50mm cover from edges)
    top_positions = [Point2D(x=50, y=450), Point2D(x=250, y=450)]
    top_group = RebarGroup(rebar=rebar_20, positions=top_positions)
    section.add_rebar_group(top_group)

    return section


def test_initial_guess_sign_correctness():
    """Verify that initial guesses have correct signs (compression positive)."""
    section = create_test_section()
    concrete = ConcreteMaterial(grade="C30/37")
    diagram = MNInteractionDiagram(section=section, concrete=concrete, use_characteristic=False)

    print("\n=== Test: Initial Guess Sign Correctness ===")

    # Test case 1: Pure compression (both should be positive)
    print("\nCase 1: Pure compression (N=500, M=0)")
    eps_top, eps_bot = diagram._estimate_initial_strains(M=0.0, N=500.0)
    print(f"  Initial guess: eps_top={eps_top:+.6f}, eps_bot={eps_bot:+.6f}")
    assert eps_top > 0, "Pure compression: top should be positive (compressed)"
    assert eps_bot > 0, "Pure compression: bottom should be positive (compressed)"
    print("  [OK] Both positive")

    # Test case 2: Pure sagging (top positive, bottom negative)
    print("\nCase 2: Pure sagging (N=0, M=50)")
    eps_top, eps_bot = diagram._estimate_initial_strains(M=50.0, N=0.0)
    print(f"  Initial guess: eps_top={eps_top:+.6f}, eps_bot={eps_bot:+.6f}")
    assert eps_top > 0, "Sagging: top should be positive (compressed)"
    assert eps_bot < 0, "Sagging: bottom should be negative (tension)"
    print("  [OK] Top compressed, bottom tension")

    # Test case 3: Pure hogging (top negative, bottom positive)
    print("\nCase 3: Pure hogging (N=0, M=-50)")
    eps_top, eps_bot = diagram._estimate_initial_strains(M=-50.0, N=0.0)
    print(f"  Initial guess: eps_top={eps_top:+.6f}, eps_bot={eps_bot:+.6f}")
    assert eps_top < 0, "Hogging: top should be negative (tension)"
    assert eps_bot > 0, "Hogging: bottom should be positive (compressed)"
    print("  [OK] Top tension, bottom compressed")

    # Test case 4: Pure tension (both should be negative)
    print("\nCase 4: Pure tension (N=-200, M=0)")
    eps_top, eps_bot = diagram._estimate_initial_strains(M=0.0, N=-200.0)
    print(f"  Initial guess: eps_top={eps_top:+.6f}, eps_bot={eps_bot:+.6f}")
    assert eps_top < 0, "Pure tension: top should be negative (tension)"
    assert eps_bot < 0, "Pure tension: bottom should be negative (tension)"
    print("  [OK] Both negative")

    print("\n[OK] All initial guesses have correct signs!")


def test_initial_guess_produces_correct_quadrant():
    """Verify initial guesses produce forces in the correct quadrant."""
    section = create_test_section()
    concrete = ConcreteMaterial(grade="C30/37")
    diagram = MNInteractionDiagram(section=section, concrete=concrete, use_characteristic=False)

    print("\n=== Test: Initial Guess Produces Correct Quadrant ===")

    test_cases = [
        # (M_target, N_target, description)
        (0.0, 500.0, "Pure compression"),
        (50.0, 0.0, "Pure sagging"),
        (-50.0, 0.0, "Pure hogging"),
        (0.0, -200.0, "Pure tension"),
        (50.0, 100.0, "Sagging + compression"),
        (-50.0, 100.0, "Hogging + compression"),
    ]

    for M_target, N_target, desc in test_cases:
        print(f"\n{desc}: M={M_target:.1f}, N={N_target:.1f}")

        # Get initial guess
        eps_top, eps_bot = diagram._estimate_initial_strains(M_target, N_target)

        # Evaluate forward model at initial guess
        point = diagram.calculate_point_from_end_strains(eps_top, eps_bot)

        print(f"  Initial guess: eps_top={eps_top:+.6f}, eps_bot={eps_bot:+.6f}")
        print(f"  Forward eval:  N={point.N:.1f} kN, M={point.M:.1f} kN.m")

        # Check that signs match (correct quadrant)
        if abs(N_target) > 1e-6:
            assert (point.N > 0) == (N_target > 0), f"N sign mismatch: {point.N} vs {N_target}"
            print("  [OK] N sign correct")

        if abs(M_target) > 1e-6:
            assert (point.M > 0) == (M_target > 0), f"M sign mismatch: {point.M} vs {M_target}"
            print("  [OK] M sign correct")

    print("\n[OK] All initial guesses produce correct quadrant!")


def test_solver_convergence_with_correct_guess():
    """Verify solver converges successfully from corrected initial guesses."""
    section = create_test_section()
    concrete = ConcreteMaterial(grade="C30/37")
    diagram = MNInteractionDiagram(section=section, concrete=concrete, use_characteristic=False)

    print("\n=== Test: Solver Convergence with Corrected Guesses ===")

    test_cases = [
        (0.0, 500.0, "Pure compression"),
        (50.0, 0.0, "Pure sagging"),
        (-50.0, 0.0, "Pure hogging"),
        (50.0, 100.0, "Sagging + compression"),
        (-30.0, 80.0, "Hogging + compression"),
    ]

    for M_target, N_target, desc in test_cases:
        print(f"\n{desc}: M={M_target:.1f}, N={N_target:.1f}")

        # Solve
        eps_top, eps_bot = diagram.find_strains_for_MN(M_target, N_target)

        # Verify round-trip
        point = diagram.calculate_point_from_end_strains(eps_top, eps_bot)

        print(f"  Solution: eps_top={eps_top:+.6f}, eps_bot={eps_bot:+.6f}")
        print(f"  Check:    N={point.N:.2f} vs {N_target:.2f}, M={point.M:.2f} vs {M_target:.2f}")

        # Assert close match
        assert abs(point.N - N_target) < 1.0, f"N error: {abs(point.N - N_target):.2f}"
        assert abs(point.M - M_target) < 1.0, f"M error: {abs(point.M - M_target):.2f}"

        print("  [OK] Converged successfully")

    print("\n[OK] All cases converged with correct initial guesses!")


if __name__ == "__main__":
    test_initial_guess_sign_correctness()
    test_initial_guess_produces_correct_quadrant()
    test_solver_convergence_with_correct_guess()
    print("\n=== All initial guess quality tests passed! ===")
