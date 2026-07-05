"""
Quick test to verify the inverse solver works with round-trip verification.
"""

import pytest
from materials.core.geometry import Point2D
from materials.reinforced_concrete.analysis.interaction_diagram import MNInteractionDiagram
from materials.reinforced_concrete.geometry import create_rectangular_section, RebarGroup
from materials.reinforced_concrete.materials import ConcreteMaterial, Rebar


def create_test_section():
    """Create a simple rectangular section with reinforcement."""
    # Use default hook_ref=1 (positive quadrant: 0 to 300 in x, 0 to 500 in y)
    section = create_rectangular_section(width=300, height=500)

    # Add 4H20 bars (2 top, 2 bottom)
    rebar_20 = Rebar(diameter=20, grade="B500B")

    # Bottom bars (50mm cover from left/right edges, 50mm from bottom)
    bottom_positions = [Point2D(x=50, y=50), Point2D(x=250, y=50)]
    bottom_group = RebarGroup(rebar=rebar_20, positions=bottom_positions)
    section.add_rebar_group(bottom_group)

    # Top bars (50mm cover from left/right edges, 50mm from top)
    top_positions = [Point2D(x=50, y=450), Point2D(x=250, y=450)]
    top_group = RebarGroup(rebar=rebar_20, positions=top_positions)
    section.add_rebar_group(top_group)

    return section


def test_inverse_solver_round_trip():
    """Test that inverse solver can find strains for known (M,N) points."""
    section = create_test_section()
    concrete = ConcreteMaterial(grade="C30/37")

    # Create diagram (but don't generate curve - not needed!)
    diagram = MNInteractionDiagram(
        section=section,
        concrete=concrete,
        use_characteristic=False,
        use_accidental=False,
    )

    # Test case 1: Pure compression
    print("\n=== Test 1: Pure Compression ===")
    N_target = 500.0  # kN
    M_target = 0.0    # kN·m

    eps_top, eps_bottom = diagram.find_strains_for_MN(M_target, N_target)
    print(f"Target: N={N_target:.2f} kN, M={M_target:.2f} kN·m")
    print(f"Found strains: eps_top={eps_top:.6f}, eps_bottom={eps_bottom:.6f}")

    # Verify round-trip
    point = diagram.calculate_point_from_end_strains(eps_top, eps_bottom)
    print(f"Round-trip: N={point.N:.2f} kN, M={point.M:.2f} kN·m")

    assert abs(point.N - N_target) < 1.0, f"N mismatch: {point.N} vs {N_target}"
    assert abs(point.M - M_target) < 1.0, f"M mismatch: {point.M} vs {M_target}"
    print("[OK] Round-trip verified!")

    # Test case 2: Sagging moment with compression
    print("\n=== Test 2: Sagging Moment + Compression ===")
    N_target = 300.0  # kN
    M_target = 80.0   # kN·m

    eps_top, eps_bottom = diagram.find_strains_for_MN(M_target, N_target)
    print(f"Target: N={N_target:.2f} kN, M={M_target:.2f} kN·m")
    print(f"Found strains: eps_top={eps_top:.6f}, eps_bottom={eps_bottom:.6f}")

    point = diagram.calculate_point_from_end_strains(eps_top, eps_bottom)
    print(f"Round-trip: N={point.N:.2f} kN, M={point.M:.2f} kN·m")

    assert abs(point.N - N_target) < 1.0, f"N mismatch: {point.N} vs {N_target}"
    assert abs(point.M - M_target) < 1.0, f"M mismatch: {point.M} vs {M_target}"
    print("[OK] Round-trip verified!")

    # Test case 3: Pure bending
    print("\n=== Test 3: Pure Bending (M only) ===")
    N_target = 0.0    # kN
    M_target = 50.0   # kN·m

    eps_top, eps_bottom = diagram.find_strains_for_MN(M_target, N_target)
    print(f"Target: N={N_target:.2f} kN, M={M_target:.2f} kN·m")
    print(f"Found strains: eps_top={eps_top:.6f}, eps_bottom={eps_bottom:.6f}")

    point = diagram.calculate_point_from_end_strains(eps_top, eps_bottom)
    print(f"Round-trip: N={point.N:.2f} kN, M={point.M:.2f} kN·m")

    assert abs(point.N - N_target) < 1.0, f"N mismatch: {point.N} vs {N_target}"
    assert abs(point.M - M_target) < 1.0, f"M mismatch: {point.M} vs {M_target}"
    print("[OK] Round-trip verified!")

    # Test case 4: Hogging moment (negative M)
    print("\n=== Test 4: Hogging Moment ===")
    N_target = 200.0   # kN
    M_target = -60.0   # kN·m (negative = hogging)

    eps_top, eps_bottom = diagram.find_strains_for_MN(M_target, N_target)
    print(f"Target: N={N_target:.2f} kN, M={M_target:.2f} kN·m")
    print(f"Found strains: eps_top={eps_top:.6f}, eps_bottom={eps_bottom:.6f}")

    point = diagram.calculate_point_from_end_strains(eps_top, eps_bottom)
    print(f"Round-trip: N={point.N:.2f} kN, M={point.M:.2f} kN·m")

    assert abs(point.N - N_target) < 1.0, f"N mismatch: {point.N} vs {N_target}"
    assert abs(point.M - M_target) < 1.0, f"M mismatch: {point.M} vs {M_target}"
    print("[OK] Round-trip verified!")

    print("\n=== All tests passed! ===")


if __name__ == "__main__":
    test_inverse_solver_round_trip()
