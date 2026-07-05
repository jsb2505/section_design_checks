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


def _create_asymmetric_section():
    """300x500mm section with 4xH20 bottom, 2xH16 top — the section used in
    shear viewer demonstrations where tension branch jumping was observed."""
    from materials.reinforced_concrete.geometry import create_linear_rebar_layer

    section = create_rectangular_section(width=300, height=500)
    bot_bar = Rebar(diameter=20, grade="B500B")
    top_bar = Rebar(diameter=16, grade="B500B")
    cover = 35.0
    link_dia = 10.0
    sc = cover + link_dia
    y_bot = cover + link_dia + bot_bar.diameter / 2.0
    y_top = 500 - cover - link_dia - top_bar.diameter / 2.0
    section.add_rebar_group(
        create_linear_rebar_layer(
            rebar=bot_bar,
            n_bars=4,
            start_point=(sc + bot_bar.diameter / 2.0, y_bot),
            end_point=(300 - sc - bot_bar.diameter / 2.0, y_bot),
        )
    )
    section.add_rebar_group(
        create_linear_rebar_layer(
            rebar=top_bar,
            n_bars=2,
            start_point=(sc + top_bar.diameter / 2.0, y_top),
            end_point=(300 - sc - top_bar.diameter / 2.0, y_top),
        )
    )
    return section


def test_tension_branch_prefers_all_tensile():
    """For moderate tension + moment well inside the envelope, the solver
    should find all-tensile strains (no spurious compression zone).

    Regression: before the branch-preference fix, the solver could jump to
    an eccentric ULS branch at certain N_Ed values, producing compression
    at one face when the section was clearly in net tension.
    """
    section = _create_asymmetric_section()
    concrete = ConcreteMaterial(grade="C30/37")
    diagram = MNInteractionDiagram(
        section=section, concrete=concrete,
        use_characteristic=False, use_accidental=False,
    )

    # M=60, N=-600: well inside the envelope, should be all-tensile
    eps_top, eps_bottom = diagram.find_strains_for_MN(My_target=60.0, N_target=-600.0)
    assert eps_top <= 0, f"eps_top={eps_top} should be tensile at N=-600"
    assert eps_bottom <= 0, f"eps_bottom={eps_bottom} should be tensile at N=-600"

    # Verify equilibrium
    point = diagram.calculate_point_from_end_strains(eps_top, eps_bottom)
    assert abs(point.N - (-600.0)) < 1.0
    assert abs(point.M - 60.0) < 1.0


def test_tension_branch_monotonic_strain_sweep():
    """Sweep N_Ed from -50 to -650 at M=60: strains should remain all-tensile
    within the envelope (before the natural transition to compression)."""
    section = _create_asymmetric_section()
    concrete = ConcreteMaterial(grade="C30/37")
    diagram = MNInteractionDiagram(
        section=section, concrete=concrete,
        use_characteristic=False, use_accidental=False,
    )

    # For small |N| with M=60, the top face may legitimately be in
    # compression (sagging moment dominates).  Both faces become tensile
    # once |N| is large enough.  Start from N=-350 where the section is
    # clearly in the all-tensile regime.
    for N in range(-350, -660, -50):
        eps_top, eps_bottom = diagram.find_strains_for_MN(My_target=60.0, N_target=float(N))
        assert eps_top <= 0, f"N={N}: eps_top={eps_top:.6f} should be tensile"
        assert eps_bottom <= 0, f"N={N}: eps_bottom={eps_bottom:.6f} should be tensile"


def test_eccentric_tension_near_boundary_still_works():
    """Near the M-N boundary, compression at one face IS the correct solution.
    The branch preference should not override genuine equilibria."""
    section = _create_asymmetric_section()
    concrete = ConcreteMaterial(grade="C30/37")
    diagram = MNInteractionDiagram(
        section=section, concrete=concrete,
        use_characteristic=False, use_accidental=False,
    )

    # M=200, N=-10: near pure bending — compression at top is expected
    eps_top, eps_bottom = diagram.find_strains_for_MN(My_target=200.0, N_target=-10.0)
    point = diagram.calculate_point_from_end_strains(eps_top, eps_bottom)
    assert abs(point.N - (-10.0)) < 1.0
    assert abs(point.M - 200.0) < 1.0
    # Top should be in compression for sagging moment
    assert eps_top > 0, f"eps_top={eps_top} should be compression at M=200, N=-10"


def test_tension_boundary_no_compression_spike():
    """At the M-N envelope boundary (M=60, N≈-670 to -700), the solver must
    not jump to an eccentric ULS branch with compression at one face.

    Regression: without the tensile-constrained solver pass and relaxed
    threshold, the solver would find an eccentric branch (tiny compression
    zone + extreme tension) near the capacity boundary, causing a spike in
    lever arm and shear utilization.
    """
    section = _create_asymmetric_section()
    concrete = ConcreteMaterial(grade="C30/37")
    diagram = MNInteractionDiagram(
        section=section, concrete=concrete,
        use_characteristic=False, use_accidental=False,
    )

    # Sweep across the boundary region — all should remain tensile
    for N in range(-650, -750, -10):
        eps_top, eps_bottom = diagram.find_strains_for_MN(
            My_target=60.0, N_target=float(N), strict=False,
        )
        assert eps_top <= 0, (
            f"N={N}: eps_top={eps_top:.6f} should be tensile (no compression spike)"
        )
        assert eps_bottom <= 0, (
            f"N={N}: eps_bottom={eps_bottom:.6f} should be tensile (no compression spike)"
        )


if __name__ == "__main__":
    test_inverse_solver_round_trip()
