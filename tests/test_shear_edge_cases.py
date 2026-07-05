"""
Test edge cases for ShearCheck, particularly compression face detection.
"""

from materials.core.geometry import Point2D
from materials.reinforced_concrete.code_checks.ec2.shear_check import ShearCheck, ShearLoadCase
from materials.reinforced_concrete.code_checks.base_check import CheckStatus
from materials.reinforced_concrete.geometry import create_rectangular_section, RebarGroup
from materials.reinforced_concrete.materials import ConcreteMaterial, ShearRebar, Rebar


def test_pure_axial_compression_face():
    """Test that pure axial load (M=0) deterministically selects top compression face."""
    # Create symmetric section
    section = create_rectangular_section(width=300, height=500)

    # Add symmetric reinforcement (top and bottom)
    rebar_20 = Rebar(diameter=20, grade="B500B")

    bottom_positions = [Point2D(x=-50, y=-200), Point2D(x=50, y=-200)]
    bottom_group = RebarGroup(rebar=rebar_20, positions=bottom_positions)
    section.add_rebar_group(bottom_group)

    top_positions = [Point2D(x=-50, y=200), Point2D(x=50, y=200)]
    top_group = RebarGroup(rebar=rebar_20, positions=top_positions)
    section.add_rebar_group(top_group)

    concrete = ConcreteMaterial(grade="C30/37")
    shear_rebar = ShearRebar(diameter=10, spacing=200, n_legs=2, grade="B500B")

    # Test with rigorous mode
    print("\n=== Testing Pure Axial Load (Rigorous Mode) ===")
    check_rigorous = ShearCheck(
        section=section,
        concrete=concrete,
        shear_reinforcement=shear_rebar,
        use_rigorous=True,
    )

    # Pure axial load: M_Ed = 0, N_Ed > 0, small V_Ed
    load_case = ShearLoadCase(V_Ed=50, M_Ed=0.0, N_Ed=500)
    print(f"Load case: V={load_case.V_Ed} kN, M={load_case.M_Ed} kN.m, N={load_case.N_Ed} kN")

    result = check_rigorous.perform_check(load_case=load_case)

    print(f"\nResult: {result.status}")
    print(f"Effective depth d: {result.details['d']:.1f} mm")

    # In pure axial case with symmetric section, eps_top should equal eps_bottom
    # With our fix, this should deterministically select "top" compression face
    # which corresponds to measuring d from top face (larger d value)
    d_rigorous = result.details['d']

    # Test with approximate mode for comparison
    print("\n=== Testing Pure Axial Load (Approximate Mode) ===")
    check_approximate = ShearCheck(
        section=section,
        concrete=concrete,
        shear_reinforcement=shear_rebar,
        use_rigorous=False,
    )

    result_approx = check_approximate.perform_check(load_case=load_case)
    d_approximate = result_approx.details['d']

    print(f"Approximate d: {d_approximate:.1f} mm")
    print(f"Rigorous d: {d_rigorous:.1f} mm")

    # They should match since both assume top compression by default
    # (within reasonable tolerance due to numerical effects)
    assert abs(d_rigorous - d_approximate) < 1.0, (
        f"Pure axial case should give same d in both modes: "
        f"rigorous={d_rigorous:.1f} vs approximate={d_approximate:.1f}"
    )

    print("\n[OK] Pure axial load edge case handled deterministically!")


def test_very_small_moment():
    """Test that very small moment doesn't cause instability in compression face detection."""
    section = create_rectangular_section(width=300, height=500)
    rebar_20 = Rebar(diameter=20, grade="B500B")

    bottom_positions = [Point2D(x=-50, y=-200), Point2D(x=50, y=-200)]
    bottom_group = RebarGroup(rebar=rebar_20, positions=bottom_positions)
    section.add_rebar_group(bottom_group)

    top_positions = [Point2D(x=-50, y=200), Point2D(x=50, y=200)]
    top_group = RebarGroup(rebar=rebar_20, positions=top_positions)
    section.add_rebar_group(top_group)

    concrete = ConcreteMaterial(grade="C30/37")
    shear_rebar = ShearRebar(diameter=10, spacing=200, n_legs=2, grade="B500B")

    check = ShearCheck(
        section=section,
        concrete=concrete,
        shear_reinforcement=shear_rebar,
        use_rigorous=True,
    )

    print("\n=== Testing Very Small Moment ===")
    # Very small moment (0.1 kN.m) with large axial load
    load_case = ShearLoadCase(V_Ed=50, M_Ed=0.1, N_Ed=500)

    result = check.perform_check(load_case=load_case)

    print(f"Load case: V={load_case.V_Ed} kN, M={load_case.M_Ed} kN.m, N={load_case.N_Ed} kN")
    print(f"Result: {result.status}")
    print(f"d: {result.details['d']:.1f} mm")

    # Should not raise errors and should give reasonable result
    assert result.status in [CheckStatus.PASS, CheckStatus.WARNING]
    assert result.details['d'] > 0

    print("\n[OK] Very small moment handled correctly!")


if __name__ == "__main__":
    test_pure_axial_compression_face()
    test_very_small_moment()
    print("\n=== All edge case tests passed! ===")
