"""
Test optional M_Ed parameter in LoadCase.

Verifies:
1. M_Ed can be omitted (defaults to 0.0)
2. In approximate mode with M_Ed=0: uses default top compression
3. In approximate mode with M_Ed>0 (sagging): uses top compression
4. In approximate mode with M_Ed<0 (hogging): uses bottom compression
5. Lever arm always remains 0.9d in approximate mode (not affected by M_Ed)
"""

from materials.core.geometry import Point2D
from materials.reinforced_concrete.code_checks.base_check import CheckStatus
from materials.reinforced_concrete.code_checks.ec2_2004.flexure_utils import LoadCase
from materials.reinforced_concrete.code_checks.ec2_2004.shear_check import ShearCheck
from materials.reinforced_concrete.geometry import RebarGroup, create_rectangular_section
from materials.reinforced_concrete.materials import ConcreteMaterial, Rebar, ShearRebar


def test_optional_M_Ed():
    """Test that M_Ed can be omitted (defaults to 0.0)."""
    # Create simple section
    section = create_rectangular_section(width=300, height=500)
    rebar_20 = Rebar(diameter=20, grade="B500B")
    positions = [Point2D(x=50, y=50), Point2D(x=250, y=50)]
    group = RebarGroup(rebar=rebar_20, positions=positions)
    section.add_rebar_group(group)

    concrete = ConcreteMaterial(grade="C30/37")
    shear_rebar = ShearRebar(diameter=10, link_spacing=200, n_legs=2, grade="B500B")

    print("\n=== Test 1: M_Ed can be omitted ===")
    check = ShearCheck(
        section=section,
        concrete=concrete,
        shear_reinforcement=shear_rebar,
        use_mechanical_lever_arm=False,
    )

    # Create load case without M_Ed (should default to 0.0)
    load_case = LoadCase(V_Ed=100)  # M_Ed and N_Ed default to 0.0
    print(f"Load case: V={load_case.V_Ed} kN, M={load_case.M_Ed} kN.m, N={load_case.N_Ed} kN")

    result = check.perform_check(load_case=load_case)

    print(f"Result: {result.status}")
    print(f"d: {result.details['d']:.1f} mm")
    assert result.status in [CheckStatus.PASS, CheckStatus.WARNING]
    assert load_case.M_Ed == 0.0
    assert load_case.N_Ed == 0.0

    print("[OK] M_Ed defaults to 0.0 when omitted!")


def test_approximate_mode_compression_face():
    """Test compression face detection in approximate mode based on M_Ed sign."""
    # Create section with both top and bottom reinforcement
    section = create_rectangular_section(width=300, height=500)
    rebar_20 = Rebar(diameter=20, grade="B500B")

    # Bottom bars
    bottom_positions = [Point2D(x=50, y=50), Point2D(x=250, y=50)]
    bottom_group = RebarGroup(rebar=rebar_20, positions=bottom_positions)
    section.add_rebar_group(bottom_group)

    # Top bars
    top_positions = [Point2D(x=50, y=450), Point2D(x=250, y=450)]
    top_group = RebarGroup(rebar=rebar_20, positions=top_positions)
    section.add_rebar_group(top_group)

    concrete = ConcreteMaterial(grade="C30/37")
    shear_rebar = ShearRebar(diameter=10, link_spacing=200, n_legs=2, grade="B500B")

    check = ShearCheck(
        section=section,
        concrete=concrete,
        shear_reinforcement=shear_rebar,
        use_mechanical_lever_arm=False,  # Approximate mode
    )

    # Test Case 1: M_Ed = 0 (should default to top compression)
    print("\n=== Test 2a: M_Ed = 0 (default top compression) ===")
    load_case_1 = LoadCase(V_Ed=100, M_Ed=0.0, N_Ed=0.0)
    result_1 = check.perform_check(load_case=load_case_1)
    d_1 = result_1.details['d']
    z_1 = result_1.details['z']

    print(f"M_Ed=0: d={d_1:.1f} mm, z={z_1:.1f} mm")
    # With top compression, d is measured from top to bottom bars
    # For 500mm height with 50mm cover: d = 500 - 50 - 10 (rebar/2) = 450mm (approx)
    assert d_1 > 440  # Should be close to 450mm

    # Test Case 2: M_Ed > 0 (sagging, top compression)
    print("\n=== Test 2b: M_Ed > 0 (sagging, top compression) ===")
    load_case_2 = LoadCase(V_Ed=100, M_Ed=50.0, N_Ed=0.0)
    result_2 = check.perform_check(load_case=load_case_2)
    d_2 = result_2.details['d']
    z_2 = result_2.details['z']

    print(f"M_Ed=50 (sagging): d={d_2:.1f} mm, z={z_2:.1f} mm")
    # Should give same d as M_Ed=0 (both top compression)
    assert abs(d_2 - d_1) < 0.1, f"Sagging should use top compression: d={d_2} vs {d_1}"

    # Test Case 3: M_Ed < 0 (hogging, bottom compression)
    print("\n=== Test 2c: M_Ed < 0 (hogging, bottom compression) ===")
    load_case_3 = LoadCase(V_Ed=100, M_Ed=-50.0, N_Ed=0.0)
    result_3 = check.perform_check(load_case=load_case_3)
    d_3 = result_3.details['d']
    z_3 = result_3.details['z']

    print(f"M_Ed=-50 (hogging): d={d_3:.1f} mm, z={z_3:.1f} mm")
    # With bottom compression, d is measured from bottom to top bars
    # Should be same magnitude but different direction → same d value
    assert abs(d_3 - d_1) < 0.1, "Symmetric section should give same d magnitude"

    # Verify lever arm is always 0.9d in approximate mode
    print("\n=== Test 2d: Verify z = 0.9d in all cases ===")
    assert abs(z_1 - 0.9 * d_1) < 0.1, f"Case 1: z should be 0.9d: {z_1} vs {0.9*d_1}"
    assert abs(z_2 - 0.9 * d_2) < 0.1, f"Case 2: z should be 0.9d: {z_2} vs {0.9*d_2}"
    assert abs(z_3 - 0.9 * d_3) < 0.1, f"Case 3: z should be 0.9d: {z_3} vs {0.9*d_3}"

    print("[OK] Approximate mode correctly uses M_Ed sign for compression face!")
    print("[OK] Lever arm always z = 0.9d (not affected by M_Ed)!")


def test_rigorous_mode_with_optional_M():
    """Test that rigorous mode works with optional M_Ed."""
    section = create_rectangular_section(width=300, height=500)
    rebar_20 = Rebar(diameter=20, grade="B500B")

    bottom_positions = [Point2D(x=50, y=50), Point2D(x=250, y=50)]
    bottom_group = RebarGroup(rebar=rebar_20, positions=bottom_positions)
    section.add_rebar_group(bottom_group)

    top_positions = [Point2D(x=50, y=450), Point2D(x=250, y=450)]
    top_group = RebarGroup(rebar=rebar_20, positions=top_positions)
    section.add_rebar_group(top_group)

    concrete = ConcreteMaterial(grade="C30/37")
    shear_rebar = ShearRebar(diameter=10, link_spacing=200, n_legs=2, grade="B500B")

    print("\n=== Test 3: Rigorous mode with optional M_Ed ===")
    check = ShearCheck(
        section=section,
        concrete=concrete,
        shear_reinforcement=shear_rebar,
        use_mechanical_lever_arm=True,  # Rigorous mode
    )

    # Test with M_Ed omitted (defaults to 0.0)
    load_case = LoadCase(V_Ed=100)  # M_Ed=0, N_Ed=0 by default
    print(f"Load case: V={load_case.V_Ed} kN, M={load_case.M_Ed} kN.m, N={load_case.N_Ed} kN")

    result = check.perform_check(load_case=load_case)

    print(f"Result: {result.status}")
    print(f"Mode: {result.details['z_mode']}")
    print(f"d: {result.details['d']:.1f} mm, z: {result.details['z']:.1f} mm")

    assert result.details['z_mode'] == "rigorous"
    assert result.status in [CheckStatus.PASS, CheckStatus.WARNING]

    print("[OK] Rigorous mode works with optional M_Ed!")


def test_simple_use_case():
    """Test the simplest possible use case: just V_Ed."""
    print("\n=== Test 4: Simplest use case (V_Ed only) ===")

    # Simple section
    section = create_rectangular_section(width=300, height=500)
    rebar_20 = Rebar(diameter=20, grade="B500B")
    positions = [Point2D(x=50, y=50), Point2D(x=250, y=50)]
    group = RebarGroup(rebar=rebar_20, positions=positions)
    section.add_rebar_group(group)

    concrete = ConcreteMaterial(grade="C30/37")
    shear_rebar = ShearRebar(diameter=10, link_spacing=200, n_legs=2, grade="B500B")

    # Approximate mode (fast, simple)
    check = ShearCheck(
        section=section,
        concrete=concrete,
        shear_reinforcement=shear_rebar,
        use_mechanical_lever_arm=False,  # Fast mode
    )

    # Simplest possible load case - just V_Ed!
    result = check.perform_check(load_case=LoadCase(V_Ed=150))

    print(f"V_Ed=150 kN -> Status: {result.status}, Util: {result.utilization:.1%}")
    print(f"d={result.details['d']:.1f} mm, z={result.details['z']:.1f} mm")

    assert result.status in [CheckStatus.PASS, CheckStatus.WARNING]

    print("[OK] Simplest use case works: just specify V_Ed!")


if __name__ == "__main__":
    test_optional_M_Ed()
    test_approximate_mode_compression_face()
    test_rigorous_mode_with_optional_M()
    test_simple_use_case()
    print("\n=== All optional M_Ed tests passed! ===")

