"""
Quick test to verify the new ShearCheck API works.
"""

from materials.core.geometry import Point2D
from materials.reinforced_concrete.code_checks.ec2_2004.shear_check import ShearCheck
from materials.reinforced_concrete.code_checks.ec2_2004.flexure_utils import LoadCase
from materials.reinforced_concrete.code_checks.base_check import CheckStatus
from materials.reinforced_concrete.geometry import create_rectangular_section, RebarGroup
from materials.reinforced_concrete.materials import ConcreteMaterial, ShearRebar, Rebar


def test_new_api_single_case():
    """Test new API with single load case."""
    # Create section
    section = create_rectangular_section(width=300, height=500)

    # Add bottom bars (tension for sagging)
    rebar_20 = Rebar(diameter=20, grade="B500B")
    positions = [Point2D(x=50, y=50), Point2D(x=250, y=50)]
    group = RebarGroup(rebar=rebar_20, positions=positions)
    section.add_rebar_group(group)

    # Materials
    concrete = ConcreteMaterial(grade="C30/37")
    shear_rebar = ShearRebar(diameter=10, link_spacing=200, n_legs=2, grade="B500B")

    # Create check with rigorous mode
    print("\n=== Creating ShearCheck (rigorous mode) ===")
    check = ShearCheck(
        section=section,
        concrete=concrete,
        shear_reinforcement=shear_rebar,
        use_mechanical_lever_arm=True,  # Default - accurate
    )
    print("Initialization complete!")

    # Single load case
    print("\n=== Checking single load case ===")
    load_case = LoadCase(V_Ed=150, M_Ed=50, N_Ed=100)
    print(f"Load case: V={load_case.V_Ed} kN, M={load_case.M_Ed} kN.m, N={load_case.N_Ed} kN")

    result = check.perform_check(load_case=load_case)

    print(f"\nResult: {result.status}")
    print(f"Utilization: {result.utilization:.1%}")
    print(f"Message: {result.message}")
    print(f"Details: d={result.details['d']:.1f} mm, z={result.details['z']:.1f} mm")
    print(f"Mode: {result.details['z_mode']}")
    print(f"rho_l: {result.details['rho_l']:.4f}")

    assert result.status in [CheckStatus.PASS, CheckStatus.WARNING], f"Unexpected status: {result.status}"
    print("\n[OK] Single case test passed!")


def test_new_api_multiple_cases():
    """Test new API with multiple load cases."""
    # Create section with both top and bottom reinforcement for hogging cases
    section = create_rectangular_section(width=300, height=500)
    rebar_20 = Rebar(diameter=20, grade="B500B")

    # Bottom bars (tension for sagging)
    bottom_positions = [Point2D(x=50, y=50), Point2D(x=250, y=50)]
    bottom_group = RebarGroup(rebar=rebar_20, positions=bottom_positions)
    section.add_rebar_group(bottom_group)

    # Top bars (tension for hogging)
    top_positions = [Point2D(x=50, y=450), Point2D(x=250, y=450)]
    top_group = RebarGroup(rebar=rebar_20, positions=top_positions)
    section.add_rebar_group(top_group)

    concrete = ConcreteMaterial(grade="C30/37")
    shear_rebar = ShearRebar(diameter=10, link_spacing=200, n_legs=2, grade="B500B")

    # Create check
    print("\n=== Creating ShearCheck for batch checking ===")
    check = ShearCheck(
        section=section,
        concrete=concrete,
        shear_reinforcement=shear_rebar,
        use_mechanical_lever_arm=True,
    )

    # Multiple load cases - use list comprehension
    print("\n=== Checking multiple load cases ===")
    load_cases = [
        LoadCase(V_Ed=150, M_Ed=50, N_Ed=100),   # Sagging
        LoadCase(V_Ed=120, M_Ed=-30, N_Ed=80),   # Hogging
        LoadCase(V_Ed=100, M_Ed=0, N_Ed=0),      # Pure shear
    ]

    results = [check.perform_check(load_case=case) for case in load_cases]

    print(f"\nChecked {len(results)} load cases:")
    for i, (case, result) in enumerate(zip(load_cases, results)):
        print(f"\n  Case {i+1}: V={case.V_Ed} kN, M={case.M_Ed} kN.m, N={case.N_Ed} kN")
        print(f"    Status: {result.status}, Util: {result.utilization:.1%}")
        print(f"    d={result.details['d']:.1f} mm, z={result.details['z']:.1f} mm")
        assert result.status in [CheckStatus.PASS, CheckStatus.WARNING], f"Case {i+1} failed"

    print("\n[OK] Multiple cases test passed!")


def test_approximate_mode():
    """Test approximate mode (fast, less accurate)."""
    section = create_rectangular_section(width=300, height=500)
    rebar_20 = Rebar(diameter=20, grade="B500B")
    positions = [Point2D(x=50, y=50), Point2D(x=250, y=50)]
    group = RebarGroup(rebar=rebar_20, positions=positions)
    section.add_rebar_group(group)

    concrete = ConcreteMaterial(grade="C30/37")
    shear_rebar = ShearRebar(diameter=10, link_spacing=200, n_legs=2, grade="B500B")

    print("\n=== Creating ShearCheck (approximate mode) ===")
    check = ShearCheck(
        section=section,
        concrete=concrete,
        shear_reinforcement=shear_rebar,
        use_mechanical_lever_arm=False,  # Fast approximate mode
    )
    print("Initialization complete (should be instant)!")

    load_case = LoadCase(V_Ed=150, M_Ed=50, N_Ed=100)
    result = check.perform_check(load_case=load_case)

    print(f"\nResult: {result.status}")
    print(f"Mode: {result.details['z_mode']}")
    assert result.details['z_mode'] == 'approximate'

    print("\n[OK] Approximate mode test passed!")


if __name__ == "__main__":
    test_new_api_single_case()
    test_new_api_multiple_cases()
    test_approximate_mode()
    print("\n=== All tests passed! ===")

