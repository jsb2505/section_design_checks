"""
Test to verify lever arm clamping warning functionality.

This creates a test case designed to trigger the z > z_d_ratio_upper * d condition
by using a section with extreme loading conditions.
"""

import warnings

from section_design_checks.core.geometry import Point2D
from section_design_checks.reinforced_concrete.code_checks.ec2_2004.flexure_utils import LoadCase
from section_design_checks.reinforced_concrete.code_checks.ec2_2004.shear_check import ShearCheck
from section_design_checks.reinforced_concrete.geometry import RebarGroup, create_rectangular_section
from section_design_checks.reinforced_concrete.materials import ConcreteMaterial, Rebar, ShearRebar


def test_lever_arm_clamping_warning():
    """Test that warning is issued when lever arm exceeds upper bound."""

    print("\n" + "="*80)
    print("TESTING LEVER ARM CLAMPING WARNING")
    print("="*80)
    print()

    # Create a section designed to potentially exceed upper bound
    # Tall section with reinforcement concentrated at extremes
    section = create_rectangular_section(width=300, height=800)

    # Add heavy top reinforcement (compression can develop here with hogging)
    rebar_25 = Rebar(diameter=25, grade='B500B')
    section.add_rebar_group(RebarGroup(rebar=rebar_25, positions=[
        Point2D(x=50, y=750),
        Point2D(x=150, y=750),
        Point2D(x=250, y=750)
    ]))

    # Add minimal bottom reinforcement
    rebar_12 = Rebar(diameter=12, grade='B500B')
    section.add_rebar_group(RebarGroup(rebar=rebar_12, positions=[
        Point2D(x=150, y=50)
    ]))

    concrete = ConcreteMaterial(grade='C30/37')
    shear_rebar = ShearRebar(diameter=10, link_spacing=200, n_legs=2, grade='B500B')

    # Test case 1: With default bounds (should warn if z > z_d_ratio_upper * d)
    print("Test 1: z_d_ratio_upper=0.95 (default)")
    print("-" * 80)

    check = ShearCheck(
        section=section,
        concrete=concrete,
        shear_reinforcement=shear_rebar,
        use_mechanical_lever_arm=True,
    )

    # Try low moments to maximize lever arm
    # Lower moment → shallow neutral axis → larger distance between force centroids
    load_cases = [
        LoadCase(V_Ed=100, M_Ed=1.0, N_Ed=500),    # Very low moment + compression (max lever arm)
        LoadCase(V_Ed=100, M_Ed=5.0, N_Ed=300),    # Low moment + moderate compression
        LoadCase(V_Ed=100, M_Ed=-2.0, N_Ed=400),   # Very low hogging moment
    ]

    for i, load_case in enumerate(load_cases, 1):
        print(f"\nLoad Case {i}: V={load_case.V_Ed}kN, M={load_case.M_Ed}kN·m, N={load_case.N_Ed}kN")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            result = check.perform_check(load_case=load_case)

            d = result.details['d']
            z_design = result.details['z']
            z_mech = result.details.get('z_mech')
            z_upper = check.z_d_ratio_upper * d

            print(f"  d = {d:.1f} mm")
            print(f"  {check.z_d_ratio_upper:.2f}d upper limit = {z_upper:.1f} mm")
            z_mech_str = f"{z_mech:.1f}" if z_mech is not None else "N/A"
            print(f"  z_mech = {z_mech_str} mm")
            print(f"  z_design (used) = {z_design:.1f} mm")

            if z_mech is not None and z_mech > z_upper:
                print(f"  [!] CLAMPED to upper bound (z_mech > {check.z_d_ratio_upper:.2f}d by {z_mech - z_upper:.1f}mm)")

                if len(w) > 0:
                    print(f"  [OK] Warning issued: '{w[0].message}'")
                else:
                    print("  [FAIL] WARNING: Clamping applied but no warning issued!")
            else:
                print(f"  [OK] No clamping needed (z_mech <= {check.z_d_ratio_upper:.2f}d)")

                if len(w) > 0:
                    print(f"  [FAIL] WARNING: Unexpected warning: '{w[0].message}'")

    print()
    print("=" * 80)

    # Test case 2: With tight bounds (wide range, should rarely clamp)
    print("\nTest 2: z_d_ratio_upper=1.0 (no effective upper clamp)")
    print("-" * 80)

    check_wide = ShearCheck(
        section=section,
        concrete=concrete,
        shear_reinforcement=shear_rebar,
        use_mechanical_lever_arm=True,
        z_d_ratio_upper=1.0,
        z_d_ratio_lower=0.10,
    )

    for i, load_case in enumerate(load_cases, 1):
        print(f"\nLoad Case {i}: V={load_case.V_Ed}kN, M={load_case.M_Ed}kN·m, N={load_case.N_Ed}kN")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            result = check_wide.perform_check(load_case=load_case)

            z_design = result.details['z']
            z_mech = result.details.get('z_mech')

            if z_mech is not None:
                print(f"  z_mech = z_design = {z_mech:.1f} mm (wide bounds)")
            else:
                print(f"  z_design = {z_design:.1f} mm (z_mech not available)")

            clamping_warnings = [x for x in w if "clamped" in str(x.message).lower()]
            if len(clamping_warnings) > 0:
                print(f"  [FAIL] WARNING: Unexpected clamping warning with wide bounds: '{clamping_warnings[0].message}'")
            else:
                print("  [OK] No clamping warning (as expected)")

    print()
    print("=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
    print()


if __name__ == "__main__":
    test_lever_arm_clamping_warning()
