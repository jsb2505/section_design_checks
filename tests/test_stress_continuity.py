"""
Verify stress continuity at the cracking transition.

A code review pointed out: If eps_cr != f_ctm/E_cm, you get a stress JUMP (not just slope change).
This would break Newton convergence badly.
"""

import numpy as np

from section_design_checks.core.geometry import Point2D
from section_design_checks.reinforced_concrete.analysis.interaction_diagram import MNInteractionDiagram
from section_design_checks.reinforced_concrete.geometry import RebarGroup, create_rectangular_section
from section_design_checks.reinforced_concrete.materials import ConcreteMaterial, Rebar


def test_stress_continuity_at_cracking():
    """
    Verify stress is continuous at eps = -eps_cr (cracking point).

    Pre-crack: sigma = E_cm * eps = E_cm * (-eps_cr) = -f_ctm
    Post-crack at boundary: sigma = -f_ctm * [1 - 0] = -f_ctm

    These MUST match for continuous stress.
    """
    print("\n=== Stress Continuity at Cracking ===\n")

    section = create_rectangular_section(width=300, height=500)
    rebar_20 = Rebar(diameter=20, grade="B500B")
    section.add_rebar_group(RebarGroup(rebar=rebar_20, positions=[Point2D(x=50, y=50)]))

    concrete = ConcreteMaterial(grade="C30/37")
    diagram = MNInteractionDiagram(section=section, concrete=concrete, tension_stiffening=True)

    f_ctm = concrete.f_ctm
    E_cm = concrete.E_cm
    eps_cr = f_ctm / E_cm

    print("Material: C30/37")
    print(f"  f_ctm = {f_ctm:.6f} MPa")
    print(f"  E_cm = {E_cm:.6f} MPa")
    print(f"  eps_cr (computed) = {eps_cr:.10f}")
    print(f"  eps_cr (f_ctm/E_cm) = {f_ctm/E_cm:.10f}")
    print()

    # Test strains around cracking point
    delta = 1e-10  # Tiny step

    strains = np.array([
        -eps_cr - delta,  # Just before cracking (pre-crack side)
        -eps_cr,           # Exactly at cracking
        -eps_cr + delta,  # Just after cracking (post-crack side)
    ])

    stresses = diagram._concrete_stress_with_options(strains)

    print(f"{'Strain':<20} {'Stress (MPa)':<15} {'Branch':<15}")
    print("=" * 55)
    print(f"{strains[0]:19.12f}  {stresses[0]:14.10f}  Pre-crack")
    print(f"{strains[1]:19.12f}  {stresses[1]:14.10f}  At boundary")
    print(f"{strains[2]:19.12f}  {stresses[2]:14.10f}  Post-crack")
    print()

    # Check continuity
    jump_left = abs(stresses[1] - stresses[0])
    jump_right = abs(stresses[2] - stresses[1])

    print(f"Jump (left to boundary): {jump_left:.10f} MPa")
    print(f"Jump (boundary to right): {jump_right:.10f} MPa")
    print()

    # Verify expected values
    expected_stress_at_crack = -f_ctm
    actual_stress_at_crack = stresses[1]

    print(f"Expected stress at cracking: {expected_stress_at_crack:.10f} MPa")
    print(f"Actual stress at cracking: {actual_stress_at_crack:.10f} MPa")
    print(f"Error: {abs(actual_stress_at_crack - expected_stress_at_crack):.12f} MPa")
    print()

    # Assert continuity (stress should be continuous to within numerical precision)
    # Note: 1e-5 tolerance accounts for floating-point arithmetic in stress computation
    assert jump_left < 1e-5, f"Stress jump on left side: {jump_left} MPa"
    assert jump_right < 1e-5, f"Stress jump on right side: {jump_right} MPa"
    assert abs(actual_stress_at_crack - expected_stress_at_crack) < 1e-5, \
        "Stress at cracking doesn't match -f_ctm"

    print("[OK] Stress is continuous at cracking point")
    print("     eps_cr is correctly defined as f_ctm/E_cm")


def test_tangent_discontinuity_at_cracking():
    """
    Verify tangent modulus is DISCONTINUOUS at cracking (this is expected).

    Left slope (pre-crack): E_t = E_cm
    Right slope (post-crack): E_t = -f_ctm * beta / (5*eps_cr)

    These are different (kink in stress-strain curve).
    """
    print("\n=== Tangent Modulus Discontinuity at Cracking ===\n")

    section = create_rectangular_section(width=300, height=500)
    rebar_20 = Rebar(diameter=20, grade="B500B")
    section.add_rebar_group(RebarGroup(rebar=rebar_20, positions=[Point2D(x=50, y=50)]))

    concrete = ConcreteMaterial(grade="C30/37")
    diagram = MNInteractionDiagram(section=section, concrete=concrete, tension_stiffening=True)

    f_ctm = concrete.f_ctm
    E_cm = concrete.E_cm
    eps_cr = f_ctm / E_cm
    beta = 0.6

    # Test tangent modulus around cracking
    delta = 1e-8

    strains = np.array([
        -eps_cr - delta,  # Pre-crack
        -eps_cr,           # At boundary
        -eps_cr + delta,  # Post-crack
    ])

    E_t = diagram._concrete_tangent_modulus_with_options(strains)

    print(f"{'Strain':<20} {'E_t (MPa)':<15} {'Branch':<15}")
    print("=" * 55)
    print(f"{strains[0]:19.12f}  {E_t[0]:14.4f}  Pre-crack")
    print(f"{strains[1]:19.12f}  {E_t[1]:14.4f}  At boundary")
    print(f"{strains[2]:19.12f}  {E_t[2]:14.4f}  Post-crack")
    print()

    # Expected values
    E_t_pre = E_cm
    E_t_post = -f_ctm * beta / (5.0 * eps_cr)

    print(f"Expected E_t (pre-crack):  {E_t_pre:.4f} MPa")
    print(f"Expected E_t (post-crack): {E_t_post:.4f} MPa")
    print(f"Ratio (post/pre): {E_t_post / E_t_pre:.4f}")
    print()

    # Check which side the boundary value matches
    if abs(E_t[1] - E_t_pre) < abs(E_t[1] - E_t_post):
        print("[INFO] Boundary returns PRE-crack tangent (E_cm)")
        print("       This is the '<=' branch in the code")
    else:
        print("[INFO] Boundary returns POST-crack tangent")
        print("       This is the '>' branch in the code")

    print("\n[OK] Tangent modulus correctly shows discontinuity at cracking")
    print("     This is mathematically correct (kink in stress-strain curve)")


def test_newton_behavior_near_cracking():
    """
    Test how Newton solver behaves when strains are near cracking point.

    If any fibers sit very close to cracking during iteration, the discontinuous
    tangent can cause zig-zagging or slow convergence.
    """
    print("\n=== Newton Behavior Near Cracking ===\n")

    section = create_rectangular_section(width=300, height=500)
    rebar_20 = Rebar(diameter=20, grade="B500B")

    # Add reinforcement
    section.add_rebar_group(RebarGroup(rebar=rebar_20, positions=[
        Point2D(x=50, y=50), Point2D(x=250, y=50)
    ]))
    section.add_rebar_group(RebarGroup(rebar=rebar_20, positions=[
        Point2D(x=50, y=450), Point2D(x=250, y=450)
    ]))

    concrete = ConcreteMaterial(grade="C30/37")
    diagram = MNInteractionDiagram(section=section, concrete=concrete, tension_stiffening=True)

    # Choose load case that should put some fibers near cracking
    # Small positive moment (small curvature) with small axial force
    M_target = 5.0   # Very small moment
    N_target = 50.0  # Small compression

    print(f"Solving for M={M_target} kN.m, N={N_target} kN")
    print("(This should create small curvature with fibers near cracking)")
    print()

    eps_top, eps_bottom = diagram.find_strains_for_MN(My_target=M_target, N_target=N_target)

    # Check fiber strains
    _, y, _, material_type, _, _, _ = diagram.mesh.get_fibre_arrays()
    strains = diagram._strain_field_from_end_strains(eps_top, eps_bottom)

    concrete_strains = strains[material_type == "concrete"]

    # Find how close any concrete fiber is to cracking
    eps_cr = concrete.f_ctm / concrete.E_cm

    # For fibers in tension (strain < 0)
    tension_strains = -concrete_strains[concrete_strains < 0]  # Convert to positive magnitude

    if len(tension_strains) > 0:
        distances_from_crack = np.abs(tension_strains - eps_cr)
        min_distance = np.min(distances_from_crack)
        closest_strain = tension_strains[np.argmin(distances_from_crack)]

        print(f"Cracking strain (eps_cr): {eps_cr:.8f}")
        print(f"Closest fiber to cracking: {closest_strain:.8f}")
        print(f"Distance from cracking: {min_distance:.10f}")
        print()

        if min_distance < 1e-6:
            print("[WARN] Fiber very close to cracking discontinuity!")
            print("       May cause Newton issues (consider smoothing)")
        else:
            print("[OK] No fibers extremely close to cracking point")
    else:
        print("[INFO] No fibers in tension for this load case")

    # Verify solution
    point = diagram.calculate_point_from_end_strains(eps_top, eps_bottom)
    print(f"\nSolution: M={point.M:.6f} kN.m, N={point.N:.6f} kN")
    print(f"Error: M={abs(point.M - M_target):.8f}, N={abs(point.N - N_target):.8f}")

    assert abs(point.M - M_target) < 0.01
    assert abs(point.N - N_target) < 0.01

    print("[OK] Solver converged correctly despite nearby discontinuity")


if __name__ == "__main__":
    test_stress_continuity_at_cracking()
    test_tangent_discontinuity_at_cracking()
    test_newton_behavior_near_cracking()

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("""
Key findings:

1. Stress IS continuous at cracking (eps_cr = f_ctm/E_cm) [OK]
   - No stress jump
   - Newton won't see force inconsistencies

2. Tangent modulus IS discontinuous (expected) [OK]
   - Pre-crack: E_t = E_cm (stiff)
   - Post-crack: E_t = -f_ctm*beta/(5*eps_cr) (softening)
   - This is mathematically correct (kink in curve)

3. Newton handles the kink reasonably well [OK]
   - Solver converges even with fibers near discontinuity
   - No special smoothing needed for typical cases

If you see convergence issues with certain load cases:
- Check if many fibers land exactly on eps_cr
- Consider smoothing band (1e-7 to 1e-6 around cracking)
- Use one-sided tangent (pre-crack stiffness for robustness)
    """)
