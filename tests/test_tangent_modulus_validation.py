"""
Validate tension stiffening tangent modulus using finite-difference check.

This test uses numerical differentiation to verify that the analytical tangent
modulus dσ/deps is correct (w.r.t. signed strain eps, not magnitude eps_t).
"""

import numpy as np

from section_design_checks.core.geometry import Point2D
from section_design_checks.reinforced_concrete.analysis.interaction_diagram import MNInteractionDiagram
from section_design_checks.reinforced_concrete.geometry import RebarGroup, create_rectangular_section
from section_design_checks.reinforced_concrete.materials import ConcreteMaterial, Rebar


def finite_difference_tangent(diagram, strain, h=1e-10):
    """
    Compute dσ/deps numerically using central difference.

    Args:
        diagram: MNInteractionDiagram instance
        strain: Signed strain (negative for tension)
        h: Step size for finite difference

    Returns:
        Numerical tangent modulus dσ/deps
    """
    # Compute stress at eps-h and eps+h
    strains_minus = np.array([strain - h])
    strains_plus = np.array([strain + h])

    sigma_minus = diagram._concrete_stress_with_options(strains_minus)[0]
    sigma_plus = diagram._concrete_stress_with_options(strains_plus)[0]

    # Central difference
    Et_numerical = (sigma_plus - sigma_minus) / (2.0 * h)

    return Et_numerical


def test_tangent_modulus_vs_finite_difference():
    """
    Validate analytical tangent modulus against numerical differentiation.

    This catches sign errors in the tangent modulus formula.
    """
    print("\n=== Tangent Modulus Validation (Finite Difference Check) ===\n")

    # Create a simple diagram with tension stiffening
    section = create_rectangular_section(width=300, height=500)
    rebar_20 = Rebar(diameter=20, grade="B500B")
    positions = [Point2D(x=50, y=50), Point2D(x=250, y=50)]
    group = RebarGroup(rebar=rebar_20, positions=positions)
    section.add_rebar_group(group)

    concrete = ConcreteMaterial(grade="C30/37")

    diagram = MNInteractionDiagram(
        section=section,
        concrete=concrete,
        tension_stiffening=True,
    )

    # Material properties
    f_ctm = concrete.f_ctm
    E_cm = concrete.E_cm
    eps_cr = f_ctm / E_cm

    # Test points across all regions
    # NOTE: Avoid exact transition points (eps=0, eps_cr, eps_c2, etc.) where derivative is discontinuous
    test_strains = [
        (-0.00005, "Pre-cracking elastic"),
        (-eps_cr * 0.9, "Just before cracking"),  # Avoid exact eps_cr
        (-0.00015, "Post-cracking (softening)"),
        (-0.0005, "Deep post-cracking"),
        (-0.001, "Near/past cutoff"),
        (0.0001, "Small compression"),  # Avoid eps=0 (tension/compression boundary)
        (0.0005, "Compression (parabolic)"),
        (0.0015, "Compression (parabolic)"),  # Avoid eps_c2 transition
    ]

    print("Material: C30/37")
    print(f"  f_ctm = {f_ctm:.3f} MPa")
    print(f"  E_cm = {E_cm:.0f} MPa")
    print(f"  eps_cr = {eps_cr:.6f}\n")

    print(f"{'Strain':<12} {'Region':<25} {'Analytical':<15} {'Numerical':<15} {'Error %':<12}")
    print("=" * 90)

    max_error = 0.0

    for strain, region in test_strains:
        # Analytical tangent modulus
        strains_array = np.array([strain])
        Et_analytical = diagram._concrete_tangent_modulus_with_options(strains_array)[0]

        # Numerical tangent modulus (finite difference)
        Et_numerical = finite_difference_tangent(diagram, strain)

        # Relative error
        if abs(Et_numerical) > 1e-6:
            error_pct = abs(Et_analytical - Et_numerical) / abs(Et_numerical) * 100
        else:
            # Both should be near zero
            error_pct = abs(Et_analytical - Et_numerical) * 100  # Absolute error for ~zero

        max_error = max(max_error, error_pct)

        print(f"{strain:11.6f}  {region:<25} {Et_analytical:14.2f}  {Et_numerical:14.2f}  {error_pct:11.4f}")

        # Assert they match to high precision
        if abs(Et_numerical) > 1.0:  # Non-trivial tangent
            assert error_pct < 0.01, \
                f"Tangent mismatch at eps={strain}: analytical={Et_analytical}, numerical={Et_numerical}"
        else:  # Near-zero tangent
            assert abs(Et_analytical - Et_numerical) < 0.1, \
                f"Tangent mismatch at eps={strain}: analytical={Et_analytical}, numerical={Et_numerical}"

    print("=" * 90)
    print(f"\nMaximum error: {max_error:.6f}%")
    print("\n[OK] All tangent moduli match finite-difference derivatives!")
    print("     This confirms the chain rule was applied correctly:")
    print("     dsigma/deps = (dsigma/deps_t) * (deps_t/deps) = (dsigma/deps_t) * (-1)")


def test_specific_post_cracking_sign():
    """
    Specific test for post-cracking sign (the bug identified in review).

    In post-cracking tension:
    - As eps becomes more negative (more tension), σ should approach 0 (weaker)
    - Therefore dσ/deps should be NEGATIVE (stress decreases as strain decreases)
    """
    print("\n=== Post-Cracking Sign Verification ===\n")

    section = create_rectangular_section(width=300, height=500)
    rebar_20 = Rebar(diameter=20, grade="B500B")
    positions = [Point2D(x=50, y=50)]
    group = RebarGroup(rebar=rebar_20, positions=positions)
    section.add_rebar_group(group)

    concrete = ConcreteMaterial(grade="C30/37")
    diagram = MNInteractionDiagram(
        section=section,
        concrete=concrete,
        tension_stiffening=True,
    )

    f_ctm = concrete.f_ctm
    E_cm = concrete.E_cm
    eps_cr = f_ctm / E_cm

    # Test in post-cracking region
    eps_post_crack = -0.0002  # Well into post-cracking

    # Compute stresses at three points
    strains = np.array([eps_post_crack - 0.0001, eps_post_crack, eps_post_crack + 0.0001])
    stresses = diagram._concrete_stress_with_options(strains)

    print("Post-cracking behavior check:")
    print(f"  eps_cr = {eps_cr:.6f}\n")
    print("  Strain       Stress (MPa)   Physical Meaning")
    print(f"  {strains[0]:.6f}   {stresses[0]:10.4f}   <- More tension")
    print(f"  {strains[1]:.6f}   {stresses[1]:10.4f}   <- Middle")
    print(f"  {strains[2]:.6f}   {stresses[2]:10.4f}   <- Less tension")

    # Physical check: more tension (more negative eps) should give weaker tension (σ closer to 0)
    # stresses[0] is most negative (most tension) →should be closest to zero in post-cracking
    # In post-cracking: as eps becomes more negative, σ approaches zero (softening)
    assert stresses[0] > stresses[1] > stresses[2], \
        "WRONG: More tension should give weaker stress (closer to zero)"

    # All should be negative (tension)
    assert all(s < 0 for s in stresses), "Tension stresses should be negative"

    print("\n  [OK] Stress correctly weakens as tension increases")

    # Analytical tangent
    Et = diagram._concrete_tangent_modulus_with_options(np.array([eps_post_crack]))[0]

    print(f"\n  Tangent modulus at eps={eps_post_crack}: {Et:.2f} MPa")

    # Should be NEGATIVE (softening behavior)
    assert Et < 0, f"Post-cracking tangent must be negative! Got {Et}"

    print("  [OK] Tangent modulus is negative (softening)")

    # Verify formula matches expected value
    beta = 0.6
    Et_expected = -f_ctm * beta / (5.0 * eps_cr)

    assert abs(Et - Et_expected) < 0.01, f"Formula mismatch: {Et} vs {Et_expected}"

    print(f"  [OK] Matches formula: -f_ctm * beta / (5*eps_cr) = {Et_expected:.2f}")

    print("\n[OK] Post-cracking sign is correct (negative tangent = softening)")


if __name__ == "__main__":
    test_tangent_modulus_vs_finite_difference()
    test_specific_post_cracking_sign()

    print("\n" + "="*90)
    print("  ALL VALIDATION TESTS PASSED!")
    print("  The tangent modulus implementation is mathematically correct.")
    print("="*90)
