"""
Example demonstrating EC2 code checks - both codified and first principles approaches.

Shows:
1. BENDING CHECK - First principles (strain compatibility, force equilibrium)
2. SHEAR CHECK - Codified approach (EC2 empirical formulas with business logic)
"""

from materials.reinforced_concrete.geometry import (
    create_rectangular_section,
    create_linear_rebar_layer,
)
from materials.reinforced_concrete.materials import (
    ConcreteMaterial,
    Rebar,
    ShearRebar,
)
from materials.reinforced_concrete.code_checks.ec2.bending_check import BendingCheck
from materials.reinforced_concrete.code_checks.ec2.shear_check import ShearCheck


def main():
    print("=" * 70)
    print("EC2 CODE CHECKS EXAMPLE")
    print("=" * 70)
    print()

    # ========================================================================
    # STEP 1: Define section geometry
    # ========================================================================
    print("1. SECTION GEOMETRY")
    print("-" * 70)

    # Create 300×500 mm beam section
    section = create_rectangular_section(
        width=300,
        height=500,
        section_name="Beam B1",
    )

    # Add bottom tension reinforcement: 3×φ20
    bottom_bars = create_linear_rebar_layer(
        rebar=Rebar(diameter=20, grade="B500B"),
        n_bars=3,
        start_point=(50, 50),  # 50mm cover
        end_point=(250, 50),
        layer_name="bottom",
    )
    section.add_rebar_group(bottom_bars)

    # Add top compression reinforcement: 2×φ16
    top_bars = create_linear_rebar_layer(
        rebar=Rebar(diameter=16, grade="B500B"),
        n_bars=2,
        start_point=(100, 450),
        end_point=(200, 450),
        layer_name="top",
    )
    section.add_rebar_group(top_bars)

    print(f"Section: {section.section_name}")
    print(f"  Dimensions: 300 × 500 mm")
    print(f"  Bottom steel: 3φ20 (A_s = {bottom_bars.total_area:.0f} mm²)")
    print(f"  Top steel: 2φ16 (A_s = {top_bars.total_area:.0f} mm²)")
    print(f"  Reinforcement ratio: {section.reinforcement_ratio:.4f}")
    print(f"  Effective depth: {section.get_effective_depth('top'):.0f} mm")
    print()

    # ========================================================================
    # STEP 2: Define materials
    # ========================================================================
    print("2. MATERIALS")
    print("-" * 70)

    # Concrete
    concrete = ConcreteMaterial(
        grade="C30/37",
        gamma_c=1.5,  # Material factor for ULS
        alpha_cc=1.0,  # Long-term effects coefficient
    )

    print(f"Concrete: {concrete.grade}")
    print(f"  f_ck = {concrete.f_ck} MPa")
    print(f"  f_cd = {concrete.f_cd:.1f} MPa (with γ_c = {concrete.gamma_c})")
    print(f"  E_cm = {concrete.E_cm:.0f} MPa")
    print()

    # Shear reinforcement: φ10 links @ 200mm spacing, 2 legs
    shear_links = ShearRebar(
        diameter=10,
        spacing=200,
        n_legs=2,
        grade="B500B",
    )

    print(f"Shear reinforcement: φ{shear_links.diameter} @ {shear_links.spacing}mm")
    print(f"  Number of legs: {shear_links.n_legs}")
    print(f"  Total A_sw = {shear_links.area:.0f} mm²")
    print(f"  A_sw/s = {shear_links.area/shear_links.spacing:.2f} mm²/mm")
    print()

    # ========================================================================
    # STEP 3: BENDING CHECK (First Principles)
    # ========================================================================
    print("3. BENDING CHECK (FIRST PRINCIPLES - Strain Compatibility)")
    print("-" * 70)
    print("Method:")
    print("  • Fiber-based strain compatibility analysis")
    print("  • Plane sections remain plane")
    print("  • Force equilibrium: ΣF = N, ΣM = M")
    print("  • Constitutive models: EC2 stress-strain (Figs 3.3, 3.8)")
    print("  • Design strengths: f_cd, f_yd (with γ_c, γ_s factors)")
    print()

    # Create bending check
    bending_check = BendingCheck(
        section=section,
        concrete=concrete,
        concrete_model_type="parabola-rectangle",  # EC2 Fig 3.3
        steel_branch_type="inclined",  # With strain hardening
        n_fibers_width=20,
        n_fibers_height=30,
    )

    # Applied loads (factored for ULS)
    M_Ed = 120.0  # kN·m
    N_Ed = 0.0    # kN (no axial force)

    print(f"Applied loads:")
    print(f"  M_Ed = {M_Ed} kN·m")
    print(f"  N_Ed = {N_Ed} kN")
    print()

    # Perform check
    result_bending = bending_check.perform_check(M_Ed=M_Ed, N_Ed=N_Ed)

    print(f"Result: {result_bending}")
    print(f"  Status: {result_bending.status.value.upper()}")
    print(f"  Utilization: {result_bending.utilization:.1%}")
    print(f"  Demand: {result_bending.demand:.1f} kN·m")
    print(f"  Capacity: {result_bending.capacity:.1f} kN·m")
    print(f"  Reference: {result_bending.code_reference}")
    print()

    # Get moment capacity at zero axial force
    M_Rd_pos, M_Rd_neg = bending_check.get_moment_capacity(N_Ed=0)
    print(f"Moment capacity (N=0):")
    print(f"  M_Rd,pos = {M_Rd_pos:.1f} kN·m")
    print(f"  M_Rd,neg = {M_Rd_neg:.1f} kN·m")
    print()

    # ========================================================================
    # STEP 4: SHEAR CHECK (Codified Approach)
    # ========================================================================
    print("4. SHEAR CHECK (CODIFIED - EC2 Empirical Formulas)")
    print("-" * 70)
    print("Method:")
    print("  • EC2 §6.2 Variable Strut Inclination Method")
    print("  • Empirical formulas (NOT first principles)")
    print("  • Business logic:")
    print("    - V_Rd,c: Concrete shear resistance (Eq. 6.2)")
    print("    - V_Rd,s: Shear reinforcement (Eq. 6.8)")
    print("    - V_Rd,max: Compression strut crushing (Eq. 6.9)")
    print("  • Strut angle: 21.8° ≤ θ ≤ 45° (cot θ = 1.0 to 2.5)")
    print()

    # Create shear check
    shear_check = ShearCheck(
        section=section,
        concrete=concrete,
        shear_reinforcement=shear_links,
        N_Ed=0.0,  # No axial force
    )

    # Applied shear force (factored for ULS)
    V_Ed = 100.0  # kN
    cot_theta = 2.5  # Maximum value for economy

    print(f"Applied loads:")
    print(f"  V_Ed = {V_Ed} kN")
    print(f"  cot(θ) = {cot_theta} (θ = {shear_check.MIN_STRUT_ANGLE_DEGS:.1f}°)")
    print()

    # Calculate individual capacities
    V_Rd_c = shear_check.find_V_Rd_c()
    V_Rd_s = shear_check.find_V_Rd_s(cot_theta=cot_theta)
    V_Rd_max = shear_check.find_V_Rd_max(cot_theta=cot_theta)

    print(f"Capacities:")
    print(f"  V_Rd,c (concrete) = {V_Rd_c:.1f} kN")
    print(f"  V_Rd,s (shear reinf) = {V_Rd_s:.1f} kN")
    print(f"  V_Rd,max (strut crushing) = {V_Rd_max:.1f} kN")
    print()

    # Perform check
    result_shear = shear_check.perform_check(V_Ed=V_Ed, cot_theta=cot_theta)

    print(f"Result: {result_shear}")
    print(f"  Status: {result_shear.status.value.upper()}")
    print(f"  Utilization: {result_shear.utilization:.1%}")
    print(f"  Demand: {result_shear.demand:.1f} kN")
    print(f"  Capacity: {result_shear.capacity:.1f} kN")
    print(f"  Governing: {result_shear.details['governing_mode']}")
    print(f"  Reference: {result_shear.code_reference}")
    print()

    # ========================================================================
    # STEP 5: Check different load combinations
    # ========================================================================
    print("5. CHECK MULTIPLE LOAD CASES")
    print("-" * 70)

    load_cases = [
        {"name": "LC1: Service", "M_Ed": 80, "V_Ed": 60, "N_Ed": 0},
        {"name": "LC2: Ultimate", "M_Ed": 120, "V_Ed": 100, "N_Ed": 0},
        {"name": "LC3: Ultimate + Comp", "M_Ed": 140, "V_Ed": 120, "N_Ed": 200},
    ]

    for lc in load_cases:
        print(f"\n{lc['name']}:")
        print(f"  Loads: M={lc['M_Ed']} kN·m, V={lc['V_Ed']} kN, N={lc['N_Ed']} kN")

        # Update axial force for both checks
        shear_check_lc = ShearCheck(
            section=section,
            concrete=concrete,
            shear_reinforcement=shear_links,
            N_Ed=lc['N_Ed'],
        )

        # Bending check
        res_bend = bending_check.perform_check(M_Ed=lc['M_Ed'], N_Ed=lc['N_Ed'])
        print(f"  Bending: {res_bend.status.value.upper()} ({res_bend.utilization:.0%})")

        # Shear check
        res_shear = shear_check_lc.perform_check(V_Ed=lc['V_Ed'])
        print(f"  Shear:   {res_shear.status.value.upper()} ({res_shear.utilization:.0%})")

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print("BENDING CHECK (First Principles):")
    print("  ✓ Uses M-N interaction diagram")
    print("  ✓ Strain compatibility + force equilibrium")
    print("  ✓ Exact solution with fiber integration")
    print("  ✓ Accounts for actual stress-strain behavior")
    print()
    print("SHEAR CHECK (Codified Approach):")
    print("  ✓ Uses EC2 empirical formulas")
    print("  ✓ Business logic for different failure modes")
    print("  ✓ Variable strut angle method")
    print("  ✓ Fast calculation")
    print()
    print("Both approaches:")
    print("  • Work with same geometry/material definitions")
    print("  • Use codified material factors (γ_c, γ_s)")
    print("  • Return standardized CheckResult")
    print("  • Can be chained together for full section checks")
    print()


if __name__ == "__main__":
    main()
