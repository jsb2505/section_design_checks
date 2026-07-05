"""
Example: Reinforced Concrete Beam Design

Demonstrates:
1. Creating materials (concrete and steel)
2. Defining a rectangular beam section
3. Adding reinforcement layers
4. Creating fiber mesh
5. Evaluating stress-strain models
"""

from materials.reinforced_concrete.materials import ConcreteMaterial, Rebar
from materials.reinforced_concrete.constitutive import (
    create_concrete_stress_strain,
    create_steel_stress_strain,
)
from materials.reinforced_concrete.geometry import (
    create_rectangular_section,
    create_linear_rebar_layer,
    FiberMesh,
)


def main():
    print("=" * 70)
    print("REINFORCED CONCRETE BEAM EXAMPLE")
    print("=" * 70)

    # =========================================================================
    # 1. DEFINE MATERIALS
    # =========================================================================
    print("\n1. MATERIALS")
    print("-" * 70)

    # Concrete C30/37
    concrete = ConcreteMaterial(
        grade="C30/37",
        gamma_c=1.5,
        alpha_cc=1.0,
        aggregate_type="quartzite",
    )

    print(f"Concrete: {concrete}")
    print(f"  f_ck = {concrete.f_ck:.1f} MPa")
    print(f"  f_cd = {concrete.f_cd:.1f} MPa")
    print(f"  f_ctm = {concrete.f_ctm:.2f} MPa")
    print(f"  E_cm = {concrete.E_cm:.0f} MPa")
    print(f"  ε_cu2 = {concrete.epsilon_cu2 * 1000:.2f}‰")

    # Steel B500B
    steel_bottom = Rebar(
        grade="B500B",
        diameter=20,
        name="Bottom reinforcement",
    )

    steel_top = Rebar(
        grade="B500B",
        diameter=12,
        name="Top reinforcement",
    )

    print(f"\n{steel_bottom}")
    print(f"  f_yk = {steel_bottom.f_yk:.1f} MPa")
    print(f"  f_yd = {steel_bottom.f_yd:.1f} MPa")
    print(f"  Area = {steel_bottom.area:.1f} mm²")

    # =========================================================================
    # 2. DEFINE SECTION GEOMETRY
    # =========================================================================
    print("\n2. SECTION GEOMETRY")
    print("-" * 70)

    # Create 300×500 mm beam
    section = create_rectangular_section(
        width=300,
        height=500,
        section_name="Beam B1",
    )

    print(f"Section: {section.section_name}")
    print(f"  Width = 300 mm")
    print(f"  Height = 500 mm")
    print(f"  Gross area = {section.get_area():.0f} mm²")

    cx, cy = section.get_centroid()
    print(f"  Centroid = ({cx:.1f}, {cy:.1f}) mm")

    # =========================================================================
    # 3. ADD REINFORCEMENT
    # =========================================================================
    print("\n3. REINFORCEMENT")
    print("-" * 70)

    # Bottom layer: 4×ϕ20 @ 50mm cover
    bottom_layer = create_linear_rebar_layer(
        rebar=steel_bottom,
        n_bars=4,
        start_point=(50, 50),
        end_point=(250, 50),
        layer_name="bottom",
    )
    section.add_rebar_group(bottom_layer)

    print(f"Bottom layer: {bottom_layer.n_bars}×ϕ{steel_bottom.diameter}")
    print(f"  Total area = {bottom_layer.total_area:.1f} mm²")

    # Top layer: 2×ϕ12 @ 50mm cover
    top_layer = create_linear_rebar_layer(
        rebar=steel_top,
        n_bars=2,
        start_point=(100, 450),
        end_point=(200, 450),
        layer_name="top",
    )
    section.add_rebar_group(top_layer)

    print(f"\nTop layer: {top_layer.n_bars}×ϕ{steel_top.diameter}")
    print(f"  Total area = {top_layer.total_area:.1f} mm²")

    print(f"\nTotal steel area = {section.total_steel_area:.1f} mm²")
    print(f"Reinforcement ratio = {section.reinforcement_ratio:.4f} ({section.reinforcement_ratio*100:.2f}%)")

    d = section.get_effective_depth(reference="top")
    print(f"Effective depth d = {d:.1f} mm")

    # =========================================================================
    # 4. STRESS-STRAIN MODELS
    # =========================================================================
    print("\n4. STRESS-STRAIN MODELS")
    print("-" * 70)

    # Concrete model (parabola-rectangle)
    concrete_model = create_concrete_stress_strain(
        concrete=concrete,
        model_type="parabola-rectangle",
        use_characteristic=False,  # Use f_cd
    )

    print(f"Concrete model: {concrete_model.name}")
    print(f"  Model type: Parabola-Rectangle (EC2 Fig 3.3)")
    print(f"  Design strength = {concrete_model.f_c:.1f} MPa")
    print(f"  ε_c2 = {concrete.epsilon_c2 * 1000:.2f}‰")
    print(f"  ε_cu2 = {concrete.epsilon_cu2 * 1000:.2f}‰")

    # Test at various strains
    test_strains = [0.001, 0.002, 0.003, 0.0035]
    print("\n  Strain-Stress evaluation:")
    for strain in test_strains:
        stress = concrete_model.get_stress(strain)
        print(f"    ε = {strain*1000:.2f}‰ → σ = {stress:.2f} MPa")

    # Steel model
    steel_model = create_steel_stress_strain(
        steel=steel_bottom,
        branch_type="inclined",  # With strain hardening
        use_characteristic=False,
    )

    print(f"\nSteel model: {steel_model.name}")
    print(f"  Model type: Inclined top branch (EC2 Fig 3.8)")
    print(f"  Design yield = {steel_model.f_y:.1f} MPa")
    print(f"  ε_yd = {steel_model.epsilon_y * 1000:.2f}‰")
    print(f"  ε_ud = {steel_bottom.epsilon_ud * 1000:.2f}‰")

    test_strains_steel = [0.001, 0.00217, 0.01, 0.025]
    print("\n  Strain-Stress evaluation:")
    for strain in test_strains_steel:
        stress = steel_model.get_stress(strain)
        print(f"    ε = {strain*1000:.2f}‰ → σ = {stress:.1f} MPa")

    # =========================================================================
    # 5. FIBER MESH
    # =========================================================================
    print("\n5. FIBER MESH GENERATION")
    print("-" * 70)

    mesh = FiberMesh(
        section=section,
        n_fibers_width=15,
        n_fibers_height=25,
        exclude_steel_area=True,
    )

    print(f"Fiber mesh: {mesh}")
    print(f"  Concrete fibers = {mesh.n_concrete_fibers}")
    print(f"  Steel fibers = {mesh.n_steel_fibers}")
    print(f"  Total fibers = {mesh.total_fibers}")

    # Get fiber arrays
    x, y, area, material_type, material_index = mesh.get_fiber_arrays()

    print(f"\n  Fiber statistics:")
    print(f"    Total concrete area = {area[material_type == 'concrete'].sum():.0f} mm²")
    print(f"    Total steel area = {area[material_type == 'steel'].sum():.0f} mm²")
    print(f"    Y-coordinate range = [{y.min():.1f}, {y.max():.1f}] mm")

    # =========================================================================
    # 6. SECTION PROPERTIES
    # =========================================================================
    print("\n6. SECTION PROPERTIES SUMMARY")
    print("-" * 70)

    print(f"Section: {section.section_name}")
    print(f"  Dimensions: 300 × 500 mm")
    print(f"  Gross concrete area: {section.get_area():.0f} mm²")
    print(f"  Steel area (bottom): {bottom_layer.total_area:.0f} mm²")
    print(f"  Steel area (top): {top_layer.total_area:.0f} mm²")
    print(f"  Total steel area: {section.total_steel_area:.0f} mm²")
    print(f"  Reinforcement ratio: {section.reinforcement_ratio:.4f}")
    print(f"  Effective depth: {d:.1f} mm")
    print(f"\nMaterials:")
    print(f"  Concrete: {concrete.grade} (f_cd = {concrete.f_cd:.1f} MPa)")
    print(f"  Steel: {steel_bottom.grade} (f_yd = {steel_bottom.f_yd:.1f} MPa)")

    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
