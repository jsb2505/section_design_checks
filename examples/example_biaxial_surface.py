"""
Example demonstrating biaxial M-M-N interaction surface generation and visualization.

This script shows how to:
1. Generate a 3D biaxial interaction surface
2. Export the surface data
3. Visualize the surface in 3D using matplotlib
"""

from section_design_checks.reinforced_concrete.materials import ConcreteMaterial, Rebar
from section_design_checks.reinforced_concrete.geometry import (
    create_rectangular_section,
    create_linear_rebar_layer,
)
from section_design_checks.reinforced_concrete.analysis import create_biaxial_interaction_surface


def main():
    """Demonstrate biaxial M-M-N surface generation and visualization."""

    print("=" * 70)
    print("Biaxial M-M-N Interaction Surface Example")
    print("=" * 70)

    # Create a square column section
    print("\n1. Creating square column section (400mm x 400mm)...")
    section = create_rectangular_section(400, 400, section_name="Square Column")

    # Add corner reinforcement
    rebar = Rebar(diameter=20, grade="B500B")
    corners = [
        (50, 50, "corner_1"),
        (350, 50, "corner_2"),
        (350, 350, "corner_3"),
        (50, 350, "corner_4"),
    ]

    for x, y, name in corners:
        layer = create_linear_rebar_layer(
            rebar=rebar,
            n_bars=1,
            start_point=(x, y),
            end_point=(x, y),
            layer_name=name,
        )
        section.add_rebar_group(layer)

    print(f"   [OK] Section created with {len(section.rebar_groups)} corner bars")

    # Create concrete material
    concrete = ConcreteMaterial(grade="C30/37")
    print(f"   [OK] Concrete: {concrete.grade} (fck = {concrete.f_ck} MPa)")

    # Generate biaxial interaction surface
    print("\n2. Generating biaxial M-M-N interaction surface...")
    surface = create_biaxial_interaction_surface(
        section=section,
        concrete=concrete,
        concrete_model_type="parabola-rectangle",
        steel_branch_type="inclined",
    )
    print(f"   [OK] Surface generator created")

    # Generate surface points
    print("\n3. Computing surface points...")
    print("   (This may take a moment - generating points at multiple angles and depths)")
    n_angles = 16  # Number of neutral axis angles
    n_depths = 20  # Number of depths per angle

    points = surface.generate_surface(n_angles=n_angles, n_depths=n_depths)
    print(f"   [OK] Generated {len(points)} points on the surface")

    # Extract data for analysis
    N_values = [p.N for p in points]
    Mx_values = [p.Mx for p in points]
    My_values = [p.My for p in points]

    print(f"\n   Surface statistics:")
    print(f"   N  range: {min(N_values):.1f} to {max(N_values):.1f} kN")
    print(f"   Mx range: {min(Mx_values):.1f} to {max(Mx_values):.1f} kN·m")
    print(f"   My range: {min(My_values):.1f} to {max(My_values):.1f} kN·m")

    # Export to files
    print("\n4. Exporting surface data...")
    surface.export_to_json("biaxial_surface.json", n_angles=n_angles, n_depths=n_depths)
    print("   [OK] Saved to: biaxial_surface.json")

    surface.export_to_csv("biaxial_surface.csv", n_angles=n_angles, n_depths=n_depths)
    print("   [OK] Saved to: biaxial_surface.csv")

    # 3D Visualization
    print("\n5. Creating 3D visualization...")
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=(14, 6))

        # Plot 1: 3D Surface with scatter points
        ax1 = fig.add_subplot(121, projection='3d')
        scatter = ax1.scatter(Mx_values, My_values, N_values, c=N_values, cmap='viridis', s=10, alpha=0.6)
        ax1.set_xlabel('Mx (kN·m)', fontsize=10)
        ax1.set_ylabel('My (kN·m)', fontsize=10)
        ax1.set_zlabel('N (kN)', fontsize=10)
        ax1.set_title('Biaxial M-M-N Interaction Surface\n(3D Scatter)', fontsize=12, fontweight='bold')
        plt.colorbar(scatter, ax=ax1, label='Axial Force N (kN)', shrink=0.5)

        # Plot 2: 2D projection (Mx vs My at constant N slices)
        ax2 = fig.add_subplot(122)

        # Find points at different N levels
        N_levels = [min(N_values), 0, max(N_values) / 2, max(N_values)]
        colors = ['red', 'blue', 'green', 'purple']

        for N_level, color in zip(N_levels, colors):
            # Find points close to this N level
            tolerance = abs(max(N_values) - min(N_values)) * 0.1
            close_points = [p for p in points if abs(p.N - N_level) < tolerance]

            if close_points:
                Mx_slice = [p.Mx for p in close_points]
                My_slice = [p.My for p in close_points]
                ax2.scatter(Mx_slice, My_slice, c=color, s=20, alpha=0.6, label=f'N ≈ {N_level:.0f} kN')

        ax2.set_xlabel('Mx (kN·m)', fontsize=10)
        ax2.set_ylabel('My (kN·m)', fontsize=10)
        ax2.set_title('2D Projections at Constant N Levels', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
        ax2.axvline(x=0, color='k', linestyle='--', linewidth=0.5)
        ax2.set_aspect('equal', adjustable='box')

        plt.tight_layout()
        plt.savefig('biaxial_surface_3d.png', dpi=150, bbox_inches='tight')
        print("   [OK] Saved visualization to: biaxial_surface_3d.png")

        plt.show()

    except ImportError as e:
        print(f"   [SKIP] Matplotlib not available for visualization: {e}")

    # Example: Check a biaxial loading case
    print("\n6. Example: Checking a biaxial loading case...")
    print("   Applied loads: N = 1000 kN, Mx = 100 kN·m, My = 80 kN·m")
    print("   (Capacity checking for biaxial loading requires interpolation on the surface)")
    print("   (This is an advanced feature - consult EC2 for interaction formulas)")

    print("\n" + "=" * 70)
    print("[SUCCESS] Biaxial surface generation complete!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - biaxial_surface.json (surface data with metadata)")
    print("  - biaxial_surface.csv (surface points in tabular format)")
    print("  - biaxial_surface_3d.png (3D visualization)")
    print("\nThe surface can be used for:")
    print("  - Biaxial bending capacity checks")
    print("  - Column design under combined loads")
    print("  - Understanding section behavior under arbitrary bending directions")


if __name__ == "__main__":
    main()
