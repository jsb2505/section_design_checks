"""
Example demonstrating the new M-N diagram export functionality.

This script shows how to:
1. Generate an M-N interaction diagram
2. Export to JSON format
3. Export to CSV format
4. Export to dictionary for programmatic use
"""

from section_design_checks.reinforced_concrete.materials import ConcreteMaterial, Rebar
from section_design_checks.reinforced_concrete.geometry import (
    create_rectangular_section,
    create_linear_rebar_layer,
)
from section_design_checks.reinforced_concrete.analysis import create_interaction_diagram


def main():
    """Demonstrate M-N diagram export functionality."""

    # Create a rectangular beam section
    section = create_rectangular_section(300, 500, section_name="Example Beam")

    # Add bottom reinforcement (tension)
    rebar_20 = Rebar(diameter=20, grade="B500B")
    bottom_layer = create_linear_rebar_layer(
        rebar=rebar_20,
        n_bars=3,
        start_point=(50, 50),
        end_point=(250, 50),
        layer_name="bottom",
    )
    section.add_rebar_group(bottom_layer)

    # Add top reinforcement (compression)
    top_layer = create_linear_rebar_layer(
        rebar=rebar_20,
        n_bars=2,
        start_point=(75, 450),
        end_point=(225, 450),
        layer_name="top",
    )
    section.add_rebar_group(top_layer)

    # Create concrete material
    concrete = ConcreteMaterial(grade="C30/37")

    # Generate M-N interaction diagram
    print("Generating M-N interaction diagram...")
    diagram = create_interaction_diagram(
        section=section,
        concrete=concrete,
        concrete_model_type="parabola-rectangle",
        steel_branch_type="inclined",
    )

    # Example 1: Export to JSON with metadata
    print("\n1. Exporting to JSON with metadata...")
    diagram.export_to_json(
        file_path="mn_diagram_full.json",
        n_points=50,
        include_metadata=True,
        indent=2,
    )
    print("   [OK] Saved to: mn_diagram_full.json")

    # Example 2: Export to CSV with all columns
    print("\n2. Exporting to CSV with strain data...")
    diagram.export_to_csv(
        file_path="mn_diagram_detailed.csv",
        n_points=50,
        include_strains=True,
    )
    print("   [OK] Saved to: mn_diagram_detailed.csv")

    # Example 3: Export simple CSV (N and M only)
    print("\n3. Exporting simplified CSV (N and M only)...")
    diagram.export_to_csv(
        file_path="mn_diagram_simple.csv",
        n_points=50,
        include_strains=False,
    )
    print("   [OK] Saved to: mn_diagram_simple.csv")

    # Example 4: Get as dictionary for programmatic use
    print("\n4. Getting diagram as dictionary...")
    data = diagram.to_dict(n_points=50, include_metadata=True)

    print(f"   Points generated: {len(data['points'])}")
    print(f"   N range: {min(data['N_array']):.1f} to {max(data['N_array']):.1f} kN")
    print(f"   M range: {min(data['M_array']):.1f} to {max(data['M_array']):.1f} kN·m")
    print(f"   Concrete grade: {data['metadata']['concrete_grade']}")
    print(f"   Number of fibers: {data['metadata']['n_fibers']}")

    # Example 5: Use with pandas
    print("\n5. Converting to pandas DataFrame...")
    try:
        import pandas as pd
        df = pd.DataFrame(data["points"])
        print("\n   First 5 points:")
        print(df.head().to_string(index=False))

        # Save to CSV using pandas
        df.to_csv("mn_diagram_pandas.csv", index=False)
        print("\n   [OK] Saved DataFrame to: mn_diagram_pandas.csv")
    except ImportError:
        print("   (pandas not installed - skipping)")

    print("\n[SUCCESS] All exports completed successfully!")
    print("\nGenerated files:")
    print("  - mn_diagram_full.json (JSON with metadata)")
    print("  - mn_diagram_detailed.csv (CSV with strain data)")
    print("  - mn_diagram_simple.csv (CSV with N and M only)")
    if 'pd' in locals():
        print("  - mn_diagram_pandas.csv (pandas export)")


if __name__ == "__main__":
    main()
