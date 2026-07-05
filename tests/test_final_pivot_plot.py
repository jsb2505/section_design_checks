"""
Final test: Use the standard plot() method which now uses PIVOT METHOD internally.
This demonstrates the complete fix for interior points issue.
"""

from materials.reinforced_concrete.geometry import (
    create_rectangular_section,
    create_linear_rebar_layer,
)
from materials.reinforced_concrete.analysis.biaxial_interaction import (
    BiaxialMNInteractionSurface,
)
from materials.reinforced_concrete.materials import ConcreteMaterial, Rebar

# Create test section
print("Creating 400x400mm square column with 4H20 corner bars...")
section = create_rectangular_section(400, 400, section_name="Square Column")
rebar_20 = Rebar(diameter=20, grade="B500B")

corners = [(50, 50), (350, 50), (350, 350), (50, 350)]
for i, (x, y) in enumerate(corners):
    layer = create_linear_rebar_layer(
        rebar=rebar_20, n_bars=1, start_point=(x, y),
        end_point=(x, y), layer_name=f"corner_{i}"
    )
    section.add_rebar_group(layer)

concrete_c30 = ConcreteMaterial(grade="C30/37")
biaxial = BiaxialMNInteractionSurface(section=section, concrete=concrete_c30)

print("\nGenerating plot using standard plot() method...")
print("(Now using EC2 Pivot Method internally)")
print()

# Add some test load points
load_points = [
    {"N_Ed": 1500, "My_Ed": 100, "Mz_Ed": 80, "name": "LC1: Compression"},
    {"N_Ed": 500, "My_Ed": 150, "Mz_Ed": 120, "name": "LC2: Low Axial"},
    {"N_Ed": -200, "My_Ed": 50, "Mz_Ed": 40, "name": "LC3: Tension"},
]

# Standard plot call - now using pivot method internally
fig = biaxial.plot(
    load_points=load_points,
    show_vectors=True,
    n_angles=36,
    n_axial_levels=16,
    title="Biaxial M-M-N Surface - EC2 Pivot Method",
    show=True
)

print()
print("Plot displayed in browser.")
print()
print("Expected Results:")
print("  - No interior points (especially in tension region)")
print("  - Uniform latitude rings (16 levels)")
print("  - Clean pole point at top (pure compression)")
print("  - All 3 load points should be visible with utilization ratios")
print("  - Projection vectors from origin to capacity surface")
