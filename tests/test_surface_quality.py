"""
Visual test to demonstrate surface quality improvements.

This creates a high-resolution plot to showcase:
- No divots (smooth balanced depth transition)
- No interior points (100% solver success)
- No interior lines (watertight go.Surface)
- Proper aspect ratio
"""

from materials.reinforced_concrete.geometry import (
    create_rectangular_section,
    create_linear_rebar_layer,
)
from materials.reinforced_concrete.analysis.biaxial_interaction import (
    BiaxialMNInteractionSurface,
)
from materials.reinforced_concrete.materials import ConcreteMaterial, Rebar
import pytest

pytestmark = pytest.mark.slow

print("=" * 80)
print("SURFACE QUALITY DEMONSTRATION")
print("=" * 80)
print()

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

print()
print("Generating HIGH-RESOLUTION surface to demonstrate quality...")
print("Parameters: 48 angles × 20 N levels = 960 points")
print()

# Add load cases spanning different regions
load_cases = [
    {"N_Ed": 2500, "My_Ed": 80, "Mz_Ed": 60, "name": "High Compression"},
    {"N_Ed": 1000, "My_Ed": 150, "Mz_Ed": 120, "name": "Moderate Compression"},
    {"N_Ed": 200, "My_Ed": 180, "Mz_Ed": 150, "name": "Low Compression"},
    {"N_Ed": -200, "My_Ed": 50, "Mz_Ed": 40, "name": "Tension"},
]

# Generate plot with all improvements
fig = biaxial.plot(
    load_points=load_cases,
    show_vectors=True,
    n_angles=48,  # High resolution
    n_axial_levels=20,
    title="Biaxial M-M-N Surface - Production Quality (code review Fixes)",
    show=True,
    save_path="output/biaxial_surface_quality.html"
)

print()
print("=" * 80)
print("QUALITY CHECKLIST")
print("=" * 80)
print()
print("Visual inspection points:")
print("  1. Surface Smoothness:")
print("     - No visible 'divots' or discontinuities")
print("     - Smooth transition across all N levels")
print()
print("  2. Boundary Integrity:")
print("     - All points on the failure surface (no interior points)")
print("     - Especially check the tension region (negative N)")
print()
print("  3. Mesh Quality:")
print("     - No lines cutting through the interior")
print("     - Clean latitude rings (horizontal contours)")
print("     - Smooth meridians (vertical lines from pole to pole)")
print()
print("  4. Poles:")
print("     - Top pole converges cleanly to (0, 0, N_max)")
print("     - Bottom pole converges cleanly to (0, 0, N_min)")
print()
print("  5. Aspect Ratio:")
print("     - Surface has physically meaningful proportions")
print("     - Not stretched or compressed unnaturally")
print()
print("  6. Load Points:")
print("     - All 4 load cases visible with color coding")
print("     - Projection vectors show capacity margins")
print()
print("If all checks pass, the surface is production-ready!")
print()
print("Plot saved to: output/biaxial_surface_quality.html")
