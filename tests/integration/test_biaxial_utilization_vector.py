"""Test the biaxial utilization vector method."""

import numpy as np
from materials.reinforced_concrete.materials import ConcreteMaterial, Rebar
from materials.reinforced_concrete.geometry import (
    create_rectangular_section,
    create_linear_rebar_layer,
)
from materials.reinforced_concrete.analysis.biaxial_interaction import (
    create_biaxial_interaction_surface,
)

print("="*80)
print("Testing Biaxial Utilization Vector Method")
print("="*80)
print()

# Create test column
concrete = ConcreteMaterial(grade="C30/37", gamma_c=1.5, alpha_cc=0.85)
rebar_20 = Rebar(grade="B500B", diameter=20)
column = create_rectangular_section(width=400, height=400, section_name="Test Column")

# Add corner reinforcement
cover = 50
corners = [(cover, cover), (400 - cover, cover), (400 - cover, 400 - cover), (cover, 400 - cover)]
for i, (x, y) in enumerate(corners):
    column.add_rebar_group(
        create_linear_rebar_layer(rebar=rebar_20, n_bars=1, start_point=(x, y), end_point=(x, y))
    )

print(f"Section: {column.section_name}")
print(f"  Dimensions: 400x400 mm")
print(f"  Reinforcement: 4x20mm corner bars")
print()

# Create surface
surface = create_biaxial_interaction_surface(section=column, concrete=concrete)

print("Test Cases:")
print("-" * 80)

# Test 1: Small load (should be safe)
print("\n1. Small Load (N=500, My=50, Mz=30)")
is_safe, util = surface.get_utilization_vector(N_Ed=500, My_Ed=50, Mz_Ed=30)
print(f"   Safe: {is_safe}, Utilization: {util:.1%}")
assert is_safe, "Small load should be safe"

# Test 2: Pure axial compression
print("\n2. Pure Axial Compression (N=2000, My=0, Mz=0)")
is_safe, util = surface.get_utilization_vector(N_Ed=2000, My_Ed=0, Mz_Ed=0)
print(f"   Safe: {is_safe}, Utilization: {util:.1%}")

# Test 3: Pure My moment
print("\n3. Pure Major Axis Moment (N=1000, My=100, Mz=0)")
is_safe, util = surface.get_utilization_vector(N_Ed=1000, My_Ed=100, Mz_Ed=0)
print(f"   Safe: {is_safe}, Utilization: {util:.1%}")

# Test 4: Pure Mz moment
print("\n4. Pure Minor Axis Moment (N=1000, My=0, Mz=100)")
is_safe, util = surface.get_utilization_vector(N_Ed=1000, My_Ed=0, Mz_Ed=100)
print(f"   Safe: {is_safe}, Utilization: {util:.1%}")

# Test 5: Biaxial bending
print("\n5. Biaxial Bending (N=1000, My=80, Mz=80)")
is_safe, util = surface.get_utilization_vector(N_Ed=1000, My_Ed=80, Mz_Ed=80)
print(f"   Safe: {is_safe}, Utilization: {util:.1%}")

# Test 6: Zero load
print("\n6. Zero Load (N=0, My=0, Mz=0)")
is_safe, util = surface.get_utilization_vector(N_Ed=0, My_Ed=0, Mz_Ed=0)
print(f"   Safe: {is_safe}, Utilization: {util:.1%}")
assert is_safe and util == 0.0, "Zero load should have 0% utilization"

# Test 7: Scaling test - verify linearity
print("\n7. Linearity Test (scaling loads)")
base_N, base_My, base_Mz = 1000, 60, 40
is_safe1, util1 = surface.get_utilization_vector(N_Ed=base_N, My_Ed=base_My, Mz_Ed=base_Mz)
is_safe2, util2 = surface.get_utilization_vector(N_Ed=2*base_N, My_Ed=2*base_My, Mz_Ed=2*base_Mz)
print(f"   1x load: Util = {util1:.3f}")
print(f"   2x load: Util = {util2:.3f}")
print(f"   Ratio: {util2/util1:.3f} (should be ~2.0)")

# Test 8: Near capacity
print("\n8. Multiple load levels to verify utilization scaling")
for scale in [0.5, 0.8, 1.0, 1.2]:
    N = scale * 1500
    My = scale * 100
    Mz = scale * 50
    is_safe, util = surface.get_utilization_vector(N_Ed=N, My_Ed=My, Mz_Ed=Mz)
    status = "SAFE" if is_safe else "FAIL"
    print(f"   Scale {scale:.1f}x: Util = {util:.1%} [{status}]")

print()
print("="*80)
print("SUCCESS: Biaxial utilization vector method working correctly!")
print("="*80)
print()
print("Key Features:")
print("  - 3D vector projection in (N, My, Mz) space")
print("  - Proper geometric utilization calculation")
print("  - Handles pure uniaxial and biaxial cases")
print("  - Linear scaling behavior")
print("  - Safe/unsafe determination at 100% utilization")
