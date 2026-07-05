"""
Generate a clean M-N interaction diagram with current implementation.
Demonstrates the improved point distribution with dense sampling in balanced region.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import matplotlib.pyplot as plt
import numpy as np
from section_design_checks.reinforced_concrete.materials import ConcreteMaterial, Rebar
from section_design_checks.reinforced_concrete.geometry import (
    create_rectangular_section,
    create_linear_rebar_layer,
)
from section_design_checks.reinforced_concrete.analysis.interaction_diagram import (
    create_interaction_diagram,
)

print("="*80)
print("Generating M-N Interaction Diagram with Improved Point Distribution")
print("="*80)
print()

# Create materials
concrete = ConcreteMaterial(grade="C30/37", gamma_c=1.5, alpha_cc=0.85)
rebar = Rebar(grade="B500B", diameter=20)

# Create beam section
beam = create_rectangular_section(width=300, height=500, section_name="Beam 300x500")

# Add bottom reinforcement (3×φ20)
bottom = create_linear_rebar_layer(
    rebar=rebar,
    n_bars=3,
    start_point=(50, 50),
    end_point=(250, 50),
    layer_name="bottom",
)
beam.add_rebar_group(bottom)

# Add top reinforcement (2×φ20)
top = create_linear_rebar_layer(
    rebar=rebar,
    n_bars=2,
    start_point=(100, 450),
    end_point=(200, 450),
    layer_name="top",
)
beam.add_rebar_group(top)

print(f"Section: {beam.section_name}")
print(f"  Bottom: 3x20mm")
print(f"  Top: 2x20mm")
print(f"  Reinforcement ratio: {beam.reinforcement_ratio:.3%}")
print()

# Create diagram with HIGH resolution
diagram = create_interaction_diagram(section=beam, concrete=concrete)

print("Generating M-N diagram with 200 points...")
points = diagram.generate_diagram(n_points=200, include_tension=True)
print(f"Generated {len(points)} points")
print()

# Extract data
N = np.array([p.N for p in points])
M = np.array([p.M for p in points])

# Analyze point distribution
print("Point Distribution Analysis:")
print(f"  N range: {N.min():.1f} to {N.max():.1f} kN")
print(f"  M range: {M.min():.1f} to {M.max():.1f} kN·m")

# Check spacing between consecutive points
distances = np.sqrt(np.diff(M)**2 + np.diff(N)**2)
print(f"\nPoint spacing (Euclidean distance):")
print(f"  Min:    {distances.min():.2f}")
print(f"  Max:    {distances.max():.2f}")
print(f"  Mean:   {distances.mean():.2f}")
print(f"  Std:    {distances.std():.2f}")
print(f"  Max/Min ratio: {distances.max()/distances.min():.2f}")
print()

# Create visualization
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Left: Full diagram
ax1 = axes[0]
ax1.plot(M, N, 'b-', linewidth=2, label='M-N Envelope')
ax1.plot(M, N, 'ko', markersize=3, alpha=0.4, label=f'Points (n={len(points)})')
ax1.axhline(0, color='k', linestyle='--', alpha=0.3, linewidth=1)
ax1.axvline(0, color='k', linestyle='--', alpha=0.3, linewidth=1)
ax1.set_xlabel('Moment M (kN·m)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Axial Force N (kN)', fontsize=12, fontweight='bold')
ax1.set_title('M-N Interaction Diagram\nBeam 300×500, 3φ20+2φ20, C30/37',
              fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(loc='best', fontsize=10)

# Right: Zoomed to balanced region to show point density
# Find balanced region (around N=0 to 50% of max compression)
balanced_mask = (N > 0) & (N < 0.5 * N.max())
if np.any(balanced_mask):
    M_balanced = M[balanced_mask]
    N_balanced = N[balanced_mask]

    ax2 = axes[1]
    ax2.plot(M, N, 'b-', linewidth=1, alpha=0.3, label='Full curve')
    ax2.plot(M_balanced, N_balanced, 'b-', linewidth=2, label='Balanced region')
    ax2.plot(M_balanced, N_balanced, 'ro', markersize=5, alpha=0.6,
            label=f'Points in region (n={np.sum(balanced_mask)})')
    ax2.axhline(0, color='k', linestyle='--', alpha=0.3, linewidth=1)
    ax2.axvline(0, color='k', linestyle='--', alpha=0.3, linewidth=1)
    ax2.set_xlabel('Moment M (kN·m)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Axial Force N (kN)', fontsize=12, fontweight='bold')
    ax2.set_title('Balanced Region (Zoomed)\nShowing Dense Point Sampling',
                  fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best', fontsize=10)

plt.tight_layout()
plt.savefig('output/figures/mn_diagram.png', dpi=150, bbox_inches='tight')
print(f"[OK] Saved diagram to: output/figures/mn_diagram.png")
# plt.show()  # Skip interactive display

print()
print("="*80)
print("[SUCCESS] M-N diagram with improved point distribution")
print("="*80)
print()
print("Key improvements:")
print("  - Dense sampling in balanced region (high curvature)")
print("  - Sparser sampling in linear regions (compression/tension)")
print("  - Smooth convex hull boundary")
print("  - 200 points for excellent resolution")
