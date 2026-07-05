"""Compare the old horizontal projection vs. new vector projection methods."""

from materials.reinforced_concrete.geometry import create_rectangular_section, create_linear_rebar_layer
from materials.reinforced_concrete.materials import ConcreteMaterial, Rebar
from materials.reinforced_concrete.analysis import create_interaction_diagram
import matplotlib.pyplot as plt
import numpy as np

# Create section
section = create_rectangular_section(width=300, height=500, section_name="Example Beam")
rebar_20 = Rebar(grade="B500B", diameter=20)

bottom_layer = create_linear_rebar_layer(rebar=rebar_20, n_bars=3, start_point=(50, 50), end_point=(250, 50), layer_name="bottom")
section.add_rebar_group(bottom_layer)

top_layer = create_linear_rebar_layer(rebar=rebar_20, n_bars=2, start_point=(75, 450), end_point=(225, 450), layer_name="top")
section.add_rebar_group(top_layer)

concrete = ConcreteMaterial(grade="C30/37", gamma_c=1.5, alpha_cc=0.85)
diagram = create_interaction_diagram(section=section, concrete=concrete, concrete_model_type="parabola-rectangle", steel_branch_type="inclined")

# Generate M-N curve
points = diagram.generate_diagram(n_points=100, include_tension=True)
M_values = [p.M for p in points]
N_values = [p.N for p in points]

# Test point to demonstrate the difference
N_Ed = 500.0
M_Ed = 200.0

# Get capacities using both methods
M_pos, M_neg = diagram.get_capacity(N_Ed)  # Horizontal projection
is_safe, util_vector = diagram.check_capacity(N_Ed, M_Ed)  # Vector projection

# Calculate old-style utilization for comparison
util_horizontal = abs(M_Ed) / M_pos if M_pos > 0 else float('inf')

# Create visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

# LEFT PLOT: Horizontal projection (old method)
ax1.plot(M_values, N_values, 'b-', linewidth=2.5, label='M-N Boundary', alpha=0.8)
ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=1)
ax1.axvline(x=0, color='k', linestyle='--', alpha=0.3, linewidth=1)

# Plot applied load point
ax1.plot(M_Ed, N_Ed, 'ro', markersize=12, label=f'Applied Load (M={M_Ed}, N={N_Ed})', zorder=5)

# Draw horizontal line to show capacity at this N
ax1.plot([0, M_pos], [N_Ed, N_Ed], 'r--', linewidth=2, alpha=0.7, label=f'Capacity at N={N_Ed}')
ax1.plot(M_pos, N_Ed, 'gs', markersize=12, label=f'Capacity Point (M={M_pos:.1f})', zorder=5)

# Show the two segments
ax1.plot([0, M_Ed], [N_Ed, N_Ed], 'orange', linewidth=4, alpha=0.7, label=f'Demand = {M_Ed}')
ax1.plot([M_Ed, M_pos], [N_Ed, N_Ed], 'green', linewidth=4, alpha=0.7, label=f'Reserve = {M_pos-M_Ed:.1f}')

ax1.annotate(f'Horizontal Utilization = {util_horizontal:.1%}',
            xy=(M_Ed/2, N_Ed), xytext=(0, 30),
            textcoords='offset points', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
            ha='center')

ax1.set_xlabel('Moment M (kN·m)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Axial Force N (kN)', fontsize=12, fontweight='bold')
ax1.set_title('OLD METHOD: Horizontal Projection\n(Incorrect for M-N diagrams)', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper right', fontsize=9)
ax1.set_xlim(-100, 300)
ax1.set_ylim(-500, 2000)

# RIGHT PLOT: Vector projection (new method)
ax2.plot(M_values, N_values, 'b-', linewidth=2.5, label='M-N Boundary', alpha=0.8)
ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=1)
ax2.axvline(x=0, color='k', linestyle='--', alpha=0.3, linewidth=1)

# Plot applied load point
ax2.plot(M_Ed, N_Ed, 'ro', markersize=12, label=f'Applied Load (M={M_Ed}, N={N_Ed})', zorder=5)

# Find the boundary intersection point by solving the ray-edge intersection
# (This is what check_capacity does internally)
M_boundary = None
N_boundary = None
max_alpha = 0.0

for i in range(len(M_values)):
    M1, N1 = M_values[i], N_values[i]
    M2, N2 = M_values[(i + 1) % len(M_values)], N_values[(i + 1) % len(N_values)]

    dM = M2 - M1
    dN = N2 - N1

    det = M_Ed * (-dN) - N_Ed * (-dM)

    if abs(det) < 1e-10:
        continue

    alpha = (M1 * (-dN) - N1 * (-dM)) / det
    s = (M_Ed * N1 - N_Ed * M1) / det

    if alpha > 1e-10 and 0 <= s <= 1:
        if alpha > max_alpha:
            max_alpha = alpha
            M_boundary = alpha * M_Ed
            N_boundary = alpha * N_Ed

if M_boundary is not None:
    # Draw vector from origin to boundary
    ax2.plot([0, M_boundary], [0, N_boundary], 'g--', linewidth=2, alpha=0.7,
             label=f'Capacity Vector (α={max_alpha:.3f})')
    ax2.plot(M_boundary, N_boundary, 'gs', markersize=12,
             label=f'Boundary Point (M={M_boundary:.1f}, N={N_boundary:.1f})', zorder=5)

    # Draw vector from origin to applied load
    ax2.plot([0, M_Ed], [0, N_Ed], 'r-', linewidth=3, alpha=0.8, label='Demand Vector')

    # Show the origin
    ax2.plot(0, 0, 'ko', markersize=10, label='Origin', zorder=5)

    # Annotate with distances
    demand_dist = np.sqrt(M_Ed**2 + N_Ed**2)
    capacity_dist = np.sqrt(M_boundary**2 + N_boundary**2)

    ax2.annotate(f'Vector Utilization = {util_vector:.1%}\n(demand/capacity = {demand_dist:.1f}/{capacity_dist:.1f})',
                xy=(M_Ed/2, N_Ed/2), xytext=(50, -50),
                textcoords='offset points', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))

ax2.set_xlabel('Moment M (kN·m)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Axial Force N (kN)', fontsize=12, fontweight='bold')
ax2.set_title('NEW METHOD: Vector Projection from Origin\n(Geometrically Correct)', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper right', fontsize=9)
ax2.set_xlim(-100, 300)
ax2.set_ylim(-500, 2000)

plt.tight_layout()
plt.savefig('utilization_methods_comparison.png', dpi=150)
print("Comparison plot saved as 'utilization_methods_comparison.png'")
print()

# Print comparison
print("=" * 70)
print("COMPARISON OF METHODS")
print("=" * 70)
print(f"Applied Load: N = {N_Ed:.0f} kN, M = {M_Ed:.0f} kN·m")
print()
print(f"OLD METHOD (Horizontal Projection at constant N):")
print(f"  Capacity at N={N_Ed}: M_Rd = {M_pos:.1f} kN·m")
print(f"  Utilization = M_Ed / M_Rd = {M_Ed:.0f} / {M_pos:.1f} = {util_horizontal:.2%}")
print()
print(f"NEW METHOD (Vector Projection from Origin):")
print(f"  Boundary intersection: (M={M_boundary:.1f}, N={N_boundary:.1f})")
print(f"  Demand magnitude: {demand_dist:.1f}")
print(f"  Capacity magnitude: {capacity_dist:.1f}")
print(f"  Utilization = ||demand|| / ||capacity|| = {util_vector:.2%}")
print()
print(f"Difference: {abs(util_horizontal - util_vector):.2%}")
print("=" * 70)
