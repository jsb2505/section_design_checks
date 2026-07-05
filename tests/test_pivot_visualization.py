"""
Visualize the PIVOT METHOD surface to verify no interior points.
"""

from materials.reinforced_concrete.geometry import (
    create_rectangular_section,
    create_linear_rebar_layer,
)
from materials.reinforced_concrete.analysis.biaxial_interaction import (
    BiaxialMNInteractionSurface,
)
from materials.reinforced_concrete.materials import ConcreteMaterial, Rebar
import plotly.graph_objects as go
import numpy as np

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

print("\nGenerating surface using PIVOT METHOD...")
points = biaxial.generate_surface_pivot(
    n_angles=36,
    n_axial_levels=16,
)

print(f"Generated {len(points)} points")

# Extract coordinates
N_vals = np.array([p.N for p in points])
My_vals = np.array([p.My for p in points])
Mz_vals = np.array([p.Mz for p in points])

print(f"N range: {N_vals.min():.1f} to {N_vals.max():.1f} kN")
print(f"My range: {My_vals.min():.1f} to {My_vals.max():.1f} kNm")
print(f"Mz range: {Mz_vals.min():.1f} to {Mz_vals.max():.1f} kNm")

# Create 3D scatter plot
fig = go.Figure()

fig.add_trace(go.Scatter3d(
    x=My_vals,
    y=Mz_vals,
    z=N_vals,
    mode='markers',
    marker=dict(
        size=3,
        color=N_vals,
        colorscale='Viridis',
        colorbar=dict(title="N (kN)"),
        showscale=True,
    ),
    name='M-M-N Surface (Pivot Method)',
    hovertemplate='<b>My:</b> %{x:.1f} kNm<br>' +
                  '<b>Mz:</b> %{y:.1f} kNm<br>' +
                  '<b>N:</b> %{z:.1f} kN<br>' +
                  '<extra></extra>',
))

fig.update_layout(
    title=dict(
        text='Biaxial M-M-N Interaction Surface<br><sub>EC2 Pivot Method - No Interior Points</sub>',
        x=0.5,
        xanchor='center',
    ),
    scene=dict(
        xaxis_title='My (kNm)',
        yaxis_title='Mz (kNm)',
        zaxis_title='N (kN)',
        aspectmode='data',
    ),
    width=1000,
    height=800,
)

print("\nOpening plot in browser...")
fig.show()
print("\nPlot displayed. Check for:")
print("  1. No interior points (especially in tension region)")
print("  2. Smooth uniform latitude rings")
print("  3. Clean convergence to pole point at top")
