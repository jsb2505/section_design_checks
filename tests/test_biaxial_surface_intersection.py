"""High-value tests for biaxial surface geometry and intersection."""

import numpy as np
import pytest
from scipy.spatial import ConvexHull

from materials.reinforced_concrete.analysis.biaxial_interaction import (
    BiaxialMNInteractionSurface,
    create_biaxial_interaction_surface,
)
from materials.reinforced_concrete.materials import ConcreteMaterial, Rebar
from materials.reinforced_concrete.geometry import (
    create_rectangular_section,
    create_linear_rebar_layer,
)


def _create_square_column() -> BiaxialMNInteractionSurface:
    """Square 400x400 mm column with corner bars."""
    section = create_rectangular_section(width=400, height=400, section_name="Test Column")
    rebar = Rebar(grade="B500B", diameter=20)
    cover = 50
    corners = [
        (cover, cover),
        (400 - cover, cover),
        (400 - cover, 400 - cover),
        (cover, 400 - cover),
    ]
    for idx, (x, y) in enumerate(corners):
        section.add_rebar_group(
            create_linear_rebar_layer(
                rebar=rebar,
                n_bars=1,
                start_point=(x, y),
                end_point=(x, y),
                layer_name=f"corner_{idx}",
            )
        )
    concrete = ConcreteMaterial(grade="C30/37", gamma_c=1.5, alpha_cc=0.85)
    return create_biaxial_interaction_surface(section=section, concrete=concrete)


def test_surface_symmetry_square_column():
    surface = _create_square_column()
    points = surface.generate_surface_pivot(n_angles=24, n_axial_levels=12)

    step = max(1, len(points) // 40)
    for point in points[::step]:
        mirrored_magnitudes = [
            abs(candidate.My + point.My) + abs(candidate.Mz + point.Mz)
            for candidate in points
            if np.isclose(candidate.N, point.N, atol=1e-3)
        ]
        assert mirrored_magnitudes, "No mirrored point found at matching axial level"
        assert min(mirrored_magnitudes) <= 15.0


def test_surface_is_convex():
    surface = _create_square_column()
    points = surface.generate_surface_pivot(n_angles=20, n_axial_levels=12)
    coords = np.array([[p.N, p.My, p.Mz] for p in points])
    hull = ConvexHull(coords)

    lhs = hull.equations[:, :3] @ coords.T + hull.equations[:, 3][:, None]
    assert np.all(lhs <= 1e-6)
    assert hull.volume > 0


def test_vector_intersection_on_surface():
    surface = _create_square_column()
    points = surface.generate_surface_pivot(n_angles=20, n_axial_levels=12)

    step = max(1, len(points) // 20)
    for point in points[::step]:
        _, _, _, is_safe, utilization = surface.get_capacity_vector_exact(
            point.N, point.My, point.Mz, surface_points=points
        )
        assert is_safe
        assert utilization == pytest.approx(1.0, rel=1e-3, abs=1e-3)
