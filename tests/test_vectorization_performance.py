"""Correctness regression tests for the vectorized biaxial pivot calculation.

Previously a module-level timing script; converted to proper slow-marked tests
that assert the vectorized ``calculate_point_pivot`` / ``generate_surface_pivot``
produce valid, complete results. (Wall-clock timing is intentionally not asserted
— it is environment-dependent and not a meaningful regression signal.)

Run with: ``pytest -m slow tests/test_vectorization_performance.py``
"""

import numpy as np
import pytest

from materials.reinforced_concrete.analysis.biaxial_interaction import (
    BiaxialMNInteractionSurface,
)
from materials.reinforced_concrete.geometry import (
    create_linear_rebar_layer,
    create_rectangular_section,
)
from materials.reinforced_concrete.materials import ConcreteMaterial, Rebar

pytestmark = pytest.mark.slow


@pytest.fixture
def square_column_surface() -> BiaxialMNInteractionSurface:
    section = create_rectangular_section(400, 400, section_name="Square Column")
    rebar_20 = Rebar(diameter=20, grade="B500B")
    for i, (x, y) in enumerate([(50, 50), (350, 50), (350, 350), (50, 350)]):
        section.add_rebar_group(
            create_linear_rebar_layer(
                rebar=rebar_20, n_bars=1, start_point=(x, y),
                end_point=(x, y), layer_name=f"corner_{i}",
            )
        )
    return BiaxialMNInteractionSurface(section=section, concrete=ConcreteMaterial(grade="C30/37"))


@pytest.mark.parametrize("na_depth", [50, 100, 150, 200, 300])
@pytest.mark.parametrize("angle", [0, 45, 90])
def test_point_pivot_produces_valid_results(square_column_surface, na_depth, angle):
    """Vectorized point calculation returns finite, physically plausible (N, My, Mz)."""
    point = square_column_surface.calculate_point_pivot(na_depth, angle)
    assert -1000 < point.N < 5000, f"N out of range: {point.N}"
    assert -500 < point.My < 500, f"My out of range: {point.My}"
    assert -500 < point.Mz < 500, f"Mz out of range: {point.Mz}"


def test_surface_generation_is_complete_and_uniform(square_column_surface):
    """High-resolution surface generates every requested point at uniform N levels."""
    n_angles, n_levels = 36, 16
    points = square_column_surface.generate_surface_pivot(n_angles=n_angles, n_axial_levels=n_levels)
    assert len(points) == n_angles * n_levels
    unique_n = np.unique(np.array([p.N for p in points]).round(decimals=1))
    assert len(unique_n) == n_levels
