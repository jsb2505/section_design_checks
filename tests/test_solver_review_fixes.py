"""Regression tests for the biaxial M-M-N surface pivot method.

Validates the theoretical axial limits, the balanced-depth transition, and the
tangent-mapping surface solver. These were previously a module-level script that
executed its assertions at import time; they are now proper, slow-marked tests so
they are collectable, selectable, and do not run during the fast lane.

Run with: ``pytest -m slow tests/test_solver_review_fixes.py``
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
    """400x400 mm column, 4H20 corner bars, C30/37."""
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


def test_theoretical_axial_limits(square_column_surface):
    """Pure tension is negative, pure compression positive and larger in magnitude."""
    n_min, n_max = square_column_surface.calculate_axial_limits()
    assert n_min < 0, "Pure tension should be negative"
    assert n_max > 0, "Pure compression should be positive"
    assert n_max > abs(n_min), "Compression should exceed tension (concrete contributes)"


def test_balanced_depth_point_within_axial_limits(square_column_surface):
    """A pivot point at an interior NA depth stays within the theoretical N limits."""
    n_min, n_max = square_column_surface.calculate_axial_limits()
    point = square_column_surface.calculate_point_pivot(na_depth=150.0, neutral_axis_angle=0.0)
    assert n_min <= point.N <= n_max


def test_surface_generation_tangent_mapping(square_column_surface):
    """Surface generation yields the full grid at uniform N levels (no solver dropouts)."""
    points = square_column_surface.generate_surface_pivot(n_angles=24, n_axial_levels=12)
    assert len(points) == 24 * 12, "Tangent mapping should produce every requested point"
    unique_n = np.unique(np.array([p.N for p in points]).round(decimals=1))
    assert len(unique_n) == 12, "Should have exactly 12 uniform N levels"


def test_balanced_depth_transition_is_stable(square_column_surface):
    """N is continuous (no divot) across the balanced-depth failure-mode transition."""
    eps_cu2, eps_ud, d_eff = 0.0035, 0.02, 350.0
    x_bal = (eps_cu2 / (eps_cu2 + eps_ud)) * d_eff
    before = square_column_surface.calculate_point_pivot(na_depth=x_bal - 10, neutral_axis_angle=0)
    after = square_column_surface.calculate_point_pivot(na_depth=x_bal + 10, neutral_axis_angle=0)
    assert abs(after.N - before.N) < 1000, "N jump across balanced depth should be finite/reasonable"
