"""Integration tests for the biaxial utilization-vector method.

Validates ``get_utilization_vector`` (3D vector projection in (N, My, Mz) space)
across uniaxial, biaxial, zero and scaled load cases. Previously a module-level
script; converted to proper integration/slow tests.

Run with: ``pytest -m slow tests/integration/test_biaxial_utilization_vector.py``
"""

import pytest

from materials.reinforced_concrete.analysis.biaxial_interaction import (
    create_biaxial_interaction_surface,
)
from materials.reinforced_concrete.geometry import (
    create_linear_rebar_layer,
    create_rectangular_section,
)
from materials.reinforced_concrete.materials import ConcreteMaterial, Rebar

pytestmark = [pytest.mark.integration, pytest.mark.slow]


@pytest.fixture
def column_surface():
    concrete = ConcreteMaterial(grade="C30/37", gamma_c=1.5, alpha_cc=0.85)
    rebar_20 = Rebar(grade="B500B", diameter=20)
    column = create_rectangular_section(width=400, height=400, section_name="Test Column")
    cover = 50
    corners = [(cover, cover), (400 - cover, cover), (400 - cover, 400 - cover), (cover, 400 - cover)]
    for x, y in corners:
        column.add_rebar_group(
            create_linear_rebar_layer(rebar=rebar_20, n_bars=1, start_point=(x, y), end_point=(x, y))
        )
    return create_biaxial_interaction_surface(section=column, concrete=concrete)


def test_small_load_is_safe(column_surface):
    is_safe, util = column_surface.get_utilization_vector(N_Ed=500, My_Ed=50, Mz_Ed=30)
    assert is_safe
    assert 0.0 < util < 1.0


def test_zero_load_is_zero_utilization(column_surface):
    is_safe, util = column_surface.get_utilization_vector(N_Ed=0, My_Ed=0, Mz_Ed=0)
    assert is_safe
    assert util == pytest.approx(0.0)


@pytest.mark.parametrize(
    "N_Ed,My_Ed,Mz_Ed",
    [(2000, 0, 0), (1000, 100, 0), (1000, 0, 100), (1000, 80, 80)],
)
def test_uniaxial_and_biaxial_cases_return_bounded_utilization(column_surface, N_Ed, My_Ed, Mz_Ed):
    is_safe, util = column_surface.get_utilization_vector(N_Ed=N_Ed, My_Ed=My_Ed, Mz_Ed=Mz_Ed)
    assert util >= 0.0
    assert is_safe == (util <= 1.0)


def test_utilization_scales_with_load(column_surface):
    """Doubling the load roughly doubles the (radial) utilization."""
    _, util1 = column_surface.get_utilization_vector(N_Ed=1000, My_Ed=60, Mz_Ed=40)
    _, util2 = column_surface.get_utilization_vector(N_Ed=2000, My_Ed=120, Mz_Ed=80)
    assert util1 > 0
    assert util2 == pytest.approx(2.0 * util1, rel=0.10)


def test_utilization_is_monotonic_in_scale(column_surface):
    utils = [
        column_surface.get_utilization_vector(N_Ed=s * 1500, My_Ed=s * 100, Mz_Ed=s * 50)[1]
        for s in (0.5, 0.8, 1.0, 1.2)
    ]
    assert all(b > a for a, b in zip(utils, utils[1:])), f"Utilization should increase with scale: {utils}"
