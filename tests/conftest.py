"""
Pytest configuration and shared fixtures.
"""

import pytest
from materials.reinforced_concrete.materials import ConcreteMaterial, Rebar, ShearRebar
from materials.reinforced_concrete.geometry import (
    create_rectangular_section,
    create_linear_rebar_layer,
)


@pytest.fixture
def concrete_c30():
    """Standard C30/37 concrete."""
    return ConcreteMaterial(grade="C30/37")


@pytest.fixture
def concrete_c50():
    """High strength C50/60 concrete."""
    return ConcreteMaterial(grade="C50/60")


@pytest.fixture
def steel_b500b():
    """Standard B500B reinforcing steel."""
    return Rebar(diameter=16, grade="B500B")


@pytest.fixture
def rebar_16():
    """16mm diameter B500B rebar."""
    return Rebar(diameter=16, grade="B500B")


@pytest.fixture
def rebar_20():
    """20mm diameter B500B rebar."""
    return Rebar(diameter=20, grade="B500B")


@pytest.fixture
def shear_links():
    """Standard shear links."""
    return ShearRebar(
        diameter=10,
        grade="B500B",
        link_spacing=200,
        n_legs=2,
        angle=90.0,
    )


@pytest.fixture
def rectangular_beam():
    """300×500 mm rectangular beam section."""
    return create_rectangular_section(width=300, height=500, section_name="Test Beam")


@pytest.fixture
def rectangular_beam_with_rebars(rectangular_beam, rebar_20):
    """300×500 mm beam with bottom reinforcement."""
    bottom_layer = create_linear_rebar_layer(
        rebar=rebar_20,
        n_bars=3,
        start_point=(50, 50),
        end_point=(250, 50),
        layer_name="bottom",
    )
    rectangular_beam.add_rebar_group(bottom_layer)
    return rectangular_beam

