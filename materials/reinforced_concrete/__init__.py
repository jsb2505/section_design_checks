"""
Reinforced Concrete module - Eurocode 2 implementation.

Complete material models, stress-strain relationships, geometry handling,
and code checks for reinforced concrete design.
"""

# Materials
from materials.reinforced_concrete.materials import (
    ConcreteMaterial,
    ConcreteGrade,
    ReinforcingSteel,
    ReinforcingSteelGrade,
    Rebar,
    ShearRebar,
    BarDiameter,
)

# Constitutive models
from materials.reinforced_concrete.constitutive import (
    ConcreteStressStrainSchematic,
    ConcreteStressStrainParabolaRectangle,
    ConcreteStressStrainBilinear,
    SteelStressStrainEC2,
    create_concrete_stress_strain,
    create_steel_stress_strain,
)

# Geometry
from materials.reinforced_concrete.geometry import (
    RCSection,
    RebarGroup,
    create_rectangular_section,
    create_circular_section,
    create_linear_rebar_layer,
    create_rectangular_perimeter_rebars,
    create_circular_perimeter_rebars,
    create_custom_rebar_layer,
    create_single_rebar,
    FiberMesh,
)

__all__ = [
    # Materials
    "ConcreteMaterial",
    "ConcreteGrade",
    "ReinforcingSteel",
    "ReinforcingSteelGrade",
    "Rebar",
    "ShearRebar",
    "BarDiameter",
    # Constitutive
    "ConcreteStressStrainSchematic",
    "ConcreteStressStrainParabolaRectangle",
    "ConcreteStressStrainBilinear",
    "SteelStressStrainEC2",
    "create_concrete_stress_strain",
    "create_steel_stress_strain",
    # Geometry
    "RCSection",
    "RebarGroup",
    "create_rectangular_section",
    "create_circular_section",
    "create_linear_rebar_layer",
    "create_rectangular_perimeter_rebars",
    "create_circular_perimeter_rebars",
    "create_custom_rebar_layer",
    "create_single_rebar",
    "FiberMesh",
]
