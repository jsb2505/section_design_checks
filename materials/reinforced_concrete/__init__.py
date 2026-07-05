"""
Reinforced Concrete module - Eurocode 2 implementation.

Complete material models, stress-strain relationships, geometry handling,
and code checks for reinforced concrete design.
"""

# Core geometry
from materials.core.geometry import Point2D

# Materials
from materials.reinforced_concrete.materials import (
    ConcreteMaterial,
    ConcreteGrade,
    AggregateType,
    ConcreteAge,
    CementClass,
    ReinforcingSteel,
    ReinforcingSteelGrade,
    Rebar,
    ShearRebar,
    STANDARD_BAR_DIAMETERS,
)

# Constitutive models
from materials.reinforced_concrete.constitutive import (
    ConcreteStressStrainSchematic,
    ConcreteStressStrainParabolaRectangle,
    ConcreteStressStrainBilinear,
    ConcreteModelType,
    SteelStressStrainEC2,
    SteelModelType,
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
    FibreMesh,
)

# Analysis
from materials.reinforced_concrete.analysis import (
    InteractionPoint,
    MNInteractionDiagram,
    create_interaction_diagram,
    BiaxialInteractionPoint,
    BiaxialMNInteractionSurface,
    create_biaxial_interaction_surface,
)

# Code checks – EC2 2004
from materials.reinforced_concrete.code_checks.ec2_2004 import (
    BendingCheck,
    ShearCheck,
    LoadCase,
    CrackingCheck,
    LoadDuration,
)

__all__ = [
    # Core geometry
    "Point2D",
    # Materials
    "ConcreteMaterial",
    "ConcreteGrade",
    "AggregateType",
    "ConcreteAge",
    "CementClass",
    "ReinforcingSteel",
    "ReinforcingSteelGrade",
    "Rebar",
    "ShearRebar",
    "STANDARD_BAR_DIAMETERS",
    # Constitutive
    "ConcreteStressStrainSchematic",
    "ConcreteStressStrainParabolaRectangle",
    "ConcreteStressStrainBilinear",
    "ConcreteModelType",
    "SteelStressStrainEC2",
    "SteelModelType",
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
    "FibreMesh",
    # Analysis
    "InteractionPoint",
    "MNInteractionDiagram",
    "create_interaction_diagram",
    "BiaxialInteractionPoint",
    "BiaxialMNInteractionSurface",
    "create_biaxial_interaction_surface",
    # Code checks – EC2 2004
    "BendingCheck",
    "ShearCheck",
    "LoadCase",
    "CrackingCheck",
    "LoadDuration",
]
