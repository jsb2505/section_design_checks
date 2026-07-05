"""
Reinforced Concrete module - Eurocode 2 implementation.

Complete material models, stress-strain relationships, geometry handling,
and code checks for reinforced concrete design.
"""

# Core geometry
from section_design_checks.core.geometry import Point2D

# Analysis
from section_design_checks.reinforced_concrete.analysis import (
    BiaxialInteractionPoint,
    BiaxialMNInteractionSurface,
    InteractionPoint,
    MNInteractionDiagram,
    create_biaxial_interaction_surface,
    create_interaction_diagram,
)

# Code checks – EC2 2004
from section_design_checks.reinforced_concrete.code_checks.ec2_2004 import (
    BendingCheck,
    CrackingCheck,
    LoadCase,
    LoadDuration,
    ShearCheck,
)

# Constitutive models
from section_design_checks.reinforced_concrete.constitutive import (
    ConcreteModelType,
    ConcreteStressStrainBilinear,
    ConcreteStressStrainParabolaRectangle,
    ConcreteStressStrainSchematic,
    SteelModelType,
    SteelStressStrainEC2,
    create_concrete_stress_strain,
    create_steel_stress_strain,
)

# Geometry
from section_design_checks.reinforced_concrete.geometry import (
    FibreMesh,
    RCSection,
    RebarGroup,
    create_circular_perimeter_rebars,
    create_circular_section,
    create_custom_rebar_layer,
    create_linear_rebar_layer,
    create_rectangular_perimeter_rebars,
    create_rectangular_section,
)

# Materials
from section_design_checks.reinforced_concrete.materials import (
    STANDARD_BAR_DIAMETERS,
    AggregateType,
    CementClass,
    ConcreteAge,
    ConcreteGrade,
    ConcreteMaterial,
    Rebar,
    ReinforcingSteel,
    ReinforcingSteelGrade,
    ShearRebar,
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
