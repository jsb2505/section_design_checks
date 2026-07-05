"""
Core abstractions for the materials library.

Provides base classes for materials, constitutive models, and geometry.
"""

from materials.core.base_material import BaseMaterial
from materials.core.constitutive import (
    BaseConstitutiveModel,
    StressStrainRelationship,
)
from materials.core.geometry import BaseGeometry, Point2D
from materials.core.units import (
    LengthUnit,
    StressUnit,
    ForceUnit,
    MomentUnit,
    LENGTH_TO_MM,
    STRESS_TO_MPA,
    FORCE_TO_KN,
)

__all__ = [
    "BaseMaterial",
    "BaseConstitutiveModel",
    "StressStrainRelationship",
    "BaseGeometry",
    "Point2D",
    "LengthUnit",
    "StressUnit",
    "ForceUnit",
    "MomentUnit",
    "LENGTH_TO_MM",
    "STRESS_TO_MPA",
    "FORCE_TO_KN",
]
