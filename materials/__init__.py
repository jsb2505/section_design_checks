"""
Materials - Structural Engineering Library

API-ready library for structural materials and code-checked section design.
"""

__version__ = "0.1.0"

# Core exports
from materials.core import (
    BaseMaterial,
    BaseConstitutiveModel,
    BaseGeometry,
    Point2D,
)

__all__ = [
    "BaseMaterial",
    "BaseConstitutiveModel",
    "BaseGeometry",
    "Point2D",
]
