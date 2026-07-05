"""
Base classes for material constitutive (stress-strain) relationships.
"""

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field


@runtime_checkable
class StressStrainRelationship(Protocol):
    """Protocol for stress-strain relationships."""

    def get_stress(self, strain: float) -> float:
        """
        Calculate stress for a given strain.

        Args:
            strain: Strain value (dimensionless)

        Returns:
            Stress in MPa
        """
        ...

    def get_stress_array(self, strains: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Calculate stress for an array of strains (vectorized).

        Args:
            strains: Array of strain values

        Returns:
            Array of stresses in MPa
        """
        ...


class BaseConstitutiveModel(BaseModel, ABC):
    """
    Base class for constitutive (stress-strain) models.

    Provides common interface for all stress-strain relationships.
    """

    model_config = ConfigDict(
        validate_assignment=True,
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    name: str = Field(
        ...,
        description="Model name/identifier",
        min_length=1,
    )

    @abstractmethod
    def get_stress(self, strain: float) -> float:
        """
        Calculate stress for a given strain.

        Args:
            strain: Strain value (dimensionless)

        Returns:
            Stress in MPa
        """
        pass  # pragma: no cover - abstract interface placeholder

    def get_stress_array(self, strains: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Calculate stress for an array of strains.

        Default implementation vectorizes get_stress().
        Override for performance if analytical vectorization is possible.

        Args:
            strains: Array of strain values

        Returns:
            Array of stresses in MPa
        """
        return np.asarray(np.vectorize(self.get_stress)(strains), dtype=np.float64)

    @abstractmethod
    def get_ultimate_strain(self) -> float:
        """
        Return the ultimate strain for this material.

        Returns:
            Ultimate strain (dimensionless)
        """
        pass  # pragma: no cover - abstract interface placeholder

    @abstractmethod
    def get_yield_stress(self) -> float:
        """
        Return the yield/characteristic stress.

        Returns:
            Yield stress in MPa
        """
        pass  # pragma: no cover - abstract interface placeholder

    def get_tangent_modulus(self, strain: float) -> float:
        """
        Calculate tangent modulus E_t = dσ/dε at given strain.

        Default implementation uses numerical differentiation.
        Override for analytical tangent modulus for better performance.

        Args:
            strain: Strain value (dimensionless)

        Returns:
            Tangent modulus in MPa (dσ/dε)
        """
        # Numerical differentiation (2-point central difference)
        h = 1e-8
        stress_plus = self.get_stress(strain + h)
        stress_minus = self.get_stress(strain - h)
        return (stress_plus - stress_minus) / (2 * h)

    def get_tangent_modulus_array(self, strains: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Calculate tangent modulus for an array of strains.

        Default implementation vectorizes get_tangent_modulus().
        Override for analytical vectorization for better performance.

        Args:
            strains: Array of strain values

        Returns:
            Array of tangent moduli in MPa
        """
        return np.asarray(np.vectorize(self.get_tangent_modulus)(strains), dtype=np.float64)

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(name='{self.name}')"
