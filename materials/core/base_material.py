"""
Base material class for all structural materials.

Provides common interface and validation for material properties.
"""

from abc import ABC, abstractmethod

from pydantic import BaseModel, ConfigDict, Field


class BaseMaterial(BaseModel, ABC):
    """
    Abstract base class for all structural materials.

    Uses Pydantic for validation and serialization.
    All material properties use standard units (see units.py).
    """

    model_config = ConfigDict(
        validate_assignment=True,
        arbitrary_types_allowed=False,
        extra="forbid",
        frozen=False,
    )

    name: str = Field(
        ...,
        description="Material name or identifier",
        min_length=1,
    )

    density: float = Field(
        ...,
        description="Material density in kg/m³",
        gt=0,
    )

    @abstractmethod
    def get_elastic_modulus(self) -> float:
        """
        Return the elastic modulus in MPa.

        Returns:
            Elastic modulus (MPa)
        """
        pass  # pragma: no cover - abstract interface placeholder

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"{self.__class__.__name__}, "
            f"(name='{self.name}, density='{self.density}')"
        )

    def __str__(self) -> str:
        """User-friendly string representation."""
        return f"{self.name}"
