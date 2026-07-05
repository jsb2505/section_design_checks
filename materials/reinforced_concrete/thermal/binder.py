"""Binder composition model for concrete thermal analysis."""

from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator


class Binder(BaseModel):
    """Defines cement binder composition with optional substitutes.

    Cement can be partially replaced with supplementary cementitious materials:
    - ggbs: Ground Granulated Blast-furnace Slag
    - pfa: Pulverized Fuel Ash (fly ash)

    These substitutes affect heat generation and strength development rates.

    Attributes:
        substitute_type: Type of cement substitute ('ggbs' or 'pfa'), None for pure cement
        substitute_percent: Percentage of cement replaced by substitute (0-100%)

    Examples:
        >>> # Pure Portland cement
        >>> pure_cement = Binder()
        >>>
        >>> # 30% GGBS replacement
        >>> ggbs_blend = Binder(substitute_type="ggbs", substitute_percent=30)
        >>>
        >>> # 20% PFA replacement
        >>> pfa_blend = Binder(substitute_type="pfa", substitute_percent=20)
    """

    substitute_type: Optional[Literal["ggbs", "pfa"]] = Field(
        default=None,
        description="Type of cement substitute: 'ggbs' or 'pfa', None for pure cement",
    )

    substitute_percent: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Percentage of cement replaced by substitute (0-100%)",
    )

    @field_validator("substitute_percent")
    @classmethod
    def validate_substitute_percent(cls, v: float, info) -> float:
        """Ensure substitute_percent is 0 when no substitute type is specified."""
        substitute_type = info.data.get("substitute_type")

        if substitute_type is None and v != 0.0:
            raise ValueError(
                "substitute_percent must be 0 when substitute_type is None"
            )

        if substitute_type is not None and v == 0.0:
            raise ValueError(
                f"substitute_percent must be > 0 when substitute_type='{substitute_type}'"
            )

        return v

    @property
    def cement_percent(self) -> float:
        """Percentage of pure Portland cement in the binder."""
        return 100.0 - self.substitute_percent

    @property
    def is_pure_cement(self) -> bool:
        """True if binder contains no substitutes."""
        return self.substitute_type is None

    def __repr__(self) -> str:
        if self.is_pure_cement:
            return "Binder(100% cement)"
        return (
            f"Binder({self.cement_percent:.1f}% cement, "
            f"{self.substitute_percent:.1f}% {self.substitute_type})"
        )
