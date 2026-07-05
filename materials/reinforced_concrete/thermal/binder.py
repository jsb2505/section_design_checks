"""Binder composition model for concrete thermal analysis."""

from typing import Optional
from pydantic import BaseModel, Field, ConfigDict, model_validator
from enum import StrEnum
from math import isclose


class BinderSubstituteType(StrEnum):
    '''
    Binder substitutes for portland cement in a concrete mix.

    Attributes:
        GGBS: Ground granulated blast-furnace slag
        PFA: Pulverised fuel ash (fly ash)
    '''
    GGBS = "ggbs"
    PFA = "pfa"


class Binder(BaseModel):
    """Defines cement binder composition with optional substitutes.

    Cement can be partially replaced with supplementary cementitious materials:
    - ggbs: Ground Granulated Blast-furnace Slag
    - pfa: Pulverized Fuel Ash (fly ash)

    These substitutes affect heat generation and strength development rates.

    Attributes:
        substitute_type: Type of cement substitute ('ggbs' or 'pfa'), None for pure cement
        substitute_percent: Percentage of cement replaced by substitute (0-100%)

    Behaviour:
        If substitute_type is changed to None, substitute_percent will be set to 0.

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
    model_config = ConfigDict(validate_assignment=True)

    substitute_type: Optional[BinderSubstituteType] = Field(
        default=None,
        description="Type of cement substitute: 'ggbs' or 'pfa', None for pure cement",
    )

    substitute_percent: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Percentage of cement replaced by substitute (0-100%)",
    )

    @model_validator(mode="after")
    def normalize_and_check(self) -> "Binder":
        # 1) Normalise: if no substitute, percent is always 0
        if self.substitute_type is None:
            if not isclose(self.substitute_percent, 0.0, abs_tol=1e-9):
                object.__setattr__(self, "substitute_percent", 0.0)
            return self

        # 2) If substitute type is set, percent must be > 0
        if self.substitute_percent <= 0.0:
            raise ValueError(
                f"substitute_percent must be > 0 when substitute_type='{self.substitute_type.value}'"
            )

        # 3) Type-specific upper limits
        max_by_type = {
            BinderSubstituteType.GGBS: 90.0,
            BinderSubstituteType.PFA: 70.0,
        }
        max_allowed = max_by_type[self.substitute_type]
        if self.substitute_percent > max_allowed:
            raise ValueError(
                f"substitute_percent must be <= {max_allowed:.0f} for substitute_type='{self.substitute_type.value}'"
            )

        return self

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
