"""
Custom user-defined stress-strain relationships.

Allows users to define arbitrary stress-strain curves via callables
for use with MNInteractionDiagram and ULS check classes.

Sign convention follows EC2:
- Strain > 0 => compression
- Stress > 0 => compression
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt
from pydantic import ConfigDict, Field, model_validator

from section_design_checks.core.constitutive import BaseConstitutiveModel


class CustomConcreteModel(BaseConstitutiveModel):
    """
    User-defined concrete stress-strain relationship.

    The user supplies a callable ``stress_fn(strain) -> stress`` that
    defines the curve.  Optional vectorized and tangent-modulus overrides
    are available for performance.

    Example::

        model = CustomConcreteModel(
            stress_fn=lambda eps: 30.0 * eps / 0.002 if eps < 0.002 else 30.0,
            ultimate_strain=0.0035,
            yield_stress=30.0,
        )
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    name: str = Field(default="Custom Concrete")

    stress_fn: Callable[[float], float] = Field(
        ...,
        description="Callable mapping strain (float) to stress (float, MPa).",
    )
    ultimate_strain: float = Field(
        ..., gt=0, description="Ultimate strain (dimensionless)."
    )
    yield_stress: float = Field(
        ..., gt=0, description="Yield / characteristic stress (MPa)."
    )
    cache_key: str = Field(
        default="",
        description="User-supplied key for snapshot-based cache invalidation.",
    )

    stress_array_fn: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]] | None = Field(
        default=None,
        description="Optional vectorized stress function for performance.",
    )
    tangent_modulus_fn: Callable[[float], float] | None = Field(
        default=None,
        description="Optional analytical tangent modulus function for performance.",
    )

    # ---- BaseConstitutiveModel interface ----

    def get_stress(self, strain: float) -> float:
        return self.stress_fn(strain)

    def get_ultimate_strain(self) -> float:
        return self.ultimate_strain

    def get_yield_stress(self) -> float:
        return self.yield_stress

    def get_stress_array(
        self, strains: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        if self.stress_array_fn is not None:
            return self.stress_array_fn(strains)
        return super().get_stress_array(strains)

    def get_tangent_modulus(self, strain: float) -> float:
        if self.tangent_modulus_fn is not None:
            return self.tangent_modulus_fn(strain)
        return super().get_tangent_modulus(strain)


class CustomSteelModel(BaseConstitutiveModel):
    """
    User-defined steel stress-strain relationship.

    Same callable pattern as :class:`CustomConcreteModel`, with an
    additional ``epsilon_y`` field required by the interaction-diagram
    solver.  If not supplied it defaults to ``yield_stress / 200_000``
    (E_s = 200 GPa).

    Example::

        model = CustomSteelModel(
            stress_fn=lambda eps: min(abs(eps) * 200_000, 500.0) * np.sign(eps),
            ultimate_strain=0.05,
            yield_stress=500.0,
        )
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    name: str = Field(default="Custom Steel")

    stress_fn: Callable[[float], float] = Field(
        ...,
        description="Callable mapping strain (float) to stress (float, MPa).",
    )
    ultimate_strain: float = Field(
        ..., gt=0, description="Ultimate strain (dimensionless)."
    )
    yield_stress: float = Field(
        ..., gt=0, description="Yield / characteristic stress (MPa)."
    )
    epsilon_y: float | None = Field(
        default=None,
        description=(
            "Yield strain. Defaults to yield_stress / 200_000 if not provided."
        ),
    )
    cache_key: str = Field(
        default="",
        description="User-supplied key for snapshot-based cache invalidation.",
    )

    stress_array_fn: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]] | None = Field(
        default=None,
        description="Optional vectorized stress function for performance.",
    )
    tangent_modulus_fn: Callable[[float], float] | None = Field(
        default=None,
        description="Optional analytical tangent modulus function for performance.",
    )

    @model_validator(mode="after")
    def _set_default_epsilon_y(self) -> CustomSteelModel:
        if self.epsilon_y is None:
            self.epsilon_y = self.yield_stress / 200_000.0
        return self

    # ---- BaseConstitutiveModel interface ----

    def get_stress(self, strain: float) -> float:
        return self.stress_fn(strain)

    def get_ultimate_strain(self) -> float:
        return self.ultimate_strain

    def get_yield_stress(self) -> float:
        return self.yield_stress

    def get_stress_array(
        self, strains: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        if self.stress_array_fn is not None:
            return self.stress_array_fn(strains)
        return super().get_stress_array(strains)

    def get_tangent_modulus(self, strain: float) -> float:
        if self.tangent_modulus_fn is not None:
            return self.tangent_modulus_fn(strain)
        return super().get_tangent_modulus(strain)
