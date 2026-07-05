"""
Reinforcing steel stress-strain relationships according to Eurocode 2.

Implements EC2 Fig 3.8 with options for:
- Inclined top branch (with strain hardening)
- Horizontal top branch (elastic-perfectly plastic)
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt
from pydantic import Field, model_validator

from materials.core.constitutive import BaseConstitutiveModel
from materials.reinforced_concrete.materials.reinforcing_steel import ReinforcingSteel


SteelModelType = Literal["inclined", "horizontal"]


class SteelStressStrainEC2(BaseConstitutiveModel):
    """
    Steel stress-strain relationship per EC2 Fig 3.8.

    Bilinear model with optional strain hardening:
    - Elastic branch: σ = E_s · ε (up to ε_y)
    - Plastic branch:
        - horizontal: σ = ±f_y (no strain limit)
        - inclined:   σ increases linearly from f_y at ε_y to f_t at ε_ud

    Notes on ε_ud:
        EC2 provides a limit strain for ductility classification / model validity.
        In section analysis it is usually safer to CLIP strains to ε_ud rather than
        forcing stress to zero, which would be non-physical for reinforcement.
    """

    steel: ReinforcingSteel = Field(
        ...,
        description="Reinforcing steel material"
    )

    branch_type: SteelModelType = Field(
        default="inclined",
        description="Top branch type (inclined=strain hardening, horizontal=perfectly plastic)"
    )

    use_characteristic: bool = Field(
        default=False,
        description="Use f_yk instead of f_yd (for characteristic calculations)"
    )

    name: str = Field(
        default="EC2 Steel",
        description="Model name"
    )

    @property
    def f_y(self) -> float:
        """Yield strength (design or characteristic)."""
        return self.steel.f_yk if self.use_characteristic else self.steel.f_yd

    @property
    def epsilon_y(self) -> float:
        """Yield strain corresponding to f_y."""
        return self.f_y / self.steel.E_s

    @model_validator(mode="after")
    def validate_strain_limits(self) -> "SteelStressStrainEC2":
        """
        Ensure model parameters are consistent.

        For inclined branch we need epsilon_ud > epsilon_y to interpolate.
        """
        if self.branch_type == "inclined":
            if self.steel.epsilon_ud <= self.epsilon_y:
                raise ValueError(
                    "Invalid steel strain limits for inclined branch: "
                    f"epsilon_ud ({self.steel.epsilon_ud:g}) must be > epsilon_y ({self.epsilon_y:g})."
                )
        return self

    def get_stress(self, strain: float) -> float:
        """
        Calculate stress for given strain (tension positive).

        Handles tension (positive strain) and compression (negative strain).

        Args:
            strain: Strain (positive for tension, negative for compression)

        Returns:
            Stress in MPa (positive for tension, negative for compression)
        """
        abs_strain = abs(strain)
        sign = 1.0 if strain >= 0.0 else -1.0

        # Elastic region
        if abs_strain <= self.epsilon_y:
            return self.steel.E_s * strain

        # Plastic region
        if self.branch_type == "horizontal":
            # Horizontal branch: no strain limit per EC2 §3.2.7(2) option b
            return sign * self.f_y

        # Inclined branch: strain hardening up to epsilon_ud, then clip at epsilon_ud
        clipped = min(abs_strain, self.steel.epsilon_ud)
        strain_ratio = (clipped - self.epsilon_y) / (self.steel.epsilon_ud - self.epsilon_y)
        stress = self.f_y + (self.steel.f_t - self.f_y) * strain_ratio
        return sign * stress

    def get_stress_array(self, strains: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Vectorized stress calculation.

        Args:
            strains: array of strains

        Returns:
            array of stresses in MPa
        """
        strains = np.asarray(strains, dtype=float)
        stresses = np.zeros_like(strains)

        abs_strains = np.abs(strains)
        # Use sign like scalar path (treat 0 as +)
        signs = np.where(strains >= 0.0, 1.0, -1.0)

        # Elastic region
        elastic = abs_strains <= self.epsilon_y
        stresses[elastic] = self.steel.E_s * strains[elastic]

        # Plastic region
        plastic = ~elastic
        if not np.any(plastic):
            return stresses

        if self.branch_type == "horizontal":
            stresses[plastic] = signs[plastic] * self.f_y
            return stresses

        # Inclined branch
        clipped = np.minimum(abs_strains[plastic], self.steel.epsilon_ud)
        strain_ratio = (clipped - self.epsilon_y) / (self.steel.epsilon_ud - self.epsilon_y)
        stress_mag = self.f_y + (self.steel.f_t - self.f_y) * strain_ratio
        stresses[plastic] = signs[plastic] * stress_mag

        return stresses

    def get_ultimate_strain(self) -> float:
        """
        Return ultimate strain limit used by the model.

        For inclined branch: ε_ud (strain limit used for clipping)
        For horizontal branch: inf (no strain limit per EC2 §3.2.7(2) option b)
        """
        return float("inf") if self.branch_type == "horizontal" else float(self.steel.epsilon_ud)

    def get_yield_stress(self) -> float:
        """Return yield strength used by the model."""
        return float(self.f_y)

    def get_stress_tension_only(self, strain: float) -> float:
        """
        Calculate stress for tension only (ignores compression).

        Args:
            strain: Strain (positive for tension)

        Returns:
            Stress in MPa (0 if strain is negative)
        """
        if strain <= 0.0:
            return 0.0
        return self.get_stress(strain)

    def get_stress_compression_only(self, strain: float) -> float:
        """
        Calculate stress for compression only (ignores tension).

        Args:
            strain: Strain (negative for compression)

        Returns:
            Stress in MPa (0 if strain is positive)
        """
        if strain >= 0.0:
            return 0.0
        return self.get_stress(strain)


def create_steel_stress_strain(
    steel: ReinforcingSteel,
    branch_type: SteelModelType = "inclined",
    use_characteristic: bool = False,
) -> SteelStressStrainEC2:
    """
    Factory function to create steel stress-strain models.

    Args:
        steel: Reinforcing steel material
        branch_type: "inclined" for strain hardening, "horizontal" for perfectly plastic
        use_characteristic: Use f_yk instead of f_yd

    Returns:
        Steel stress-strain model
    """
    return SteelStressStrainEC2(
        steel=steel,
        branch_type=branch_type,
        use_characteristic=use_characteristic,
    )
