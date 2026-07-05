"""
Reinforcing steel stress-strain relationships according to Eurocode 2.

Implements EC2 Fig 3.8 with options for:
- Inclined top branch (with strain hardening)
- Horizontal top branch (elastic-perfectly plastic)
"""

from typing import Literal
import numpy as np
import numpy.typing as npt
from pydantic import Field, computed_field
from materials.core.constitutive import BaseConstitutiveModel
from materials.reinforced_concrete.materials.reinforcing_steel import ReinforcingSteel


class SteelStressStrainEC2(BaseConstitutiveModel):
    """
    Steel stress-strain relationship per EC2 Fig 3.8.

    Bilinear model with optional strain hardening:
    - Elastic branch: σ = E_s · ε (up to ε_yd)
    - Plastic branch: σ = f_yd (horizontal) or linearly increases to f_t (inclined)

    Formulation (inclined branch):
        σ_s = E_s · ε                           for |ε| ≤ ε_yd
        σ_s = f_yd + (f_t - f_yd)·(ε - ε_yd)/(ε_ud - ε_yd)  for ε_yd < |ε| ≤ ε_ud
        σ_s = 0                                  for |ε| > ε_ud

    Formulation (horizontal branch):
        σ_s = E_s · ε                           for |ε| ≤ ε_yd
        σ_s = f_yd                              for ε_yd < |ε| ≤ ε_ud
        σ_s = 0                                  for |ε| > ε_ud
    """

    steel: ReinforcingSteel = Field(
        ...,
        description="Reinforcing steel material"
    )

    branch_type: Literal["inclined", "horizontal"] = Field(
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

    @computed_field
    @property
    def f_y(self) -> float:
        """Yield strength (design or characteristic)."""
        return self.steel.f_yk if self.use_characteristic else self.steel.f_yd

    @computed_field
    @property
    def epsilon_y(self) -> float:
        """Yield strain."""
        return self.f_y / self.steel.E_s

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
        sign = 1.0 if strain >= 0 else -1.0

        # Beyond ultimate strain
        if abs_strain > self.steel.epsilon_ud:
            return 0.0

        # Elastic region
        if abs_strain <= self.epsilon_y:
            return self.steel.E_s * strain

        # Plastic region
        if self.branch_type == "horizontal":
            return sign * self.f_y
        else:  # inclined
            # Linear interpolation from f_y to f_t
            strain_ratio = (abs_strain - self.epsilon_y) / (self.steel.epsilon_ud - self.epsilon_y)
            stress = self.f_y + (self.steel.f_t - self.f_y) * strain_ratio
            return sign * stress

    def get_stress_array(self, strains: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Vectorized stress calculation."""
        stresses = np.zeros_like(strains)
        abs_strains = np.abs(strains)
        signs = np.sign(strains)

        # Elastic region
        elastic = abs_strains <= self.epsilon_y
        stresses[elastic] = self.steel.E_s * strains[elastic]

        # Plastic region (below ultimate)
        plastic = (abs_strains > self.epsilon_y) & (abs_strains <= self.steel.epsilon_ud)

        if self.branch_type == "horizontal":
            stresses[plastic] = signs[plastic] * self.f_y
        else:  # inclined
            strain_ratio = (abs_strains[plastic] - self.epsilon_y) / (self.steel.epsilon_ud - self.epsilon_y)
            stress_magnitude = self.f_y + (self.steel.f_t - self.f_y) * strain_ratio
            stresses[plastic] = signs[plastic] * stress_magnitude

        # Beyond ultimate: already zero

        return stresses

    def get_ultimate_strain(self) -> float:
        """Return ultimate strain."""
        return self.steel.epsilon_ud

    def get_yield_stress(self) -> float:
        """Return yield strength."""
        return self.f_y

    def get_stress_tension_only(self, strain: float) -> float:
        """
        Calculate stress for tension only (ignores compression).

        Useful for tension reinforcement calculations where compression
        contribution is negligible.

        Args:
            strain: Strain (positive for tension)

        Returns:
            Stress in MPa (0 if strain is negative)
        """
        if strain <= 0:
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
        if strain >= 0:
            return 0.0
        return self.get_stress(strain)


SteelModelType = Literal["inclined", "horizontal"]


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

    Example:
        >>> steel = ReinforcingSteel(grade="B500B")
        >>> model = create_steel_stress_strain(steel, "inclined")
        >>> stress = model.get_stress(0.01)  # 1% strain
    """
    return SteelStressStrainEC2(
        steel=steel,
        branch_type=branch_type,
        use_characteristic=use_characteristic,
    )
