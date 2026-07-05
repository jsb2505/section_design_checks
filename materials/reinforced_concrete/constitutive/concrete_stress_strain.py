"""
Concrete stress-strain relationships according to Eurocode 2.

Implements three EC2 models:
1. Schematic (Fig 3.2) - for analysis
2. Parabola-Rectangle (Fig 3.3) - for design
3. Bilinear (Fig 3.4) - simplified design
"""

from typing import Literal
import numpy as np
import numpy.typing as npt
from pydantic import Field, computed_field
from materials.core.constitutive import BaseConstitutiveModel
from materials.reinforced_concrete.materials.concrete import ConcreteMaterial


class ConcreteStressStrainSchematic(BaseConstitutiveModel):
    """
    Schematic stress-strain diagram for concrete (EC2 Fig 3.2).

    Used for structural analysis and strain calculations.
    Uses mean strength f_cm and modulus E_cm.

    Formulation:
        σ_c = f_cm · [k·η - η²] / [1 + (k-2)·η]
        where:
            k = 1.05 · E_cm · |ε_c1| / f_cm
            η = ε / |ε_c1|
    """

    concrete: ConcreteMaterial = Field(
        ...,
        description="Concrete material"
    )

    name: str = Field(
        default="EC2 Schematic",
        description="Model name"
    )

    @computed_field
    @property
    def k(self) -> float:
        """
        Calculate k parameter.

        k = 1.05 · E_cm · |ε_c1| / f_cm

        Returns:
            k parameter
        """
        return 1.05 * self.concrete.E_cm * abs(self.concrete.epsilon_c1) / self.concrete.f_cm

    def get_stress(self, strain: float) -> float:
        """
        Calculate stress for given strain (compression positive).

        Args:
            strain: Compressive strain (positive for compression)

        Returns:
            Compressive stress in MPa (positive for compression)
        """
        # No tension capacity
        if strain <= 0:
            return 0.0

        # Beyond ultimate strain
        if strain > self.concrete.epsilon_cu1:
            return 0.0

        # Calculate eta
        eta = strain / abs(self.concrete.epsilon_c1)

        # Calculate stress
        numerator = self.k * eta - eta ** 2
        denominator = 1 + (self.k - 2) * eta

        if denominator == 0:
            return 0.0

        stress = self.concrete.f_cm * numerator / denominator

        return max(0.0, stress)

    def get_stress_array(self, strains: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Vectorized stress calculation."""
        # No tension
        stresses = np.zeros_like(strains)

        # Valid compression range
        valid = (strains > 0) & (strains <= self.concrete.epsilon_cu1)

        eta = strains[valid] / abs(self.concrete.epsilon_c1)
        numerator = self.k * eta - eta ** 2
        denominator = 1 + (self.k - 2) * eta

        # Avoid division by zero
        mask = denominator != 0
        stresses[valid][mask] = self.concrete.f_cm * numerator[mask] / denominator[mask]

        return np.maximum(0.0, stresses)

    def get_ultimate_strain(self) -> float:
        """Return ultimate strain."""
        return self.concrete.epsilon_cu1

    def get_yield_stress(self) -> float:
        """Return peak stress (mean strength)."""
        return self.concrete.f_cm


class ConcreteStressStrainParabolaRectangle(BaseConstitutiveModel):
    """
    Parabola-rectangle stress-strain diagram (EC2 Fig 3.3).

    Standard design diagram for ULS bending/compression.

    Formulation:
        σ_c = f_cd · [1 - (1 - ε/ε_c2)^n]  for 0 ≤ ε ≤ ε_c2
        σ_c = f_cd                          for ε_c2 < ε ≤ ε_cu2
        σ_c = 0                             for ε > ε_cu2
    """

    concrete: ConcreteMaterial = Field(
        ...,
        description="Concrete material"
    )

    use_characteristic: bool = Field(
        default=False,
        description="Use f_ck instead of f_cd (for characteristic calculations)"
    )

    name: str = Field(
        default="EC2 Parabola-Rectangle",
        description="Model name"
    )

    @computed_field
    @property
    def f_c(self) -> float:
        """Design or characteristic strength depending on use_characteristic flag."""
        return self.concrete.f_ck if self.use_characteristic else self.concrete.f_cd

    def get_stress(self, strain: float) -> float:
        """
        Calculate stress for given strain (compression positive).

        Args:
            strain: Compressive strain (positive for compression)

        Returns:
            Compressive stress in MPa (positive for compression)
        """
        # No tension
        if strain <= 0:
            return 0.0

        # Beyond ultimate
        if strain > self.concrete.epsilon_cu2:
            return 0.0

        # Rectangular portion
        if strain >= self.concrete.epsilon_c2:
            return self.f_c

        # Parabolic portion
        ratio = 1.0 - strain / self.concrete.epsilon_c2
        stress = self.f_c * (1.0 - ratio ** self.concrete.n)

        return stress

    def get_stress_array(self, strains: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Vectorized stress calculation."""
        stresses = np.zeros_like(strains)

        # Parabolic region
        parabolic = (strains > 0) & (strains <= self.concrete.epsilon_c2)
        ratio = 1.0 - strains[parabolic] / self.concrete.epsilon_c2
        stresses[parabolic] = self.f_c * (1.0 - ratio ** self.concrete.n)

        # Rectangular region
        rectangular = (strains > self.concrete.epsilon_c2) & (strains <= self.concrete.epsilon_cu2)
        stresses[rectangular] = self.f_c

        return stresses

    def get_ultimate_strain(self) -> float:
        """Return ultimate strain."""
        return self.concrete.epsilon_cu2

    def get_yield_stress(self) -> float:
        """Return design/characteristic strength."""
        return self.f_c


class ConcreteStressStrainBilinear(BaseConstitutiveModel):
    """
    Bilinear stress-strain diagram (EC2 Fig 3.4).

    Simplified design diagram.

    Formulation:
        σ_c = f_cd · ε/ε_c3       for 0 ≤ ε ≤ ε_c3
        σ_c = f_cd                for ε_c3 < ε ≤ ε_cu3
        σ_c = 0                   for ε > ε_cu3
    """

    concrete: ConcreteMaterial = Field(
        ...,
        description="Concrete material"
    )

    use_characteristic: bool = Field(
        default=False,
        description="Use f_ck instead of f_cd"
    )

    name: str = Field(
        default="EC2 Bilinear",
        description="Model name"
    )

    @computed_field
    @property
    def f_c(self) -> float:
        """Design or characteristic strength."""
        return self.concrete.f_ck if self.use_characteristic else self.concrete.f_cd

    def get_stress(self, strain: float) -> float:
        """
        Calculate stress for given strain (compression positive).

        Args:
            strain: Compressive strain (positive for compression)

        Returns:
            Compressive stress in MPa (positive for compression)
        """
        # No tension
        if strain <= 0:
            return 0.0

        # Beyond ultimate
        if strain > self.concrete.epsilon_cu3:
            return 0.0

        # Constant stress portion
        if strain >= self.concrete.epsilon_c3:
            return self.f_c

        # Linear portion
        stress = self.f_c * strain / self.concrete.epsilon_c3

        return stress

    def get_stress_array(self, strains: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Vectorized stress calculation."""
        stresses = np.zeros_like(strains)

        # Linear region
        linear = (strains > 0) & (strains <= self.concrete.epsilon_c3)
        stresses[linear] = self.f_c * strains[linear] / self.concrete.epsilon_c3

        # Constant region
        constant = (strains > self.concrete.epsilon_c3) & (strains <= self.concrete.epsilon_cu3)
        stresses[constant] = self.f_c

        return stresses

    def get_ultimate_strain(self) -> float:
        """Return ultimate strain."""
        return self.concrete.epsilon_cu3

    def get_yield_stress(self) -> float:
        """Return design/characteristic strength."""
        return self.f_c


ConcreteModelType = Literal["schematic", "parabola-rectangle", "bilinear"]


def create_concrete_stress_strain(
    concrete: ConcreteMaterial,
    model_type: ConcreteModelType = "parabola-rectangle",
    use_characteristic: bool = False,
) -> BaseConstitutiveModel:
    """
    Factory function to create concrete stress-strain models.

    Args:
        concrete: Concrete material
        model_type: Type of model to create
        use_characteristic: Use f_ck instead of f_cd (ignored for schematic)

    Returns:
        Concrete stress-strain model

    Example:
        >>> concrete = ConcreteMaterial(grade="C30/37")
        >>> model = create_concrete_stress_strain(concrete, "parabola-rectangle")
        >>> stress = model.get_stress(0.002)  # At ε_c2
    """
    if model_type == "schematic":
        return ConcreteStressStrainSchematic(concrete=concrete)
    elif model_type == "parabola-rectangle":
        return ConcreteStressStrainParabolaRectangle(
            concrete=concrete,
            use_characteristic=use_characteristic
        )
    elif model_type == "bilinear":
        return ConcreteStressStrainBilinear(
            concrete=concrete,
            use_characteristic=use_characteristic
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
