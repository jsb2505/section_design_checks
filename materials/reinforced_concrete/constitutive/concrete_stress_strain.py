"""
Concrete stress-strain relationships according to Eurocode 2.

Implements three EC2 models:
1. Schematic (Fig 3.2) - for analysis
2. Parabola-Rectangle (Fig 3.3) - for design
3. Bilinear (Fig 3.4) - simplified design

Sign convention:
- Strain > 0 => compression
- Stress > 0 => compression
- No tensile capacity is modelled (stress = 0 for strain <= 0)

Ultimate strain handling:
- Concrete models are treated as domain-limited to ε_cu* (ULS).
- To avoid numerical instability from tiny overshoots, strains are tolerance-clipped:
    if ε > ε_cu + tol: stress = 0
    if ε_cu < ε <= ε_cu + tol: evaluate at ε_cu
This avoids discontinuities while not creating a post-crushing plateau.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Optional

import numpy as np
import numpy.typing as npt
from pydantic import Field, model_validator

from materials.core.constitutive import BaseConstitutiveModel
from materials.reinforced_concrete.materials.concrete import ConcreteMaterial


class ConcreteModelType(StrEnum):
    '''
    Concrete stress-strain relationships types
    as per Figure 3.2, 3.3 and 3.4  EC2.

    Attributes:
        SCHEMATIC: for structural analysis
        PARABOLA_RECTANGLE: for design of cross-sections
        BILINEAR: for (simplified) design of cross-sections
        LINEAR_ELASTIC: for SLS analysis
    '''
    SCHEMATIC = "schematic"
    PARABOLA_RECTANGLE = "parabola-rectangle"
    BILINEAR = "bilinear"
    LINEAR_ELASTIC = "linear-elastic"


def _apply_ultimate_tolerance_clip(
    strains: npt.NDArray[np.float64],
    epsilon_cu: float,
    tol: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    """
    Apply tolerance-based clipping at ultimate strain.

    For complex-step differentiation: Clipping creates discontinuities that break
    the complex-step method. When complex input is detected, we skip clipping entirely
    and let the constitutive model handle strains beyond limits naturally.

    Returns:
        strains_clipped: copy of strains with values in (ε_cu, ε_cu+tol] clipped to ε_cu
        killed: boolean mask where strain > ε_cu + tol (these should yield stress=0)
    """
    # Accept complex input for complex-step differentiation
    strains = np.asarray(strains)

    # For complex-step: skip clipping to preserve derivatives
    # The solver won't actually reach these extreme strains, so this is safe
    if np.iscomplexobj(strains):
        killed = np.zeros(strains.shape, dtype=bool)
        return strains, killed

    # Use real part for comparisons
    strains_real = strains

    if tol <= 0.0:
        # No tolerance behaviour requested: no clipping, hard cutoff handled by caller
        killed = strains_real > epsilon_cu
        return strains, killed

    killed = strains_real > (epsilon_cu + tol)
    strains_clipped = strains.copy()

    near = (strains_real > epsilon_cu) & (~killed)
    if np.any(near):
        strains_clipped[near] = epsilon_cu

    return strains_clipped, killed


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

    concrete: ConcreteMaterial = Field(..., description="Concrete material")
    name: str = Field(default="EC2 Schematic", description="Model name")

    # Tolerance to avoid numerical discontinuity at ε_cu1
    ultimate_strain_tol: float = Field(
        default=1e-12,
        ge=0.0,
        description="Tolerance for ultimate strain clipping (dimensionless strain).",
    )


    @property
    def k(self) -> float:
        """k = 1.05 · E_cm · |ε_c1| / f_cm"""
        return 1.05 * self.concrete.E_cm * abs(self.concrete.epsilon_c1) / self.concrete.f_cm


    @model_validator(mode="after")
    def validate_parameters(self) -> "ConcreteStressStrainSchematic":
        if self.concrete.f_cm <= 0:
            raise ValueError(f"Concrete f_cm must be > 0, got {self.concrete.f_cm}")
        if abs(self.concrete.epsilon_c1) <= 0:
            raise ValueError(f"Concrete epsilon_c1 must be non-zero, got {self.concrete.epsilon_c1}")
        if self.concrete.epsilon_cu1 <= 0:
            raise ValueError(f"Concrete epsilon_cu1 must be > 0, got {self.concrete.epsilon_cu1}")
        return self


    def get_stress(self, strain: float) -> float:
        """
        Calculate stress for given strain (compression positive).

        Args:
            strain: Compressive strain (positive for compression)

        Returns:
            Compressive stress in MPa (positive for compression)
        """
        if strain <= 0.0:
            return 0.0

        eps_cu = float(self.concrete.epsilon_cu1)
        tol = float(self.ultimate_strain_tol)

        if strain > eps_cu + tol:
            return 0.0
        if strain > eps_cu:
            strain = eps_cu

        eta = strain / abs(self.concrete.epsilon_c1)

        numerator = self.k * eta - eta * eta
        denominator = 1.0 + (self.k - 2.0) * eta

        # Avoid blow-up near singularities
        if abs(denominator) < 1e-12:
            return 0.0

        stress = self.concrete.f_cm * numerator / denominator
        return max(0.0, float(stress))


    def get_stress_array(self, strains: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Vectorized stress calculation for ULS (cracked concrete assumption).

        Sign convention: compression positive.

        ULS assumption (cracked section):
            - Tensile concrete (strain ≤ 0) contributes zero stress
            - Only compressive concrete (strain > 0) carries stress
            - This is standard for Ultimate Limit State design per EC2 §6.1
        """
        strains = np.asarray(strains, dtype=float)
        stresses = np.zeros_like(strains)

        # ULS cracked section: concrete in tension (strain ≤ 0) has zero stress.
        # Only fibres with positive strain (compression) contribute.
        comp = strains > 0.0
        if not np.any(comp):
            return stresses

        # Clip strains exceeding ultimate strain (with small tolerance)
        eps_cu = float(self.concrete.epsilon_cu1)
        strains_clipped, killed = _apply_ultimate_tolerance_clip(
            strains=strains,
            epsilon_cu=eps_cu,
            tol=float(self.ultimate_strain_tol),
        )

        # Valid fibres: in compression AND not killed by strain limit
        valid = comp & (~killed)
        if not np.any(valid):
            return stresses

        # Sargin formula: σ = f_cm * (k·η - η²) / (1 + (k-2)·η)
        # where η = ε / ε_c1 (normalised strain)
        eps1 = abs(self.concrete.epsilon_c1)
        eta = strains_clipped[valid] / eps1

        numerator = self.k * eta - eta**2
        denominator = 1.0 + (self.k - 2.0) * eta

        # Avoid division by zero (denominator ≈ 0 is rare but possible)
        mask = np.abs(denominator) >= 1e-12
        if np.any(mask):
            valid_idx = np.flatnonzero(valid)
            write_idx = valid_idx[mask]
            stresses[write_idx] = self.concrete.f_cm * (numerator[mask] / denominator[mask])

        # Final clamp: ensure no tensile stress (numerical safety)
        return np.maximum(0.0, stresses)


    def get_ultimate_strain(self) -> float:
        """Return ultimate strain."""
        return float(self.concrete.epsilon_cu1)


    def get_yield_stress(self) -> float:
        """Return peak stress (mean strength)."""
        return float(self.concrete.f_cm)


class ConcreteStressStrainParabolaRectangle(BaseConstitutiveModel):
    """
    Parabola-rectangle stress-strain diagram (EC2 Fig 3.3).

    Standard design diagram for ULS bending/compression.

    Formulation:
        σ_c = f_c · [1 - (1 - ε/ε_c2)^n]  for 0 ≤ ε ≤ ε_c2
        σ_c = f_c                         for ε_c2 < ε ≤ ε_cu2
        σ_c = 0                           for ε > ε_cu2

    The strength f_c can be (mutually exclusive):
        - f_cd (design strength) when use_characteristic=False and use_accidental=False (default)
        - f_ck (characteristic strength) when use_characteristic=True
        - f_cd_accidental (accidental design strength) when use_accidental=True

    Note: use_characteristic and use_accidental cannot both be True.

    EC2 §3.1.9 Confinement:
        When sigma_2 > 0 is provided, confined concrete properties are used per EC2 §3.1.9:
        - fck,c = fck(1.000 + 5.0·σ₂/fck)  for σ₂ ≤ 0.05·fck  (Eq. 3.24)
        - fck,c = fck(1.125 + 2.5·σ₂/fck)  for σ₂ > 0.05·fck  (Eq. 3.25)
        - εc2,c = εc2·(fck,c/fck)²         (Eq. 3.26)
        - εcu2,c = εcu2 + 0.2·σ₂/fck       (Eq. 3.27)

        where σ₂ is the effective lateral compressive stress at ULS due to confinement.
    """

    concrete: ConcreteMaterial = Field(..., description="Concrete material")

    use_characteristic: bool = Field(
        default=False,
        description="Use f_ck instead of f_cd (mutually exclusive with use_accidental)"
    )

    use_accidental: bool = Field(
        default=False,
        description="Use f_cd_accidental instead of f_cd (mutually exclusive with use_characteristic)"
    )

    sigma_2: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="EC2 §3.1.9: Effective lateral compressive stress (MPa) for confinement. "
                    "When provided, confined concrete properties are used."
    )

    name: str = Field(default="EC2 Parabola-Rectangle", description="Model name")

    ultimate_strain_tol: float = Field(
        default=1e-12,
        ge=0.0,
        description="Tolerance for ultimate strain clipping (dimensionless strain).",
    )

    @property
    def is_ec2_confined(self) -> bool:
        """True if EC2 §3.1.9 confinement is active (sigma_2 > 0)."""
        return self.sigma_2 is not None and self.sigma_2 > 0.0

    @property
    def f_ck_c(self) -> float:
        """
        Confined characteristic strength per EC2 §3.1.9.

        fck,c = fck(1.000 + 5.0·σ₂/fck)  for σ₂ ≤ 0.05·fck
        fck,c = fck(1.125 + 2.5·σ₂/fck)  for σ₂ > 0.05·fck

        Returns unconfined f_ck if sigma_2 is None or zero.
        """
        f_ck = float(self.concrete.f_ck)
        if not self.is_ec2_confined:
            return f_ck

        sigma_2 = float(self.sigma_2)  # type: ignore[arg-type]
        ratio = sigma_2 / f_ck

        if sigma_2 <= 0.05 * f_ck:
            return f_ck * (1.000 + 5.0 * ratio)
        else:
            return f_ck * (1.125 + 2.5 * ratio)

    @property
    def epsilon_c2_c(self) -> float:
        """
        Confined strain at peak stress per EC2 §3.1.9 Eq. 3.26.

        εc2,c = εc2·(fck,c/fck)²

        Returns unconfined ε_c2 if sigma_2 is None or zero.
        """
        eps_c2 = float(self.concrete.epsilon_c2)
        if not self.is_ec2_confined:
            return eps_c2

        f_ck = float(self.concrete.f_ck)
        strength_ratio = self.f_ck_c / f_ck
        return eps_c2 * (strength_ratio ** 2)

    @property
    def epsilon_cu2_c(self) -> float:
        """
        Confined ultimate strain per EC2 §3.1.9 Eq. 3.27.

        εcu2,c = εcu2 + 0.2·σ₂/fck

        Returns unconfined ε_cu2 if sigma_2 is None or zero.
        """
        eps_cu2 = float(self.concrete.epsilon_cu2)
        if not self.is_ec2_confined:
            return eps_cu2

        f_ck = float(self.concrete.f_ck)
        sigma_2 = float(self.sigma_2)  # type: ignore[arg-type]
        return eps_cu2 + 0.2 * sigma_2 / f_ck

    @property
    def f_c(self) -> float:
        """
        Design, characteristic, or accidental strength depending on flags.

        When EC2 §3.1.9 confinement is active (sigma_2 > 0), the confined
        characteristic strength f_ck_c is used as the base, then:
        - If use_characteristic=True: returns f_ck_c (no reduction)
        - If use_accidental=True: returns f_ck_c * alpha_cc / gamma_c_accidental
        - Otherwise (default): returns f_ck_c * alpha_cc / gamma_c
        """
        if self.is_ec2_confined:
            if self.use_characteristic:
                return self.f_ck_c
            if self.use_accidental:
                accidental_factor = float(self.concrete.alpha_cc) / float(self.concrete.gamma_c_accidental)
                return self.f_ck_c * accidental_factor
            # Default: design strength
            design_factor = float(self.concrete.alpha_cc) / float(self.concrete.gamma_c)
            return self.f_ck_c * design_factor

        if self.use_characteristic:
            return self.concrete.f_ck
        if self.use_accidental:
            return self.concrete.f_cd_accidental
        return self.concrete.f_cd

    @property
    def epsilon_c2_eff(self) -> float:
        """Effective strain at peak (confined or unconfined)."""
        return self.epsilon_c2_c if self.is_ec2_confined else float(self.concrete.epsilon_c2)

    @property
    def epsilon_cu2_eff(self) -> float:
        """Effective ultimate strain (confined or unconfined)."""
        return self.epsilon_cu2_c if self.is_ec2_confined else float(self.concrete.epsilon_cu2)

    @model_validator(mode="after")
    def validate_parameters(self) -> "ConcreteStressStrainParabolaRectangle":
        if self.use_characteristic and self.use_accidental:
            raise ValueError(
                "Cannot set both use_characteristic=True and use_accidental=True. "
                "Choose one: characteristic (f_ck), design (f_cd), or accidental (f_cd_accidental)."
            )
        if self.concrete.epsilon_c2 <= 0:
            raise ValueError(f"Concrete epsilon_c2 must be > 0, got {self.concrete.epsilon_c2}")
        if self.concrete.epsilon_cu2 <= 0:
            raise ValueError(f"Concrete epsilon_cu2 must be > 0, got {self.concrete.epsilon_cu2}")
        if self.concrete.epsilon_cu2 < self.concrete.epsilon_c2:
            raise ValueError("Concrete epsilon_cu2 must be >= epsilon_c2")
        if self.f_c <= 0:
            raise ValueError(f"Concrete strength f_c must be > 0, got {self.f_c}")
        if self.concrete.n <= 0:
            raise ValueError(f"Concrete exponent n must be > 0, got {self.concrete.n}")
        return self


    def get_stress(self, strain: float) -> float:
        """Calculate stress for given strain (compression positive)."""
        if strain <= 0.0:
            return 0.0

        eps_cu = self.epsilon_cu2_eff
        eps_c2 = self.epsilon_c2_eff
        tol = float(self.ultimate_strain_tol)

        if strain > eps_cu + tol:
            return 0.0
        if strain > eps_cu:
            strain = eps_cu

        # Rectangular portion
        if strain >= eps_c2:
            return float(self.f_c)

        # Parabolic portion
        ratio = 1.0 - strain / eps_c2
        return float(self.f_c * (1.0 - ratio ** self.concrete.n))


    def get_stress_array(self, strains: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Vectorized stress calculation.

        Supports complex input for complex-step differentiation (preserves imaginary part).
        For branch selection (comparisons), uses real part only.
        """
        # Accept complex input for complex-step differentiation
        strains = np.asarray(strains)
        stresses = np.zeros_like(strains)

        # Use real part for comparisons (branch selection)
        strains_real = np.real(strains)
        comp = strains_real > 0.0
        if not np.any(comp):
            return stresses

        eps_cu = self.epsilon_cu2_eff
        eps_c2 = self.epsilon_c2_eff

        # Clip using real part, but preserve complex
        strains_real_clipped, killed = _apply_ultimate_tolerance_clip(
            strains=strains_real,
            epsilon_cu=eps_cu,
            tol=float(self.ultimate_strain_tol),
        )
        # Reconstruct complex with clipped real part
        if np.iscomplexobj(strains):
            strains_clipped = strains_real_clipped + 1j * np.imag(strains)
        else:
            strains_clipped = strains_real_clipped

        valid = comp & (~killed)
        if not np.any(valid):
            return stresses

        # Parabolic region: 0 < ε <= ε_c2 (use real for comparison)
        parabolic = valid & (strains_real_clipped <= eps_c2)
        if np.any(parabolic):
            ratio = 1.0 - strains_clipped[parabolic] / eps_c2
            stresses[parabolic] = self.f_c * (1.0 - ratio ** self.concrete.n)

        # Rectangular region: ε_c2 < ε <= ε_cu2 (use real for comparison)
        rectangular = valid & (strains_real_clipped > eps_c2)
        stresses[rectangular] = self.f_c

        return stresses


    def get_ultimate_strain(self) -> float:
        """Return ultimate strain (confined or unconfined)."""
        return self.epsilon_cu2_eff


    def get_yield_stress(self) -> float:
        """Return design/characteristic strength."""
        return float(self.f_c)


    def get_tangent_modulus(self, strain: float) -> float:
        """
        Compute tangent modulus E_t = dσ/dε at given strain.

        For parabola-rectangle model:
        - Parabolic region (0 < ε ≤ ε_c2):
            σ = f_c * [1 - (1 - ε/ε_c2)^n]
            E_t = f_c * n * (1/ε_c2) * (1 - ε/ε_c2)^(n-1)

        - Rectangular region (ε_c2 < ε ≤ ε_cu2):
            σ = f_c (constant)
            E_t = 0 (zero gradient)

        - Outside limits (ε ≤ 0 or ε > ε_cu2):
            E_t = 0 (no tension stiffness, post-crushing)

        Args:
            strain: Compressive strain (positive for compression)

        Returns:
            Tangent modulus in MPa (compression positive)
        """
        if strain <= 0.0:
            return 0.0  # No tensile stiffness

        eps_c2 = self.epsilon_c2_eff
        eps_cu = self.epsilon_cu2_eff

        if strain > eps_cu:
            return 0.0  # Post-crushing: no stiffness

        # Rectangular region: constant stress → zero gradient
        if strain >= eps_c2:
            return 0.0

        # Parabolic region: dσ/dε = f_c * n * (1/ε_c2) * (1 - ε/ε_c2)^(n-1)
        n = float(self.concrete.n)
        ratio = 1.0 - strain / eps_c2
        E_t = self.f_c * n * (1.0 / eps_c2) * (ratio ** (n - 1))
        return float(E_t)


    def get_tangent_modulus_array(self, strains: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Vectorized tangent modulus calculation.

        Args:
            strains: Array of compressive strains

        Returns:
            Array of tangent moduli in MPa
        """
        strains = np.asarray(strains)
        E_t = np.zeros_like(strains, dtype=float)

        eps_c2 = self.epsilon_c2_eff
        n = float(self.concrete.n)

        # Only parabolic region has non-zero gradient
        # (rectangular and outside limits have E_t=0)
        parabolic = (strains > 0.0) & (strains < eps_c2)
        if np.any(parabolic):
            ratio = 1.0 - strains[parabolic] / eps_c2
            E_t[parabolic] = self.f_c * n * (1.0 / eps_c2) * (ratio ** (n - 1))

        return E_t


class ConcreteStressStrainBilinear(BaseConstitutiveModel):
    """
    Bilinear stress-strain diagram (EC2 Fig 3.4).

    Simplified design diagram.

    Formulation:
        σ_c = f_c · ε/ε_c3       for 0 ≤ ε ≤ ε_c3
        σ_c = f_c                for ε_c3 < ε ≤ ε_cu3
        σ_c = 0                  for ε > ε_cu3

    The strength f_c can be (mutually exclusive):
        - f_cd (design strength) when use_characteristic=False and use_accidental=False (default)
        - f_ck (characteristic strength) when use_characteristic=True
        - f_cd_accidental (accidental design strength) when use_accidental=True

    Note: use_characteristic and use_accidental cannot both be True.
    """

    concrete: ConcreteMaterial = Field(..., description="Concrete material")

    use_characteristic: bool = Field(
        default=False,
        description="Use f_ck instead of f_cd (mutually exclusive with use_accidental)"
    )

    use_accidental: bool = Field(
        default=False,
        description="Use f_cd_accidental instead of f_cd (mutually exclusive with use_characteristic)"
    )

    name: str = Field(default="EC2 Bilinear", description="Model name")

    ultimate_strain_tol: float = Field(
        default=1e-12,
        ge=0.0,
        description="Tolerance for ultimate strain clipping (dimensionless strain).",
    )

    @property
    def f_c(self) -> float:
        """Design, characteristic, or accidental strength depending on flags."""
        if self.use_characteristic:
            return self.concrete.f_ck
        if self.use_accidental:
            return self.concrete.f_cd_accidental
        return self.concrete.f_cd

    @model_validator(mode="after")
    def validate_parameters(self) -> "ConcreteStressStrainBilinear":
        if self.use_characteristic and self.use_accidental:
            raise ValueError(
                "Cannot set both use_characteristic=True and use_accidental=True. "
                "Choose one: characteristic (f_ck), design (f_cd), or accidental (f_cd_accidental)."
            )
        if self.concrete.epsilon_c3 <= 0:
            raise ValueError(f"Concrete epsilon_c3 must be > 0, got {self.concrete.epsilon_c3}")
        if self.concrete.epsilon_cu3 <= 0:
            raise ValueError(f"Concrete epsilon_cu3 must be > 0, got {self.concrete.epsilon_cu3}")
        if self.concrete.epsilon_cu3 < self.concrete.epsilon_c3:
            raise ValueError("Concrete epsilon_cu3 must be >= epsilon_c3")
        if self.f_c <= 0:
            raise ValueError(f"Concrete strength f_c must be > 0, got {self.f_c}")
        return self

    def get_stress(self, strain: float) -> float:
        """Calculate stress for given strain (compression positive)."""
        if strain <= 0.0:
            return 0.0

        eps_cu = float(self.concrete.epsilon_cu3)
        tol = float(self.ultimate_strain_tol)

        if strain > eps_cu + tol:
            return 0.0
        if strain > eps_cu:
            strain = eps_cu

        # Constant stress portion
        if strain >= self.concrete.epsilon_c3:
            return float(self.f_c)

        # Linear portion
        return float(self.f_c * strain / self.concrete.epsilon_c3)

    def get_stress_array(self, strains: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Vectorized stress calculation."""
        strains = np.asarray(strains, dtype=float)
        stresses = np.zeros_like(strains)

        comp = strains > 0.0
        if not np.any(comp):
            return stresses

        eps_cu = float(self.concrete.epsilon_cu3)
        strains_clipped, killed = _apply_ultimate_tolerance_clip(
            strains=strains,
            epsilon_cu=eps_cu,
            tol=float(self.ultimate_strain_tol),
        )

        valid = comp & (~killed)
        if not np.any(valid):
            return stresses

        # Linear region: 0 < ε <= ε_c3
        linear = valid & (strains_clipped <= self.concrete.epsilon_c3)
        if np.any(linear):
            stresses[linear] = self.f_c * strains_clipped[linear] / self.concrete.epsilon_c3

        # Constant region: ε_c3 < ε <= ε_cu3
        constant = valid & (strains_clipped > self.concrete.epsilon_c3)
        stresses[constant] = self.f_c

        return stresses

    def get_ultimate_strain(self) -> float:
        """Return ultimate strain."""
        return float(self.concrete.epsilon_cu3)

    def get_yield_stress(self) -> float:
        """Return design/characteristic strength."""
        return float(self.f_c)


class ConcreteStressStrainLinearElastic(BaseConstitutiveModel):
    """
    Linear elastic stress-strain relationship for concrete.

    Primarily used for Serviceability Limit State (SLS) analysis where concrete
    stresses remain in the elastic range. Non-linearity at SLS is typically handled
    through the effective modulus E_cm,eff rather than the constitutive law.

    Formulation:
        Compression (ε > 0): σ = E_mod × ε
        Tension (ε < 0, when include_tension=True):
            σ = E_mod × ε  for |ε| ≤ f_ctm / E_mod
            σ = 0           for |ε| > f_ctm / E_mod  (cracked, brittle cutoff)
        Tension (ε < 0, when include_tension=False):
            σ = 0

    The elastic modulus can be set explicitly (e.g. E_cm_eff for long-term
    creep-reduced analysis) or defaults to E_cm from the concrete material.

    Sign convention:
        - Strain > 0 => compression, Stress > 0 => compression
        - Strain < 0 => tension, Stress < 0 => tension (when include_tension=True)
    """

    concrete: ConcreteMaterial = Field(..., description="Concrete material")
    name: str = Field(default="EC2 Linear Elastic", description="Model name")

    elastic_modulus: Optional[float] = Field(
        default=None,
        description="Elastic modulus in MPa. If None, uses concrete.E_cm.",
        gt=0.0,
    )

    include_tension: bool = Field(
        default=False,
        description="If True, model concrete tension up to f_ctm (brittle cutoff).",
    )

    @property
    def E_mod(self) -> float:
        """Effective elastic modulus (MPa)."""
        if self.elastic_modulus is not None:
            return self.elastic_modulus
        return self.concrete.E_cm

    @property
    def cracking_strain(self) -> float:
        """Cracking strain (negative, tension convention). Only meaningful when include_tension=True."""
        return -self.concrete.f_ctm / self.E_mod

    def get_stress(self, strain: float) -> float:
        """
        Calculate stress for given strain.

        Args:
            strain: Strain (compression positive)

        Returns:
            Stress in MPa (compression positive, tension negative when enabled)
        """
        if strain > 0.0:
            return self.E_mod * strain

        if strain == 0.0:
            return 0.0

        # Tension (strain < 0)
        if not self.include_tension:
            return 0.0

        if strain < self.cracking_strain:
            return 0.0

        return self.E_mod * strain

    def get_stress_array(self, strains: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Vectorized stress calculation."""
        strains = np.asarray(strains)
        stresses = np.zeros_like(strains)

        # Compression: σ = E × ε
        comp = strains > 0.0
        if np.any(comp):
            stresses[comp] = self.E_mod * strains[comp]

        # Tension (optional)
        if self.include_tension:
            tension = (strains < 0.0) & (strains >= self.cracking_strain)
            if np.any(tension):
                stresses[tension] = self.E_mod * strains[tension]

        return stresses

    def get_tangent_modulus(self, strain: float) -> float:
        """Tangent modulus (analytical)."""
        if strain > 0.0:
            return self.E_mod
        if self.include_tension and self.cracking_strain <= strain < 0.0:
            return self.E_mod
        return 0.0

    def get_tangent_modulus_array(self, strains: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Vectorized tangent modulus."""
        strains = np.asarray(strains)
        E_t = np.zeros_like(strains)

        active = strains > 0.0
        if self.include_tension:
            active = active | ((strains < 0.0) & (strains >= self.cracking_strain))

        E_t[active] = self.E_mod
        return E_t

    def get_ultimate_strain(self) -> float:
        """No defined ultimate strain for linear elastic at SLS; return large value."""
        return 0.01  # 1% - well beyond any SLS strain

    def get_yield_stress(self) -> float:
        """Return f_ck as reference strength."""
        return float(self.concrete.f_ck)


def create_concrete_stress_strain(
    concrete: ConcreteMaterial,
    model_type: ConcreteModelType = ConcreteModelType.PARABOLA_RECTANGLE,
    use_characteristic: bool = False,
    use_accidental: bool = False,
    elastic_modulus: Optional[float] = None,
    include_tension: bool = False,
) -> BaseConstitutiveModel:
    """
    Factory function to create concrete stress-strain models.

    Args:
        concrete: Concrete material
        model_type: Type of model to create
        use_characteristic: Use f_ck instead of f_cd (ignored for schematic and linear-elastic)
        use_accidental: Use f_cd_accidental instead of f_cd (ignored for schematic and linear-elastic)
        elastic_modulus: Elastic modulus override in MPa (only used for LINEAR_ELASTIC)
        include_tension: Model tension up to f_ctm (only used for LINEAR_ELASTIC)

    Returns:
        Concrete stress-strain model

    Raises:
        ValueError: If both use_characteristic and use_accidental are True

    Note:
        use_characteristic and use_accidental are mutually exclusive.
    """
    match model_type:
        case ConcreteModelType.SCHEMATIC:
            return ConcreteStressStrainSchematic(concrete=concrete)

        case ConcreteModelType.PARABOLA_RECTANGLE:
            return ConcreteStressStrainParabolaRectangle(
                concrete=concrete,
                use_characteristic=use_characteristic,
                use_accidental=use_accidental
            )

        case ConcreteModelType.BILINEAR:
            return ConcreteStressStrainBilinear(
                concrete=concrete,
                use_characteristic=use_characteristic,
                use_accidental=use_accidental
            )

        case ConcreteModelType.LINEAR_ELASTIC:
            return ConcreteStressStrainLinearElastic(
                concrete=concrete,
                elastic_modulus=elastic_modulus,
                include_tension=include_tension,
            )

        case _:
            raise ValueError(f"Unknown model type: {model_type}")
