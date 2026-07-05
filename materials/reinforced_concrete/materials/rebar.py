"""
Rebar (reinforcing bar) with geometry and material properties.
"""

from typing import Literal, get_args
import math
from pydantic import Field, computed_field, field_validator
from materials.reinforced_concrete.materials.reinforcing_steel import (
    ReinforcingSteel,
)


# Standard bar diameters in mm (EC2 common practice) - single source of truth
BarDiameter = Literal[6, 8, 10, 12, 16, 20, 25, 28, 32, 40]


class Rebar(ReinforcingSteel):
    """
    Reinforcing bar with diameter and material properties.

    Extends ReinforcingSteel to include geometric properties.
    """

    diameter: float = Field(
        ...,
        description="Bar diameter in mm",
        gt=0,
    )

    @field_validator("diameter")
    @classmethod
    def validate_diameter(cls, v: float) -> float:
        """Validate bar diameter is a standard size."""
        standard_diameters = get_args(BarDiameter)

        if v not in standard_diameters:
            raise ValueError(
                f"Bar diameter {v} mm is not a standard size. "
                f"Must be one of {list(standard_diameters)} mm."
            )
        return v

    @computed_field
    @property
    def area(self) -> float:
        """
        Cross-sectional area of the bar.

        A = π · d² / 4

        Returns:
            Area in mm²
        """
        return math.pi * (self.diameter ** 2) / 4.0

    @computed_field
    @property
    def perimeter(self) -> float:
        """
        Perimeter of the bar (for bond calculations).

        P = π · d

        Returns:
            Perimeter in mm
        """
        return math.pi * self.diameter

    def __str__(self) -> str:
        """User-friendly representation."""
        return f"ϕ{self.diameter} {self.grade} (A={self.area:.1f} mm²)"


class ShearRebar(Rebar):
    """
    Shear reinforcement (links/stirrups) with spacing and leg configuration.
    """

    spacing: float = Field(
        ...,
        description="Link spacing along member axis (mm)",
        gt=0,
    )

    n_legs: int = Field(
        default=2,
        description="Number of link legs crossing the shear plane",
        ge=1,
    )

    angle: float = Field(
        default=90.0,
        description="Angle of links to member axis (degrees)",
        ge=45.0,
        le=90.0,
    )

    @computed_field
    @property
    def total_area_per_spacing(self) -> float:
        """
        Total area of shear reinforcement per spacing interval.

        A_sw = n_legs × A_bar

        Returns:
            Total area in mm²
        """
        return self.n_legs * self.area

    @computed_field
    @property
    def area_per_unit_length(self) -> float:
        """
        Shear reinforcement area per unit length.

        A_sw / s (mm²/mm)

        Returns:
            Area per unit length
        """
        return self.total_area_per_spacing / self.spacing

    @computed_field
    @property
    def rho_w(self) -> float:
        """
        Shear reinforcement ratio (§6.2.3).

        ρ_w = A_sw / (s · b_w · sin(α))

        For vertical links (α=90°), sin(α)=1.
        Note: Requires section width b_w for full calculation.
        This returns A_sw/(s·sin(α)) in mm²/mm.

        Returns:
            Partial ratio (divide by b_w for full ρ_w)
        """
        angle_rad = math.radians(self.angle)
        return self.total_area_per_spacing / (self.spacing * math.sin(angle_rad))

    def find_max_link_spacing(self, effective_depth: float) -> float:
        """
        Maximum longitudinal spacing between shear links (EC2 §9.2.2(6), Eq. 9.6N).

        s_l,max = 0.75 · d · (1 + cot α)

        where:
        - d is the effective depth
        - α is the angle of links to member axis

        For vertical links (α = 90°), cot(90°) = 0, so s_l,max = 0.75·d

        Args:
            effective_depth: Effective depth of section in mm

        Returns:
            Maximum spacing in mm
        """
        angle_rad = math.radians(self.angle)
        cot_alpha = 1.0 / math.tan(angle_rad)
        return 0.75 * effective_depth * (1 + cot_alpha)

    def find_max_leg_spacing(self, effective_depth: float) -> float:
        """
        Maximum transverse spacing between link legs (EC2 §9.2.2(8), Eq. 9.8N).

        s_t,max = max(600 mm, 0.75·d)

        Args:
            effective_depth: Effective depth of section in mm

        Returns:
            Maximum leg spacing in mm
        """
        return max(600.0, 0.75 * effective_depth)

    def __str__(self) -> str:
        """User-friendly representation."""
        return (
            f"ϕ{self.diameter} {self.grade} links @ {self.spacing}mm c/c, "
            f"{self.n_legs} legs, {self.angle}°"
        )
