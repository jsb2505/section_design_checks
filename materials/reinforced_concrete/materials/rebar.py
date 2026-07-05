"""
Rebar (reinforcing bar) with geometry and material properties.
"""

import warnings
from typing import Final
from math import pi, sin, radians
from pydantic import Field, computed_field, field_validator
from materials.reinforced_concrete.materials.reinforcing_steel import ReinforcingSteel
from materials.utils.helpers import cot
from materials.core.units import LengthUnit, LENGTH_TO_MM

# Standard bar diameters in mm (EC2 common practice) - single source of truth
STANDARD_BAR_DIAMETERS: Final = (6, 8, 10, 12, 16, 20, 25, 28, 32, 40)


class Rebar(ReinforcingSteel):
    """
    Reinforcing bar with diameter and material properties.

    Extends ReinforcingSteel to include geometric properties.
    """

    diameter: float = Field(..., description="Bar diameter in mm", gt=0)

    @computed_field
    @property
    def area(self) -> float:
        """Cross-sectional area (mm²): A = π d² / 4."""
        return pi * (float(self.diameter) ** 2) / 4.0

    @computed_field
    @property
    def perimeter(self) -> float:
        """Perimeter (mm): P = π d."""
        return pi * float(self.diameter)
    
    @computed_field
    @property
    def mass_per_metre(self) -> float:
        """
        Mass per unit length (kg/m).
        Calculation: Area(m²) * Density(kg/m³)
        """
        mm_per_m = LENGTH_TO_MM[LengthUnit.M]
        area_m2 = self.area / mm_per_m ** 2
        return area_m2 * self.density
    
    @property
    def is_standard(self) -> bool:
        """Checks if the chosen diameter is a standard Eurocode size."""
        return self.diameter in STANDARD_BAR_DIAMETERS
    
    @field_validator("diameter")
    @classmethod
    def check_standard_size(cls, v: float) -> float:
        if v not in STANDARD_BAR_DIAMETERS:
            warnings.warn(
                f"Diameter {v}mm is not in standard list: {STANDARD_BAR_DIAMETERS}",
                category=UserWarning,
            )
        return v

    def __str__(self) -> str:
        return f"ϕ{self.diameter} {self.grade} (A={self.area:.1f} mm²)"


class ShearRebar(Rebar):
    """Shear reinforcement (links/stirrups) with spacing and leg configuration."""

    spacing: float = Field(..., description="Link spacing along member axis (or pitch if spiral) (mm)", gt=0)
    n_legs: int = Field(default=2, description="Number of link legs crossing the shear plane", ge=1)
    angle: float = Field(default=90.0, description="Angle of links to member axis (degrees)", ge=45.0, le=90.0)

    @computed_field
    @property
    def total_area_per_spacing(self) -> float:
        """A_sw (mm²) = n_legs × A_bar."""
        return self.n_legs * self.area

    @computed_field
    @property
    def area_per_unit_length(self) -> float:
        """A_sw / s (mm²/mm)."""
        return self.total_area_per_spacing / self.spacing

    @computed_field
    @property
    def a_sw_over_s_sin_alpha(self) -> float:
        """
        A_sw / (s · sin α) in mm²/mm.
        Divide by b_w to get full EC2 ρ_w.
        """
        angle_rad = radians(self.angle)
        return self.total_area_per_spacing / (self.spacing * sin(angle_rad))


    #------------------------
    # Utility Helper Methods
    #------------------------

    def max_link_spacing(self, effective_depth: float) -> float:
        """
        EC2 §9.2.2(6): s_l,max = 0.75 d (1 + cot α).

        Note:
            This is an NDP. This function is teh base EC2 version.
        """
        if effective_depth <= 0:
            raise ValueError("effective_depth must be > 0")

        if abs(self.angle - 90.0) < 1e-9:
            cot_alpha = 0.0
        else:
            cot_alpha = cot(radians(self.angle))

        return 0.75 * effective_depth * (1.0 + cot_alpha)

    def max_leg_spacing(self, effective_depth: float) -> float:
        """EC2 §9.2.2(8): s_t,max = min(600 mm, 0.75 d)."""
        if effective_depth <= 0:
            raise ValueError("effective_depth must be > 0")
        # TODO this is an NDP. Germans differ
        return min(600.0, 0.75 * effective_depth)

    def __str__(self) -> str:
        return (
            f"ϕ{self.diameter} {self.grade} links @ {self.spacing}mm c/c, "
            f"{self.n_legs} legs, {self.angle}°"
        )