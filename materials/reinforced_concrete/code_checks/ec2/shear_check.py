"""
Shear check using codified EC2 stress approach.

This is a CODIFIED check with business logic - uses EC2 formulas directly
rather than first principles. Implements §6.2 Variable Strut Inclination Method.
"""

from typing import Optional
from math import sqrt, radians, cos, atan, degrees
from pydantic import Field, computed_field

from materials.reinforced_concrete.code_checks.base_check import (
    BaseCodeCheck,
    CheckResult,
)
from materials.reinforced_concrete.geometry import RCSection
from materials.reinforced_concrete.materials import ConcreteMaterial, ShearRebar


class ShearCheck(BaseCodeCheck):
    """
    EC2 shear check using Variable Strut Inclination Method (§6.2).

    This is a CODIFIED approach with business logic:
    - Uses EC2 empirical formulas (§6.2.2, §6.2.3)
    - Concrete shear resistance V_Rd,c (Eq. 6.2)
    - Shear reinforcement resistance V_Rd,s (Eq. 6.8)
    - Compression strut resistance V_Rd,max (Eq. 6.9, 6.14)
    - Variable strut angle 21.8° ≤ θ ≤ 45° (cot θ = 1.0 to 2.5)

    NOT first principles - uses empirical codified equations.

    Attributes:
        section: RC section geometry
        concrete: Concrete material
        shear_reinforcement: Shear links/stirrups (optional)
        N_Ed: Design axial force in kN (compression positive)
        d: Effective depth in mm (if None, uses section.get_effective_depth())

    Example:
        >>> from materials.reinforced_concrete.geometry import create_rectangular_section
        >>> from materials.reinforced_concrete.materials import ConcreteMaterial, ShearRebar
        >>>
        >>> section = create_rectangular_section(width=300, height=500)
        >>> # ... add tension reinforcement ...
        >>>
        >>> concrete = ConcreteMaterial(grade="C30/37")
        >>> shear_rebar = ShearRebar(diameter=10, spacing=200, n_legs=2, grade="B500B")
        >>>
        >>> check = ShearCheck(
        ...     section=section,
        ...     concrete=concrete,
        ...     shear_reinforcement=shear_rebar,
        ... )
        >>>
        >>> result = check.perform_check(V_Ed=150)  # kN
        >>> print(result)
    """

    section: RCSection = Field(
        ...,
        description="RC section geometry",
    )

    concrete: ConcreteMaterial = Field(
        ...,
        description="Concrete material",
    )

    shear_reinforcement: Optional[ShearRebar] = Field(
        default=None,
        description="Shear links/stirrups (None if unreinforced)",
    )

    N_Ed: float = Field(
        default=0.0,
        description="Design axial force in kN (compression positive)",
    )

    d: Optional[float] = Field(
        default=None,
        description="Effective depth in mm (uses section.get_effective_depth() if None)",
    )

    # Constants from EC2
    MIN_COT_THETA: float = 1.0  # θ = 45°
    MAX_COT_THETA: float = 2.5  # θ = 21.8°

    @computed_field
    @property
    def effective_depth(self) -> float:
        """Effective depth in mm."""
        if self.d is not None:
            return self.d
        else:
            return self.section.get_effective_depth(face="top")

    @computed_field
    @property
    def breadth(self) -> float:
        """Web breadth in mm."""
        # For simple sections, use width. For T-beams, would use web width
        # This is a simplification - could be enhanced
        return self.section.outline.bounds[2] - self.section.outline.bounds[0]

    @computed_field
    @property
    def sigma_cp(self) -> float:
        """
        Compressive stress in concrete due to axial force (§6.2.2(1)).

        σ_cp = N_Ed / A_c, limited to 0.2·f_cd

        Returns:
            Stress in MPa
        """
        if self.N_Ed <= 0:
            return 0.0

        # A_c = b·d (simplified - could use full section area)
        A_c = self.breadth * self.effective_depth  # mm²

        # Convert N_Ed from kN to N, stress to MPa
        sigma_cp_uncapped = (self.N_Ed * 1000) / A_c

        # Limit to 0.2·f_cd per §6.2.2(1)
        return min(sigma_cp_uncapped, 0.2 * self.concrete.f_cd)

    def find_k_factor(self) -> float:
        """
        Size effect factor (§6.2.2(1)).

        k = 1 + √(200/d) ≤ 2.0

        Returns:
            k factor (dimensionless)
        """
        d_mm = self.effective_depth
        return min(2.0, 1.0 + sqrt(200 / d_mm))

    def find_v_min(self) -> float:
        """
        Minimum shear strength coefficient (§6.2.2(1), Eq. 6.3N).

        v_min = 0.035·k^(3/2)·√f_ck

        Returns:
            v_min in MPa
        """
        k = self.find_k_factor()
        f_ck = self.concrete.f_ck
        return 0.035 * (k ** 1.5) * sqrt(f_ck)

    def find_rho_l(self) -> float:
        """
        Longitudinal reinforcement ratio (§6.2.2(1)).

        ρ_l = A_sl / (b_w·d) ≤ 0.02

        Returns:
            ρ_l (dimensionless)
        """
        # Get tension reinforcement area
        A_sl = self.section.get_total_rebar_area()  # mm²

        if A_sl == 0:
            return 0.0

        b_w = self.breadth
        d = self.effective_depth

        rho_l = A_sl / (b_w * d)

        return min(rho_l, 0.02)

    def find_V_Rd_c(self) -> float:
        """
        Design shear resistance without shear reinforcement (§6.2.2, Eq. 6.2).

        V_Rd,c = [C_Rd,c·k·(100·ρ_l·f_ck)^(1/3) + k_1·σ_cp]·b_w·d

        But not less than:
        V_Rd,c,min = (v_min + k_1·σ_cp)·b_w·d

        Returns:
            V_Rd,c in kN
        """
        # Parameters
        C_Rd_c = 0.18 / self.concrete.gamma_c
        k = self.find_k_factor()
        rho_l = self.find_rho_l()
        f_ck = self.concrete.f_ck
        k_1 = 0.15
        sigma_cp = self.sigma_cp
        b_w = self.breadth
        d = self.effective_depth

        # Main formula (Eq. 6.2a)
        V_Rd_c = (C_Rd_c * k * ((100 * rho_l * f_ck) ** (1/3)) +
                 k_1 * sigma_cp) * b_w * d

        # Minimum value (Eq. 6.2b)
        v_min = self.find_v_min()
        V_Rd_c_min = (v_min + k_1 * sigma_cp) * b_w * d

        # Convert from N to kN
        return max(V_Rd_c, V_Rd_c_min) / 1000

    def find_nu_factor(self) -> float:
        """
        Strength reduction factor for concrete cracked in shear (§6.2.2(6), Eq. 6.6N).

        ν = 0.6·(1 - f_ck/250)

        Returns:
            ν factor (dimensionless)
        """
        f_ck = self.concrete.f_ck
        return 0.6 * (1 - f_ck / 250)

    def find_V_Rd_s(self, cot_theta: float = 2.5) -> float:
        """
        Shear resistance of shear reinforcement (§6.2.3(3), Eq. 6.8).

        V_Rd,s = (A_sw/s)·z·f_ywd·cot(θ)

        Args:
            cot_theta: Cotangent of strut angle (1.0 to 2.5)

        Returns:
            V_Rd,s in kN (0 if no shear reinforcement)
        """
        if self.shear_reinforcement is None:
            return 0.0

        # Shear reinforcement ratio (mm²/mm)
        A_sw = self.shear_reinforcement.area  # mm²
        s = self.shear_reinforcement.spacing  # mm
        A_sw_over_s = A_sw / s

        # Lever arm (simplified as 0.9d per §6.2.3)
        z = 0.9 * self.effective_depth

        # Design yield strength
        f_ywd = self.shear_reinforcement.f_yd

        # Limit cot(θ)
        cot_theta_limited = max(self.MIN_COT_THETA, min(self.MAX_COT_THETA, cot_theta))

        # V_Rd,s (N)
        V_Rd_s = A_sw_over_s * z * f_ywd * cot_theta_limited

        # Convert to kN
        return V_Rd_s / 1000

    def find_V_Rd_max(self, cot_theta: float = 2.5) -> float:
        """
        Maximum shear resistance limited by crushing of compression struts (§6.2.3, Eq. 6.9).

        V_Rd,max = α_cw·b_w·z·ν·f_cd / (cot(θ) + tan(θ))

        Args:
            cot_theta: Cotangent of strut angle (1.0 to 2.5)

        Returns:
            V_Rd,max in kN
        """
        # Coefficient α_cw (§6.2.3(3))
        sigma_cp = self.sigma_cp
        f_cd = self.concrete.f_cd

        if sigma_cp == 0:
            alpha_cw = 1.0
        elif sigma_cp <= 0.25 * f_cd:
            alpha_cw = 1.0 + sigma_cp / f_cd  # Eq. 6.11aN
        elif sigma_cp <= 0.5 * f_cd:
            alpha_cw = 1.25  # Eq. 6.11bN
        else:
            alpha_cw = 2.5 * (1 - sigma_cp / f_cd)  # Eq. 6.11cN

        # Parameters
        b_w = self.breadth
        z = 0.9 * self.effective_depth
        nu = self.find_nu_factor()

        # Limit cot(θ)
        cot_theta_limited = max(self.MIN_COT_THETA, min(self.MAX_COT_THETA, cot_theta))

        # tan(θ) = 1/cot(θ)
        tan_theta = 1 / cot_theta_limited

        # V_Rd,max (N)
        V_Rd_max = (alpha_cw * b_w * z * nu * f_cd) / (cot_theta_limited + tan_theta)

        # Convert to kN
        return V_Rd_max / 1000

    def perform_check(
        self,
        V_Ed: float,
        cot_theta: float = 2.5,
        warning_threshold: float = 0.95,
    ) -> CheckResult:
        """
        Perform shear check per EC2 §6.2.

        Checks:
        1. V_Ed ≤ V_Rd,c (if no shear reinforcement)
        2. V_Ed ≤ V_Rd,s (if shear reinforcement provided)
        3. V_Ed ≤ V_Rd,max (crushing of struts)

        Args:
            V_Ed: Design shear force in kN
            cot_theta: Cotangent of strut angle (1.0 to 2.5, default 2.5 for economy)
            warning_threshold: Utilization threshold for warning

        Returns:
            CheckResult with pass/fail status
        """
        # Calculate capacities
        V_Rd_c = self.find_V_Rd_c()
        V_Rd_s = self.find_V_Rd_s(cot_theta=cot_theta)
        V_Rd_max = self.find_V_Rd_max(cot_theta=cot_theta)

        # Determine governing capacity
        if self.shear_reinforcement is None:
            # No shear reinforcement - check concrete capacity
            V_Rd = V_Rd_c
            governing_mode = "concrete (no shear reinforcement)"
            code_ref = "EC2 §6.2.2"
        else:
            # Shear reinforcement provided
            # Need to satisfy both V_Rd,s and V_Rd,max
            if V_Ed > V_Rd_c:
                # Shear reinforcement is engaged
                V_Rd = min(V_Rd_s, V_Rd_max)

                if V_Rd_s < V_Rd_max:
                    governing_mode = "shear reinforcement"
                    code_ref = "EC2 §6.2.3 (Eq. 6.8)"
                else:
                    governing_mode = "compression strut"
                    code_ref = "EC2 §6.2.3 (Eq. 6.9)"
            else:
                # Concrete alone is sufficient
                V_Rd = V_Rd_c
                governing_mode = "concrete"
                code_ref = "EC2 §6.2.2"

        # Create message
        utilization = V_Ed / V_Rd if V_Rd > 0 else float('inf')

        if utilization <= 1.0:
            if utilization >= warning_threshold:
                message = f"High shear utilization - governed by {governing_mode}"
            else:
                message = f"Shear capacity adequate - governed by {governing_mode}"
        else:
            deficit = (utilization - 1.0) * 100
            if self.shear_reinforcement is None:
                message = f"Shear capacity exceeded - provide shear reinforcement"
            elif governing_mode == "compression strut":
                message = f"Compression strut capacity exceeded - increase section size"
            else:
                message = f"Shear reinforcement capacity exceeded - reduce spacing or increase diameter"

        # Details
        details = {
            "V_Ed": V_Ed,
            "V_Rd": V_Rd,
            "V_Rd_c": V_Rd_c,
            "V_Rd_s": V_Rd_s if self.shear_reinforcement else None,
            "V_Rd_max": V_Rd_max if self.shear_reinforcement else None,
            "governing_mode": governing_mode,
            "cot_theta": cot_theta,
            "theta_deg": degrees(atan(1 / cot_theta)),
            "section_name": self.section.section_name or "unnamed",
            "d": self.effective_depth,
            "b_w": self.breadth,
            "rho_l": self.find_rho_l(),
        }

        return self._create_result(
            check_name="Shear check (EC2 §6.2)",
            demand=V_Ed,
            capacity=V_Rd,
            units="kN",
            code_reference=code_ref,
            warning_threshold=warning_threshold,
            message=message,
            details=details,
        )

    def get_required_shear_reinforcement(
        self,
        V_Ed: float,
        cot_theta: float = 2.5,
        f_ywd: Optional[float] = None,
    ) -> float:
        """
        Calculate required A_sw/s for given shear force.

        Args:
            V_Ed: Design shear force in kN
            cot_theta: Cotangent of strut angle
            f_ywd: Design yield strength (uses rebar f_yd if None)

        Returns:
            Required A_sw/s in mm²/mm
        """
        V_Rd_c = self.find_V_Rd_c()

        if V_Ed <= V_Rd_c:
            return 0.0  # Concrete alone is sufficient

        # Required shear reinforcement
        z = 0.9 * self.effective_depth

        if f_ywd is None and self.shear_reinforcement is not None:
            f_ywd = self.shear_reinforcement.f_yd
        elif f_ywd is None:
            # Default to B500B
            f_ywd = 434.8  # MPa

        cot_theta_limited = max(self.MIN_COT_THETA, min(self.MAX_COT_THETA, cot_theta))

        # From V_Rd,s = (A_sw/s)·z·f_ywd·cot(θ)
        # A_sw/s = V_Ed / (z·f_ywd·cot(θ))

        # Convert V_Ed to N
        A_sw_over_s = (V_Ed * 1000) / (z * f_ywd * cot_theta_limited)

        return A_sw_over_s
