# utils.py
'''
Docstring for materials.reinforced_concrete.code_checks.ec2.shear_utils

Utility functions for shear design checks according to Eurocode 2 (EC2).
'''
from math import sqrt, isfinite, radians
from materials.utils.helpers import cot
from materials.reinforced_concrete.geometry import RCSection


def calculate_section_breadth(section: RCSection) -> float:
    """
    Web breadth in mm.

    Args:
        section: RCSection object

    Returns:
        shear breadth in mm
    """
    # TODO FOR WEB BEAMS FIND THE BREADTH USED FOR SHEAR AREA
    bounds = section.outline.bounds
    return bounds[2] - bounds[0]


def find_cot_theta_for_V_Ed(
    V_Ed: float,
    K: float,  # product of: alpha_cw * b_w * z * nu * f_cd
    link_angle_degrees: float = 90.0,
    cot_min: float = 1.0,
    cot_max: float = 2.5,
) -> float:
    """
    Find the cotangent of strut angle θ that satisfies V_Rd,max = V_Ed.

    This is useful for determining the actual strut angle in the section based on
    the applied shear force, which is then used for the tension shift rule.

    Solves: V = [α_cw · b_w · z · ν · f_cd * (cot(θ) + cot(α)] / [1 + cot²(θ)]

    Let:
        x = cot(θ),
        C = cot(α)
        K = α_cw · b_w · z · ν · f_cd
    
    Then:
        Vx² - Kx + (V - KC) = 0

    Args:
        V_Ed: Design shear force in kN
        K: product of: (alpha_cw * b_w * z * nu * f_cd) in N
        link_angle_degrees: angle of the shear links in degrees (default = 90.0)
        cot_min: lower bound cotangent of angle of strut (default = 1.0)
        cot_max: upper bound cotangent of angle of strut (default = 2.5)

    Returns:
        cot(θ) clamped to EC2 range [1.0, 2.5]
    """
    # Normalise inputs
    # Convert V_Ed from kN to N
    V_Ed_N = abs(float(V_Ed)) * 1000.0
    K = float(K)
    C = cot(radians(link_angle_degrees))

    if K <= 0.0 or V_Ed_N <= 1e-9:  # guard against invalid inputs and div zero error
        return float(cot_min)  # use maximum angle (θ = 45°, cot = 1.0)

    # Quadratic: Vx² - Kx + (V - KC) = 0
    a = V_Ed_N
    b = -K
    c = V_Ed_N - K*C

    # discriminant = b^2 - 4ac
    discriminant = (b**2) - (4 * a * c)

    # No real solution - V_Ed exceeds capacity even at optimal angle
    if discriminant < 0.0:
        # allow tiny negative due to rounding
        if discriminant > -1e-12:
            discriminant = 0.0
        else:
            return float(cot_min)  # Use steepest angle

    # Two solutions: use the larger one (flatter angle, more efficient)
    x1 = (-b + sqrt(discriminant)) / (2 * a)
    x2 = (-b - sqrt(discriminant)) / (2 * a)

    # Pick the larger positive, finite root (flatter strut angle)
    candidates = [x for x in (x1, x2) if isfinite(x) and x > 0.0]
    if not candidates:
        return float(cot_min)  # Use steepest angle

    cot_theta_calc = max(candidates)

    # Clamp to EC2 bounds
    cot_theta = clamp_cot_theta(cot_theta_calc, cot_min=cot_min, cot_max=cot_max)

    return cot_theta


def clamp_cot_theta(cot_theta: float, *, cot_min: float = 1.0, cot_max: float = 2.5) -> float:
    """
    Clamps the cotangent of the compressive strut angle to within bounds.

    Args:
        cot_theta: calculated or user supplied theta value unbounded
        cot_min: lower bound cotangent of angle of strut (default = 1.0)
        cot_max: upper bound cotangent of angle of strut (default = 2.5)

    Returns:
        Clamped cot theta within bounds
    """
    return max(cot_min, min(cot_max, cot_theta))


def find_alpha_cw(f_cd: float, sigma_cp: float) -> float:
    """
    Calculate coefficient α_cw for strut capacity (§6.2.3(3)).

    This coefficient accounts for the state of stress in the compression chord.

    Args:
        sigma_cp: Compressive stress from axial force in MPa

    Returns:
        Coefficient α_cw (dimensionless)
    """
    if sigma_cp == 0:
        return 1.0
    elif sigma_cp <= 0.25 * f_cd:
        return 1.0 + sigma_cp / f_cd
    elif sigma_cp <= 0.5 * f_cd:
        return 1.25
    else:
        return 2.5 * (1 - sigma_cp / f_cd)


def find_nu_factor(f_ck: float) -> float:
    """
    Strength reduction factor for concrete cracked in shear (§6.2.2(6), Eq. 6.6N).

    ν = 0.6·(1 - f_ck/250)

    Args:
        f_ck: Characteristic cylinder strength of concrete in MPa

    Returns:
        ν factor (dimensionless)
    """
    return 0.6 * (1 - f_ck / 250)


def find_k_factor(d: float) -> float:
    """
    Size effect factor (§6.2.2(1)).

    k = 1 + √(200/d) ≤ 2.0

    Args:
        d: Effective depth in mm

    Returns:
        k factor (dimensionless)
    """
    if d <= 0:
        raise ValueError(f"Effective depth must be > 0, got {d} mm")
    return min(2.0, 1.0 + sqrt(200 / d))


def find_v_min(f_ck: float, k_factor: float) -> float:
    """
    Minimum shear strength coefficient (§6.2.2(1), Eq. 6.3N).

    v_min = 0.035·k^(3/2)·√f_ck

    Args:
        f_ck: Characteristic cylinder strength of concrete in MPa
        k_factor: Size effect factor (§6.2.2(1)) (dimensionless)

    Returns:
        v_min in MPa
    """
    return 0.035 * (k_factor ** 1.5) * sqrt(f_ck)


def sigma_cp_from_N_and_area(N_Ed: float, A_mm2: float) -> float:
    """
        Compressive stress in the concrete from axial load or prestressing

        Args:
            N_Ed: Design axial force in kN (positive is compression)
            A_mm2: Cross-sectional area of section in mm²

        Returns:
            sigma_cp in MPa
    """
    return (N_Ed * 1000.0) / A_mm2


def cap_sigma_cp_upper(sigma_cp: float, f_cd: float) -> float:
    """
        Capped compressive stress in the concrete from axial load or prestressing.
        Only caps positive compressive stresses.

        Args:
            sigma_cp: uncapped compressive stress in MPa
            f_cd: Design cylinder strength of concrete in MPa

        Returns:
            sigma_cp_capped in MPa
    """
    return min(sigma_cp, 0.2 * f_cd)
