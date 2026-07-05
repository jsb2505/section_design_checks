# shear_utils.py
'''
Docstring for materials.reinforced_concrete.code_checks.ec2.shear_utils

Utility functions for shear design checks according to Eurocode 2 (EC2).
'''
from dataclasses import dataclass
from math import sqrt, isfinite, radians, copysign, tan
from typing import Optional, cast

from shapely.geometry import MultiLineString, Point

from materials.utils.helpers import cot
from materials.reinforced_concrete.geometry import RCSection
from materials.reinforced_concrete.materials import ShearRebar
from materials.reinforced_concrete.ndp import get_ndp, get_ndp_callable
from materials.core.units import LengthUnit, ForceUnit, from_mm, from_kn


# ==============================================================================
# Tension Shift Rule (EC2 §9.2.1.3)
# ==============================================================================

@dataclass(frozen=True)
class TensionShiftResult:
    """
    Result of tension shift calculation (EC2 §9.2.1.3).

    The tension shift rule accounts for the additional tensile force in
    longitudinal reinforcement due to the truss model for shear. This shifts
    the moment envelope by a_l towards the support.

    Attributes:
        M_design: The design moment after applying tension shift (kN·m).
                  Sign matches M_Ed.
        M_add: Additional moment magnitude from tension shift (kN·m), always >= 0.
        shift_distance_a_l: Shift distance a_l (mm).
        cot_theta: Strut angle cotangent, if shear reinforcement was provided.
        capped_by_M_cap: True if M_design was capped by M_cap.
        z: Lever arm used in calculation (mm).
        d: Effective depth used in calculation (mm).
    """
    M_design: float
    M_add: float
    shift_distance_a_l: float
    cot_theta: Optional[float]
    capped_by_M_cap: bool
    z: float
    d: float


def calculate_tension_shift(
    *,
    M_Ed: float,
    V_Ed: float,
    z: float,
    d: float,
    M_cap: Optional[float] = None,
    b_w: Optional[float] = None,
    f_cd: Optional[float] = None,
    f_ck: Optional[float] = None,
    sigma_cp: float = 0.0,
    shear_reinforcement: Optional[ShearRebar] = None,
    cot_theta_override: Optional[float] = None,
) -> TensionShiftResult:
    """
    Apply EC2 §9.2.1.3 tension shift rule to a bending moment.

    The tension shift accounts for the additional tensile force in longitudinal
    reinforcement due to shear (truss analogy). The moment envelope is effectively
    shifted towards the support by distance a_l.

    With shear reinforcement (variable strut angle method):
        - cot(θ) is calculated from V_Ed using the V_Rd,max formula
        - a_l = 0.5 · z · cot(θ) for vertical links (EC2 Eq. 9.2)

    Without shear reinforcement:
        - a_l = d (EC2 §9.2.1.3(2))

    The shifted moment is: M_design = M_Ed + sign(M_Ed) * M_add
    where M_add = |V_Ed| * a_l / 1000 (converting a_l from mm to m)

    If M_cap is provided, the result is capped: |M_design| ≤ |M_cap|

    Args:
        M_Ed: Design bending moment (kN·m)
        V_Ed: Design shear force (kN)
        z: Lever arm (mm). Typically 0.9d or from strain analysis.
        d: Effective depth (mm). Used as a_l when no shear reinforcement.
        M_cap: Optional moment capacity cap (kN·m). Limits |M_design| ≤ |M_cap|.
        b_w: Web width (mm). Required if shear_reinforcement is provided and
                cot_theta_override is not.
        f_cd: Design concrete strength (MPa). Required if shear_reinforcement is
              provided and cot_theta_override is not.
        f_ck: Characteristic concrete strength (MPa). Required if shear_reinforcement
              is provided and cot_theta_override is not.
        sigma_cp: Axial stress in concrete (MPa), for α_cw calculation. Default 0.
        shear_reinforcement: Optional ShearRebar object. If provided, calculates
                            cot(θ) from V_Ed using the variable strut angle method
                            (unless cot_theta_override is given).
        cot_theta_override: Optional user-supplied cot(θ) value. When provided with
                           shear_reinforcement, this value is used directly instead
                           of calculating cot(θ) from V_Ed and V_Rd,max. Is clamped
                           to be in the valid EC2 range [1.0, 2.5].

    Returns:
        TensionShiftResult with shifted moment and calculation details.

    Raises:
        ValueError: If shear_reinforcement is provided but b_w, f_cd, or f_ck is missing.

    Example:
        >>> # Without shear reinforcement (simple case)
        >>> result = calculate_tension_shift(
        ...     M_Ed=100.0, V_Ed=50.0, z=450.0, d=500.0
        ... )
        >>> print(f"M_design = {result.M_design:.1f} kN·m")  # M_Ed + V_Ed * d / 1000

        >>> # With shear reinforcement
        >>> from materials.reinforced_concrete.materials import ShearRebar, Rebar
        >>> links = ShearRebar(rebar=Rebar(diameter=10), spacing=150, n_legs=2)
        >>> result = calculate_tension_shift(
        ...     M_Ed=100.0, V_Ed=150.0, z=450.0, d=500.0,
        ...     b_w=300.0, f_cd=20.0, f_ck=30.0,
        ...     shear_reinforcement=links
        ... )
    """
    # Validate inputs for shear reinforcement case (only needed when computing cot_theta)
    if shear_reinforcement is not None and cot_theta_override is None:
        missing = []
        if b_w is None:
            missing.append("b_w")
        if f_cd is None:
            missing.append("f_cd")
        if f_ck is None:
            missing.append("f_ck")
        if missing:
            raise ValueError(
                f"When shear_reinforcement is provided without cot_theta_override, "
                f"the following parameters are required: {', '.join(missing)}"
            )

    abs_M_Ed = abs(float(M_Ed))
    abs_V_Ed = abs(float(V_Ed))
    z = float(z)
    d = float(d)

    # Calculate shift distance a_l and cot(θ)
    cot_theta: Optional[float] = None

    if shear_reinforcement is not None:
        if cot_theta_override is not None:
            # User-supplied cot(θ)
            cot_theta = clamp_cot_theta(cot_theta_override)
        else:
            # Type narrowing: validation above ensures these are not None
            assert f_cd is not None
            assert f_ck is not None
            assert b_w is not None

            # Variable strut angle method (EC2 §6.2.3)
            # K = α_cw · b_w · z · ν · f_cd
            alpha_cw = find_alpha_cw(f_cd=f_cd, sigma_cp=sigma_cp)
            nu = find_nu_factor(f_ck=f_ck)
            K = alpha_cw * b_w * z * nu * f_cd  # in N

            cot_theta = find_cot_theta_for_V_Ed(
                V_Ed=V_Ed,
                K=K,
                link_angle_degrees=shear_reinforcement.angle,
            )
        # EC2 §9.2.1.3: a_l = z(cot θ - cot α)/2
        # where α is the stirrup angle (90° for vertical, typically 45° for inclined)
        alpha_rad = radians(float(shear_reinforcement.angle))
        cot_alpha = 1.0 / tan(alpha_rad) if shear_reinforcement.angle != 90 else 0.0
        a_l = z * (cot_theta - cot_alpha) / 2.0
        # Ensure a_l is non-negative (inclined stirrups with steep strut angles could give negative)
        a_l = max(a_l, 0.0)
    else:
        # No shear reinforcement: a_l = d (EC2 §9.2.1.3(2))
        a_l = d

    # Calculate additional moment
    M_add = abs_V_Ed * from_mm(a_l, LengthUnit.M)

    # Calculate shifted moment magnitude
    abs_M_design = abs_M_Ed + M_add

    # Apply M_cap if provided
    capped = False
    if M_cap is not None:
        abs_M_cap = abs(float(M_cap))
        if abs_M_design > abs_M_cap:
            abs_M_design = abs_M_cap
            capped = True

    # Restore sign of original moment
    M_design = copysign(abs_M_design, float(M_Ed))

    return TensionShiftResult(
        M_design=M_design,
        M_add=M_add,
        shift_distance_a_l=a_l,
        cot_theta=cot_theta,
        capped_by_M_cap=capped,
        z=z,
        d=d,
    )


def calculate_section_breadth(
    section: RCSection,
    n_slices: int = 50,
) -> float:
    """
    Calculate the minimum web breadth b_w for shear design per EC2 §6.2.

    EC2 defines b_w as "the minimum width between tension and compression chords".
    This function finds the minimum horizontal width of the section by slicing
    at multiple heights.

    For rectangular sections, this returns the section width.
    For T-beams or I-beams, this returns the web width (narrowest part).

    Args:
        section: RCSection object
        n_slices: Number of horizontal slices for sampling (default 50)

    Returns:
        Minimum web breadth b_w in mm
    """
    from shapely.geometry import LineString

    outline = section.outline
    min_x, min_y, max_x, max_y = outline.bounds

    # Sample the section at multiple heights
    height = max_y - min_y
    if height < 1e-6:
        return max_x - min_x  # Degenerate case

    min_width = max_x - min_x  # Start with bounding box width

    for i in range(1, n_slices):
        # Slice at this height (avoid exact boundaries)
        y = min_y + (i / n_slices) * height

        # Create horizontal line across section
        line = LineString([(min_x - 1, y), (max_x + 1, y)])

        # Find intersection with section outline
        intersection = outline.intersection(line)

        if intersection.is_empty:
            continue

        # Calculate width at this height
        if isinstance(intersection, LineString):
            # Single intersection line
            coords = list(intersection.coords)
            if len(coords) >= 2:
                width = abs(coords[-1][0] - coords[0][0])
                min_width = min(min_width, width)
        elif isinstance(intersection, MultiLineString):
            # Multiple intersection segments (e.g., hollow section)
            # Sum the widths of all segments
            total_width = 0.0
            for geom in intersection.geoms:
                coords = list(geom.coords)
                if len(coords) >= 2:
                    total_width += abs(coords[-1][0] - coords[0][0])
            if total_width > 0:
                min_width = min(min_width, total_width)
        elif isinstance(intersection, Point):
            # Tangent point - skip
            continue

    return min_width


def find_cot_theta_for_V_Ed(
    V_Ed: float,
    K: float,  # product of: alpha_cw * b_w * z * nu_1 * f_cd
    link_angle_degrees: float = 90.0,
    cot_min: float = 1.0,
    cot_max: float = 2.5,
) -> float:
    """
    Find the cotangent of strut angle θ that satisfies V_Rd,max = V_Ed.

    This is useful for determining the actual strut angle in the section based on
    the applied shear force, which is then used for the tension shift rule.

    Solves: V = [α_cw · b_w · z · ν₁ · f_cd * (cot(θ) + cot(α)] / [1 + cot²(θ)]

    Let:
        x = cot(θ),
        C = cot(α)
        K = α_cw · b_w · z · ν₁ · f_cd
    
    Then:
        Vx² - Kx + (V - KC) = 0

    Args:
        V_Ed: Design shear force in kN
        K: product of: (α_cw · b_w · z · ν₁ · f_cd) in N
        link_angle_degrees: angle of the shear links in degrees (default = 90.0)
        cot_min: lower bound cotangent of angle of strut (default = 1.0)
        cot_max: upper bound cotangent of angle of strut (default = 2.5)

    Returns:
        cot(θ) clamped to EC2 range [1.0, 2.5]
    """
    # Normalise inputs
    V_Ed_N = from_kn(abs(float(V_Ed)), ForceUnit.N)
    K = float(K)

    if abs(link_angle_degrees - 90.0) < 1e-9:
        C = 0.0
    else:
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


def clamp_cot_theta(
    cot_theta: float,
    *,
    cot_min: Optional[float] = None,
    cot_max: Optional[float] = None,
) -> float:
    """
    Clamps the cotangent of the compressive strut angle to within bounds.

    Args:
        cot_theta: calculated or user supplied theta value unbounded
        cot_min: lower bound cotangent of angle of strut (default from NDP)
        cot_max: upper bound cotangent of angle of strut (default from NDP)

    Returns:
        Clamped cot theta within bounds
    """
    if cot_min is None:
        cot_min = cast(float, get_ndp("cot_theta_lower_lim"))
    if cot_max is None:
        cot_max = cast(float, get_ndp("cot_theta_upper_lim"))
    return max(cot_min, min(cot_max, cot_theta))


def find_alpha_cw(f_cd: float, sigma_cp: float) -> float:
    """
    Calculate coefficient α_cw for strut capacity (§6.2.3(3)).

    This coefficient accounts for the state of stress in the compression chord.

    NDP: National Annex dependent:
    - Base EC2 & UK NA: Piecewise formula based on σ_cp/f_cd ratio
    - German NA: α_cw = 1.0 (constant)

    Args:
        f_cd: Design compressive strength of concrete in MPa
        sigma_cp: Compressive stress from axial force in MPa

    Returns:
        Coefficient α_cw (dimensionless)
    """
    alpha_cw_fn = get_ndp_callable("alpha_cw")
    return alpha_cw_fn(f_cd, sigma_cp)


def find_nu_factor(f_ck: float) -> float:
    """
    Strength reduction factor for concrete cracked in shear (§6.2.2(6), Eq. 6.6N).

    NDP: Can be either:
    - Formula: ν = 0.6·(1 - f_ck/250) (Eurocode, UK National Annex)
    - Constant: ν = 0.675 (German National Annex for shear)

    Args:
        f_ck: Characteristic cylinder strength of concrete in MPa

    Returns:
        ν factor (dimensionless)
    """
    nu_fn = get_ndp_callable("nu_shear")
    return nu_fn(f_ck)


def find_nu_1_factor(f_ck: float, link_angle_degrees: float) -> float:
    """
    Strength reduction factor ν₁ for V_Rd,max (§6.2.3(3), Eq. 6.14).

    NDP: National Annex dependent:
    - Base EC2: ν₁ = ν = 0.6·(1 - f_ck/250)
    - UK NA: ν₁ = ν·(1 - 0.5·cos(α)) where α is shear reinforcement angle
    - German NA: ν₁ = 0.75·ν₂ where ν₂ = max(1.1 - f_ck/500, 1.0)

    Args:
        f_ck: Characteristic cylinder strength of concrete in MPa
        link_angle_degrees: Shear reinforcement angle to longitudinal axis in degrees

    Returns:
        ν₁ factor (dimensionless)
    """
    nu_1_fn = get_ndp_callable("nu_1")
    return nu_1_fn(f_ck, link_angle_degrees)


def find_nu_1_factor_note_2(f_ck: float, link_angle_degrees: float) -> float:
    """
    Increased strength reduction factor ν₁ for V_Rd,max when σ_s < 0.8·f_yk (§6.2.3(3) Note 2).

    NDP: National Annex dependent:
    - Base EC2: ν₁ = 0.6 for f_ck ≤ 60, ν₁ = max(0.9 - f_ck/200, 0.5) for f_ck > 60
    - UK NA: ν₁ = 0.54·(1 - 0.5·cos(α)) for f_ck ≤ 60, ν₁ = max((0.84 - f_ck/200)·(1 - 0.5·cos(α)), 0.5) for f_ck > 60
    - German NA: Same as base EC2

    Args:
        f_ck: Characteristic cylinder strength of concrete in MPa
        link_angle_degrees: Shear reinforcement angle to longitudinal axis in degrees

    Returns:
        ν₁ factor (dimensionless) - increased value for low-stress reinforcement
    """
    nu_1_fn = get_ndp_callable("nu_1_note_2")
    return nu_1_fn(f_ck, link_angle_degrees)


def find_nu_factor_torsion(f_ck: float) -> float:
    """
    Strength reduction factor for torsion (§6.3.2(4), Eq. 6.30N).

    NDP: Can be either:
    - Formula: ν = 0.6·(1 - f_ck/250) (Eurocode, UK National Annex)
    - Constant: ν = 0.525 (German National Annex for torsion)

    Args:
        f_ck: Characteristic cylinder strength of concrete in MPa

    Returns:
        ν factor for torsion (dimensionless)
    """
    nu_fn = get_ndp_callable("nu_torsion")
    return nu_fn(f_ck)


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


def find_v_min(f_ck: float, k_factor: float, d: float, gamma_c: float) -> float:
    """
    Minimum shear strength coefficient (§6.2.2(1), Eq. 6.3N).

    v_min = coeff·k^(3/2)·√f_ck

    The coefficient is a NDP:
    - EU: 0.035 (constant)
    - German NA: varies with d and gamma_c

    Args:
        f_ck: Characteristic cylinder strength of concrete in MPa
        k_factor: Size effect factor (§6.2.2(1)) (dimensionless)
        d: Effective depth in mm (used by some National Annexes)
        gamma_c: Concrete partial safety factor (used by some National Annexes)

    Returns:
        v_min in MPa
    """
    coeff_fn = get_ndp_callable("v_min_coefficient")
    coeff = coeff_fn(d, gamma_c)
    return coeff * (k_factor ** 1.5) * sqrt(f_ck)


def sigma_cp_from_N_and_area(N_Ed: float, area: float) -> float:
    """
    Compressive stress in the concrete from axial load or prestressing

    Args:
        N_Ed: Design axial force in kN (positive is compression)
        area: Cross-sectional area of section (mm²)

    Returns:
        sigma_cp in MPa
    """
    return from_kn(N_Ed, ForceUnit.N) / area


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


def find_minimum_ratio_of_shear_reinforcement(f_ck: float, f_yk: float, f_ctm: float) -> float:
    '''
    Calculates the minimum ratio of shear reinforcement.
    Ref: EC2 §9.2.2(5) (9.5N)

    This ratio is a NDP:
    - EU: ρ_w_min = 0.08 * sqrt(f_ck) / f_yk
    - German NA: ρ_w_min = 0.16 * f_ctm / f_yk

    Args:
        f_ck: Characteristic cylinder strength of concrete
        f_yk: Characteristic yield strength of rebar
        f_ctm: Characteristic mean tensile strength of concrete

    Returns:
        ρ_w_min: the minimum ratio of shear reinforcement (dimensionless, empirical formula)
    '''
    rho_w_min_fn = get_ndp_callable("rho_w_min")
    return rho_w_min_fn(f_ck, f_yk, f_ctm)
