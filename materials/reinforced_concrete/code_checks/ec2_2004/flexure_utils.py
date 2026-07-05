"""
Shared utility functions for flexure-related code checks (bending, cracking).

Contains geometry helpers and EC2 formula implementations used by both
BendingCheck and CrackingCheck.
"""

from typing import TYPE_CHECKING, List, Literal, Optional, Tuple
import warnings

if TYPE_CHECKING:
    from materials.reinforced_concrete.geometry import RCSection
    from materials.reinforced_concrete.analysis.interaction_diagram import MNInteractionDiagram


def calculate_section_height(section: "RCSection") -> float:
    """
    Section height from bounding box.

    Args:
        section: RCSection object

    Returns:
        Section height in mm
    """
    bounds = section.outline.bounds  # (minx, miny, maxx, maxy)
    return bounds[3] - bounds[1]


def calculate_section_breadth(section: "RCSection") -> float:
    """
    Section breadth (width) from bounding box.

    Args:
        section: RCSection object

    Returns:
        Section breadth in mm
    """
    bounds = section.outline.bounds  # (minx, miny, maxx, maxy)
    return bounds[2] - bounds[0]


def calculate_neutral_axis_depth_from_strains(
    eps_top: float,
    eps_bottom: float,
    section_height: float,
) -> Optional[float]:
    """
    Calculate neutral axis depth from top face given strain profile.

    The neutral axis is where strain = 0 (transition from compression to tension).
    Uses linear interpolation (plane sections assumption).

    Sign convention: compression positive, tension negative.

    Args:
        eps_top: Strain at top fibre (compression positive)
        eps_bottom: Strain at bottom fibre (compression positive)
        section_height: Total section height in mm

    Returns:
        Neutral axis depth from top (mm), or None if:
        - Section is fully in compression (both strains positive)
        - Section is fully in tension (both strains negative)
        - Strains are equal (no curvature)
    """
    # Check for uniform strain (no neutral axis)
    strain_diff = eps_top - eps_bottom
    if abs(strain_diff) < 1e-12:
        return None

    # If both positive (compression) or both negative (tension), NA outside section
    if eps_top > 0 and eps_bottom > 0:
        return None  # Fully in compression
    if eps_top < 0 and eps_bottom < 0:
        return None  # Fully in tension

    # Linear interpolation: find where strain = 0
    # y measured from top: strain(y) = eps_top + (eps_bottom - eps_top) * y / h
    # Set strain = 0: y_NA = eps_top * h / (eps_top - eps_bottom)
    x = eps_top * section_height / strain_diff

    # Sanity check: NA should be within section
    if x < 0 or x > section_height:
        return None

    return x


def calculate_compression_face_from_strains(
    eps_top: float,
    eps_bottom: float,
    strain_tol: float = 1e-12,
) -> Optional[Literal["top", "bottom"]]:
    """
    Determine which face is in compression from strain state.

    Sign convention: compression positive, tension negative.

    Args:
        eps_top: Strain at top fibre
        eps_bottom: Strain at bottom fibre
        strain_tol: Tolerance for considering strains as zero

    Returns:
        "top" if top face is more compressive
        "bottom" if bottom face is more compressive
        None if both faces are in tension (no compression face)
    """
    # If both in tension, no compression face
    if eps_top <= strain_tol and eps_bottom <= strain_tol:
        return None

    # Return the face with higher (more positive/compressive) strain
    return "top" if eps_top >= eps_bottom else "bottom"


def find_effective_depth_for_flexure(
    section: "RCSection",
    diagram: Optional["MNInteractionDiagram"],
    M_Ed: float,
    N_Ed: float,
    eps_top: Optional[float] = None,
    eps_bottom: Optional[float] = None,
    *,
    m_tol: float = 1e-6,
    strain_tol: float = 1e-15,
    warn_on_fallback: bool = True,
) -> float:
    """
    Effective depth d (mm) measured from the governing compression face.

    This is a shared implementation used by multiple check classes.

    If strains are provided, compression face is taken as the face with the larger
    (more positive) strain (compression is positive in this codebase).

    If strains are not provided:
    - If a diagram/solver is available and |M_Ed| is significant, strains are solved.
    - Otherwise, fallback returns min(d_top, d_bottom) for conservatism.

    Args:
        section: RCSection object
        diagram: MNInteractionDiagram for strain solving (optional)
        M_Ed: Design moment in kN·m
        N_Ed: Design axial force in kN
        eps_top: Pre-computed top strain (optional)
        eps_bottom: Pre-computed bottom strain (optional)
        m_tol: Tolerance for considering moment as zero
        strain_tol: Tolerance for strain comparisons
        warn_on_fallback: Whether to emit warnings when using fallback

    Returns:
        Effective depth d in mm
    """
    # Get effective depths for each compression face assumption
    d_top: Optional[float] = None
    d_bot: Optional[float] = None

    try:
        d_top = float(section.get_effective_depth(compression_face="top"))
    except ValueError:
        pass  # No rebar in bottom tension zone

    try:
        d_bot = float(section.get_effective_depth(compression_face="bottom"))
    except ValueError:
        pass  # No rebar in top tension zone

    # If neither worked, we have a problem
    if d_top is None and d_bot is None:
        raise ValueError("Cannot compute effective depth: no rebars found in either tension zone")

    # Helper to get conservative depth (handles one being None)
    def _get_conservative_d() -> float:
        if d_top is not None and d_bot is not None:
            return min(d_top, d_bot)
        elif d_top is not None:
            return d_top
        else:
            assert d_bot is not None
            return d_bot

    # Pure shear / pure axial / no clear bending => conservative depth
    if abs(M_Ed) <= m_tol:
        return _get_conservative_d()

    # If strains missing, try to solve if we have a diagram
    if (eps_top is None or eps_bottom is None) and diagram is not None:
        try:
            eps_top, eps_bottom = diagram.find_strains_for_MN(M_Ed, N_Ed)
        except Exception:
            eps_top, eps_bottom = None, None

    # Still missing -> fallback conservative
    if eps_top is None or eps_bottom is None:
        if warn_on_fallback:
            warnings.warn(
                "Effective depth fallback used (strain state unavailable). "
                "Returning conservative min(d_top, d_bottom).",
                stacklevel=3,
            )
        return _get_conservative_d()

    # If there is no compression anywhere, compression face is undefined -> fallback
    if eps_top <= strain_tol and eps_bottom <= strain_tol:
        if warn_on_fallback:
            warnings.warn(
                "Effective depth fallback used (both faces in tension; compression face undefined). "
                "Returning conservative min(d_top, d_bottom).",
                stacklevel=3,
            )
        return _get_conservative_d()

    # Otherwise: choose the more compressive face (bigger + strain)
    compression_face = "top" if eps_top >= eps_bottom else "bottom"

    if compression_face == "top":
        if d_top is not None:
            return d_top
        if warn_on_fallback:
            warnings.warn(
                "Effective depth fallback used (no rebar in tension zone for this compression face).",
                stacklevel=3,
            )
        return _get_conservative_d()
    else:
        if d_bot is not None:
            return d_bot
        if warn_on_fallback:
            warnings.warn(
                "Effective depth fallback used (no rebar in tension zone for this compression face).",
                stacklevel=3,
            )
        return _get_conservative_d()


def find_mean_effective_depth(
    section: "RCSection",
    tension_face: Literal["top", "bottom"] = "bottom",
    zone_fraction: float = 0.5,
) -> float:
    """
    Mean effective depth to tension reinforcement centroid.

    This returns the same as get_effective_depth but with clearer naming for
    serviceability checks like cracking.

    Args:
        section: RCSection object
        tension_face: Which face is in tension ("top" or "bottom")
        zone_fraction: Fraction of section depth considered as tension zone

    Returns:
        Mean effective depth in mm
    """
    # compression_face is opposite of tension_face
    compression_face: Literal["top", "bottom"] = "top" if tension_face == "bottom" else "bottom"
    return section.get_effective_depth(
        compression_face=compression_face,
        zone_fraction=zone_fraction,
    )


def find_equivalent_diameter(
    bars: List[Tuple[float, int]],
) -> float:
    """
    Equivalent bar diameter for mixed bar sizes (EC2 §7.3.4(3), Eq. 7.12).

    For a section with n₁ bars of diameter φ₁ and n₂ bars of diameter φ₂:
        φ_eq = (n₁·φ₁² + n₂·φ₂²) / (n₁·φ₁ + n₂·φ₂)

    This generalises to multiple bar sizes.

    Args:
        bars: List of tuples (diameter_mm, count) for each bar size

    Returns:
        Equivalent diameter in mm

    Example:
        >>> # 4 x 16mm bars and 2 x 20mm bars
        >>> phi_eq = find_equivalent_diameter([(16, 4), (20, 2)])
        >>> print(f"φ_eq = {phi_eq:.1f} mm")
    """
    if not bars:
        raise ValueError("No bars provided for equivalent diameter calculation")

    # Check for single bar size (or all same diameter)
    diameters = set(d for d, n in bars if n > 0)
    if len(diameters) == 1:
        return diameters.pop()

    numerator = 0.0
    denominator = 0.0

    for diameter, count in bars:
        if count <= 0:
            continue
        numerator += count * diameter ** 2
        denominator += count * diameter

    if denominator <= 0:
        raise ValueError("Total bar count is zero")

    return numerator / denominator


def get_tension_rebars_from_strain_state(
    section: "RCSection",
    eps_top: float,
    eps_bottom: float,
) -> List[Tuple[float, int, float]]:
    """
    Identify rebars in the tension zone based on strain state.

    Returns rebars where the strain at their location is negative (tension).

    Args:
        section: RCSection object
        eps_top: Strain at top fibre (compression positive)
        eps_bottom: Strain at bottom fibre (compression positive)

    Returns:
        List of tuples (diameter_mm, count, y_position) for each tension rebar group
    """
    bounds = section.outline.bounds
    h = bounds[3] - bounds[1]  # section height
    y_min = bounds[1]

    tension_rebars = []

    for group in section.rebar_groups:
        diameter = float(group.rebar.diameter)
        for pos in group.positions:
            # Calculate strain at this y position (linear interpolation)
            y_rel = (pos.y - y_min) / h  # normalised position from bottom (0 to 1)
            # strain = eps_bottom + (eps_top - eps_bottom) * y_rel
            strain_at_bar = eps_bottom + (eps_top - eps_bottom) * y_rel

            # Tension is negative strain
            if strain_at_bar < 0:
                tension_rebars.append((diameter, 1, pos.y))

    return tension_rebars


def calculate_rebar_stress_from_strain(
    strain: float,
    E_s: float,
    f_yk: float,
    k: float = 1.0,
    epsilon_uk: float = 0.05,
) -> float:
    """
    Calculate stress in rebar for a given strain (SLS - characteristic).

    Uses bilinear stress-strain with hardening for serviceability.

    Args:
        strain: Strain at rebar location (tension negative, compression positive)
        E_s: Steel elastic modulus in MPa
        f_yk: Characteristic yield strength in MPa
        k: Hardening ratio (f_t/f_y, typically 1.0 to 1.08)
        epsilon_uk: Ultimate strain (typically 0.05 for Class B)

    Returns:
        Stress in MPa (tension negative, compression positive)
    """
    # TODO I think this function may only apply if the SteelModelType
    # used in SteelStressStrainEC2 is INCLINED. SteelStressStrainEC2 is
    # passed when making an MNInteractionDiagram instance. As it uses k.
    # To investigate more.

    # Yield strain
    epsilon_yk = f_yk / E_s

    # Absolute strain for calculation
    abs_strain = abs(strain)

    if abs_strain <= epsilon_yk:
        # Elastic region
        stress = E_s * abs_strain
    elif abs_strain <= epsilon_uk:
        # Plastic region with hardening
        f_t = k * f_yk
        stress = f_yk + (f_t - f_yk) * (abs_strain - epsilon_yk) / (epsilon_uk - epsilon_yk)
    else:
        # Beyond ultimate - cap at ultimate stress
        stress = k * f_yk

    # Apply sign (tension negative, compression positive)
    return stress if strain >= 0 else -stress
