"""
Shared utility functions for flexure-related code checks (bending, cracking).

Contains geometry helpers and EC2 formula implementations used by both
BendingCheck and CrackingCheck.
"""

from typing import TYPE_CHECKING, List, Literal, Optional, Tuple

# Public type alias for the fallback policy
EffectiveDepthFallback = Literal["ratio_of_h", "centroid"]
import warnings

from materials.reinforced_concrete.constitutive import SteelModelType
from materials.reinforced_concrete.ndp import get_ndp_callable

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


def calculate_modular_ratio(E_s: float, E_cm: float) -> float:
    """
    Modular ratio alpha_e = E_s / E_cm.

    Args:
        E_s: Steel elastic modulus in MPa.
        E_cm: Concrete elastic modulus in MPa.

    Returns:
        Modular ratio (dimensionless).
    """
    if E_cm <= 0:
        raise ValueError("E_cm must be > 0 to compute modular ratio.")
    return E_s / E_cm


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

    Returns a compression face only when the section has a clear
    compression/tension split (one face positive, one negative).
    Returns None for net tension (both ≤ 0) and net compression (both ≥ 0),
    since effective depth is physically undefined in those cases.

    Args:
        eps_top: Strain at top fibre
        eps_bottom: Strain at bottom fibre
        strain_tol: Tolerance for considering strains as zero

    Returns:
        "top" if top face is in compression and bottom is in tension
        "bottom" if bottom face is in compression and top is in tension
        None if both faces have the same sign (no compression/tension split)
    """
    # Both in tension → no compression face
    if eps_top <= strain_tol and eps_bottom <= strain_tol:
        return None

    # Both in compression → no tension face, d is undefined
    if eps_top >= -strain_tol and eps_bottom >= -strain_tol:
        return None

    # Clear split: return the more compressive face
    return "top" if eps_top >= eps_bottom else "bottom"


def _get_section_height(section: "RCSection") -> float:
    """Section height from bounding box (mm)."""
    _, min_y, _, max_y = section.get_bounding_box()
    return max_y - min_y


def _try_centroid_depth(
    section: "RCSection",
    compression_face: Literal["top", "bottom"],
) -> Optional[float]:
    """Try to get effective depth for a given compression face, return None on failure."""
    try:
        return float(section.get_effective_depth(compression_face=compression_face))
    except ValueError:
        return None


def _apply_d_fallback(
    section: "RCSection",
    d_fallback: EffectiveDepthFallback,
    d_ratio: float,
    reason: str,
    warn_on_fallback: bool,
    _stacklevel: int,
) -> float:
    """
    Return a fallback effective depth when strain-based derivation is not possible.

    Triggered when the section is in net compression, net tension, pure axial,
    or when the strain solver fails — i.e. whenever there is no clear
    compression/tension split.

    Policies:
        "ratio_of_h": d = d_ratio * h  (default 0.9h). Always available.
        "centroid":   min(d_top, d_bot) from rebar centroids.
                      Falls back to ratio_of_h if rebar is missing on a face.
    """
    if d_fallback == "centroid":
        d_top = _try_centroid_depth(section, "top")
        d_bot = _try_centroid_depth(section, "bottom")
        if d_top is not None and d_bot is not None:
            d = min(d_top, d_bot)
        elif d_top is not None:
            d = d_top
        elif d_bot is not None:
            d = d_bot
        else:
            # No rebar on either face — ultimate fallback to ratio_of_h
            d = d_ratio * _get_section_height(section)
    else:
        # "ratio_of_h" (default)
        d = d_ratio * _get_section_height(section)

    if warn_on_fallback:
        warnings.warn(
            f"Effective depth fallback ({reason}): d = {d:.1f} mm",
            stacklevel=_stacklevel,
        )
    return d


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
    d_fallback: EffectiveDepthFallback = "ratio_of_h",
    d_ratio: float = 0.9,
    _stacklevel: int = 2,
) -> float:
    """
    Effective depth d (mm) measured from the governing compression face.

    This is the single source of truth used by ShearCheck, BendingCheck,
    MNInteractionDiagram, and CircularSectionCheck.

    When the strain state allows (clear compression/tension split), d is
    computed from rebar centroid geometry via ``section.get_effective_depth()``.

    When the strain state is ambiguous (net compression, net tension, pure
    axial, solver failure), the ``d_fallback`` policy is applied:
        - ``"ratio_of_h"``: d = d_ratio * h  (default 0.9h, always available)
        - ``"centroid"``:   min(d_top, d_bot) from rebar centroids, with
          ultimate fallback to ratio_of_h if rebar missing on a face.

    Args:
        section: RCSection object.
        diagram: MNInteractionDiagram for strain solving (optional).
        M_Ed: Design moment in kN·m.
        N_Ed: Design axial force in kN (compression positive).
        eps_top: Pre-computed top strain (optional).
        eps_bottom: Pre-computed bottom strain (optional).
        m_tol: Tolerance for considering moment as zero.
        strain_tol: Tolerance for strain comparisons.
        warn_on_fallback: Whether to emit warnings when using fallback.
        d_fallback: Fallback policy — ``"ratio_of_h"`` or ``"centroid"``.
        d_ratio: Ratio of section height h used for ``"ratio_of_h"`` policy.
        _stacklevel: Warning stacklevel (internal, for wrapper call depth).

    Returns:
        Effective depth d in mm.
    """
    sl = _stacklevel + 1  # adjust for this frame

    def _fallback(reason: str) -> float:
        return _apply_d_fallback(
            section=section,
            d_fallback=d_fallback,
            d_ratio=d_ratio,
            reason=reason,
            warn_on_fallback=warn_on_fallback,
            _stacklevel=sl + 1,
        )

    # Pure shear / pure axial / no clear bending → fallback
    if abs(M_Ed) <= m_tol:
        return _fallback("no bending (|M_Ed| ≈ 0)")

    # If strains missing, try to solve via diagram
    if (eps_top is None or eps_bottom is None) and diagram is not None:
        try:
            eps_top, eps_bottom = diagram.find_strains_for_MN(M_Ed, N_Ed)
        except Exception:
            eps_top, eps_bottom = None, None

    # Still missing → fallback
    if eps_top is None or eps_bottom is None:
        return _fallback("strain state unavailable")

    # Determine compression face from strains
    compression_face = calculate_compression_face_from_strains(
        eps_top, eps_bottom, strain_tol=strain_tol,
    )

    # No compression face (both in tension or both in compression) → fallback
    if compression_face is None:
        return _fallback("no compression/tension split")

    # Have a compression face — try to get d from rebar geometry
    d = _try_centroid_depth(section, compression_face)
    if d is not None:
        return d

    # Compression face known but no rebar in the corresponding tension zone → fallback
    return _fallback("no rebar in tension zone for this compression face")


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


def calculate_rebar_characteristic_stress_from_strain(
    strain: float,
    *,
    steel_model_type: SteelModelType = SteelModelType.INCLINED,
    E_s: float = 200_000,
    f_yk: float = 500,
    k: float = 1.0,
    epsilon_uk: float = 0.05,
) -> float:
    """
    Calculate stress in rebar for a given strain (SLS - characteristic).

    Uses bilinear stress-strain with hardening for serviceability.

    Args:
        strain: Strain at rebar location (tension negative, compression positive)
        steel_model_type: SteelModelType.INCLINED or SteelModelType.HORIZONTAL
        E_s: Steel elastic modulus in MPa
        f_yk: Characteristic yield strength in MPa
        k: Hardening ratio (f_t/f_y, typically 1.0 to 1.08) - only used if steel model type is 'inclined'
        epsilon_uk: Ultimate strain (typically 0.05 for Class B) - only used if steel model type is 'inclined'

    Returns:
        Stress in MPa (tension negative, compression positive)
    """
    # Yield strain
    epsilon_yk = f_yk / E_s

    # Absolute strain for calculation
    abs_strain = abs(strain)

    if abs_strain <= epsilon_yk:
        # Elastic region
        stress = E_s * abs_strain
    else:
        if steel_model_type is SteelModelType.HORIZONTAL:
            # perfectly plastic post-yield
            stress = f_yk

        elif steel_model_type is SteelModelType.INCLINED:
            # hardening up to k*f_yk at epsilon_uk
            if abs_strain <= epsilon_uk:
                # Plastic region with hardening
                f_t = k * f_yk
                stress = f_yk + (f_t - f_yk) * (abs_strain - epsilon_yk) / (epsilon_uk - epsilon_yk)
            else:
                # Beyond ultimate - cap at ultimate stress
                stress = k * f_yk
        else:
            raise ValueError(f"Unsupported steel model type: {steel_model_type!r}")

    # Apply sign (tension negative, compression positive)
    return stress if strain >= 0 else -stress


def find_area_of_steel_minimum(b: float, d: float, f_ctm: float, f_yk: float) -> float:
    '''
    Minimum area of longitudinal tension reinforcement in mm².
    Ref: EC2 §9.2.1.1(1) (9.1N)

    Args:
        b: Mean breadth of the section in the tensile zone
        d: Effective depth to tensile reinforcement
        f_ctm: The mean tensile strength of concrete
        f_yk: The characteristic yield strength of the rebar

    Returns:
        A_s,min: The minimum amount of longitudinal reinforcement allowed
    '''
    if b <= 0:
        raise ValueError(f"b must be > 0, got {b}")
    if d <= 0:
        raise ValueError(f"d must be > 0, got {d}")
    if f_ctm < 0:
        raise ValueError(f"f_ctm must be >= 0, got {f_ctm}")
    if f_yk <= 0:
        raise ValueError(f"f_yk must be > 0, got {f_yk}")

    ratio_fn = get_ndp_callable("as_min_flexural_ratio")
    ratio = float(ratio_fn(f_ctm=float(f_ctm), f_yk=float(f_yk)))
    return ratio * float(b) * float(d)


def find_area_of_steel_maximum(section_area: float) -> float:
    '''
    Maximum area of longitudinal tension or compression reinforcement in mm².
    Ref: EC2 §9.2.1.1(3)

    Args:
        section_area: The cross-sectional area of the section
    
    Returns:
        A_s,max: The maximum amount of longitudinal reinforcement allowed
    '''
    if section_area < 0:
        raise ValueError(f"section_area must be >= 0, got {section_area}")

    ratio_fn = get_ndp_callable("as_max_flexural_ratio")
    ratio = float(ratio_fn(section_area=float(section_area)))
    return ratio * float(section_area)
