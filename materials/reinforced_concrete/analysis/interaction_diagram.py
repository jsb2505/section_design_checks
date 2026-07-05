"""
M-N interaction diagram generator using fibre-based strain compatibility.

Implements EC2 ultimate limit state analysis for combined axial force and bending
about a single axis (major axis in this 2D implementation).

Key modelling choices / conventions
-----------------------------------
Sign convention (global):
- Axial force N > 0 => compression
- Axial force N < 0 => tension
- Strain: compression positive, tension negative (consistent with concrete convention)
- Concrete constitutive models expect compression strain > 0 and return compression stress > 0
- Steel constitutive models return stress with the same sign as strain (compression positive, tension negative)

Strain compatibility:
- Plane sections remain plane.
- A neutral axis depth (NA) is assumed from the compression face (top or bottom).
- The extreme compression fibre is assigned a maximum concrete compressive strain (default: ε_cu from concrete model),
  and all other strains follow by similar triangles.

Ultimate strain handling:
- The solver enforces the maximum concrete compressive strain via the assumed strain field.
- The concrete material model is domain-limited with a small tolerance clip at ε_cu to improve numerical robustness.

Tension stiffening (optional):
- When tension_stiffening=True, concrete in tension contributes post-cracking using a simplified EC2-style
  average tension stress-strain relationship.
- This is NOT a pure “ULS no-tension” model; enabling it can alter the envelope, especially near tension-controlled regions.

Closed envelope:
- The returned diagram is a closed loop: pure compression → (+M branch) → pure tension → (-M branch) → back to pure compression.
- Closing points (pure compression and pure tension) are computed via the SAME fibre integration used elsewhere for consistency.

Notes on confinement:
- If confined_concrete=True, a Mander-style confined concrete response is applied in compression.
- To avoid “double factoring”, confinement is computed at characteristic level (using f_ck and f_yk) then reduced to design level
  via the same factor used for unconfined concrete (alpha_cc / gamma_c), so the solver remains consistent with design strengths.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, NamedTuple, Sequence, Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from materials.reinforced_concrete.code_checks.ec2_2004.shear_utils import TensionShiftResult
    from materials.reinforced_concrete.materials.rebar import ShearRebar

import warnings
import csv
import json
import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field
from scipy.optimize import least_squares, leastsq

from materials.utils.helpers import as_float
from materials.reinforced_concrete.constitutive import (
    create_concrete_stress_strain,
    create_steel_stress_strain,
    SteelModelType,
    ConcreteModelType,
)
from materials.reinforced_concrete.geometry import FibreMesh, RCSection
from materials.reinforced_concrete.materials import ConcreteMaterial
from materials.core.units import ForceUnit, MomentUnit, to_kn, to_knm


# ------------------------
# Types / small utilities
# ------------------------

def _ray_segment_intersection_alpha(
    ray_dir: Tuple[float, float],
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    tol: float = 1e-12,
) -> Optional[float]:
    """
    Intersect ray from origin: R(t) = t * ray_dir, t >= 0
    with segment S(s) = p1 + s*(p2-p1), s in [0,1].

    Returns:
        alpha=t if intersection exists, else None.

    Uses 2D cross-product formulation for robustness.
    """
    rx, ry = ray_dir
    x1, y1 = p1
    x2, y2 = p2
    sx, sy = (x2 - x1, y2 - y1)

    denom = rx * sy - ry * sx  # cross(r, svec)
    if abs(denom) <= tol:
        return None  # parallel or nearly parallel

    # t = cross(p1, svec) / cross(r, svec)
    # s = cross(p1, r)    / cross(r, svec)
    t = (x1 * sy - y1 * sx) / denom
    s = (x1 * ry - y1 * rx) / denom

    if t >= -tol and (-tol <= s <= 1.0 + tol):
        return max(0.0, t)
    return None


# ----------------------------
# Output model
# ----------------------------

class InteractionPoint(BaseModel):
    """Single point on M-N interaction diagram."""
    model_config = ConfigDict(frozen=True)

    N: float = Field(..., description="Axial force in kN (positive = compression)")
    M: float = Field(..., description="Moment about section centroid in kN·m")
    neutral_axis_depth: float = Field(..., description="Neutral axis depth from compression face (mm)")
    compression_from_bottom: bool = Field(..., description="True if compression face is bottom, else top")

    max_concrete_strain: float = Field(..., description="Maximum concrete strain in this state (compression positive)")
    max_steel_strain: float = Field(..., description="Maximum absolute steel strain in this state")

    def __repr__(self) -> str:
        face = "bottom" if self.compression_from_bottom else "top"
        return (
            f"InteractionPoint(N={self.N:.1f} kN, M={self.M:.1f} kN·m, "
            f"NA={self.neutral_axis_depth:.2f} mm, comp={face})"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "N": self.N,
            "M": self.M,
            "neutral_axis_depth": self.neutral_axis_depth,
            "compression_from_bottom": self.compression_from_bottom,
            "max_concrete_strain": self.max_concrete_strain,
            "max_steel_strain": self.max_steel_strain,
        }


# ----------------------------
# Results Class
# ----------------------------

class CapacityResult(NamedTuple):
    N_Rd: Optional[float]
    M_Rd: Optional[float]
    is_safe: bool
    utilization: float
    details: Optional[dict] = None # Default value for the 5th item


# ----------------------------
# Main solver
# ----------------------------

class MNInteractionDiagram:
    """
    M-N interaction diagram generator using fibre-based strain compatibility (2D single-axis).

    Generates two branches:
    - Compression from TOP (typically +M)
    - Compression from BOTTOM (typically -M)

    Returns a closed envelope ordered:
        pure compression → top-compression branch → pure tension → bottom-compression branch → pure compression
    """

    def __init__(
        self,
        section: RCSection,
        concrete: ConcreteMaterial,
        concrete_model_type: ConcreteModelType = ConcreteModelType.PARABOLA_RECTANGLE,
        steel_model_type: SteelModelType = SteelModelType.INCLINED,
        n_fibres_width: int = 20,
        n_fibres_height: int = 30,
        tension_stiffening: bool = False,
        use_characteristic: bool = False,
        use_accidental: bool = False,
        confined_concrete: bool = False,
        confinement_rho_s: Optional[float] = None,
        confinement_f_yh: Optional[float] = None,
        ignore_compression_steel: bool = False,
        elastic_modulus: Optional[float] = None,
        include_tension: bool = False,
        concrete_model_override: Optional[Any] = None,
        steel_models_override: Optional[List[Any]] = None,
    ):
        self.section = section
        self.concrete = concrete

        self.tension_stiffening = tension_stiffening
        self.confined_concrete = confined_concrete
        self.ignore_compression_steel = ignore_compression_steel
        self.confinement_rho_s = confinement_rho_s
        self.elastic_modulus = elastic_modulus
        self.include_tension = include_tension

        # IMPORTANT: treat confinement_f_yh as CHARACTERISTIC if provided.
        # If None, default to the first longitudinal group's characteristic yield strength.
        self.confinement_f_yh = confinement_f_yh

        # Constitutive models for design-level capacity evaluation
        if concrete_model_override is not None:
            self.concrete_model = concrete_model_override
        else:
            self.concrete_model = create_concrete_stress_strain(
                concrete=concrete,
                model_type=concrete_model_type,
                use_characteristic=use_characteristic,
                use_accidental=use_accidental,
                elastic_modulus=elastic_modulus,
                include_tension=include_tension,
            )

        if steel_models_override is not None:
            if len(steel_models_override) == 0:
                raise ValueError("steel_models_override must contain at least one model")
            self.steel_models = list(steel_models_override)
        else:
            if len(section.rebar_groups) == 0:
                raise ValueError("Section must have at least one rebar group")

            # Steel models per group (support different grades)
            self.steel_models = [
                create_steel_stress_strain(
                    steel=g.rebar,
                    branch_type=steel_model_type,
                    use_characteristic=use_characteristic,
                    use_accidental=use_accidental
                )
                for g in section.rebar_groups
            ]

        # Check if concrete model already has EC2 3.1.9 confinement
        _model_has_ec2_confinement = getattr(self.concrete_model, 'is_ec2_confined', False)

        # Confined concrete parameter checks
        if self.confined_concrete:
            if concrete_model_override is not None:
                raise ValueError(
                    "Cannot use confined_concrete=True with concrete_model_override. "
                    "Apply confinement within your custom model instead."
                )
            # Prevent double confinement: Mander + EC2 3.1.9
            if _model_has_ec2_confinement:
                raise ValueError(
                    "Cannot use confined_concrete=True (Mander model) when the concrete stress-strain "
                    "model already has EC2 §3.1.9 confinement (sigma_2 > 0). Use one or the other."
                )

            if self.confinement_rho_s is None:
                raise ValueError("confinement_rho_s must be provided when confined_concrete=True")
            if not (0.0 < self.confinement_rho_s <= 0.1):
                raise ValueError(f"confinement_rho_s must be in (0, 0.1], got {self.confinement_rho_s}")

            if self.confinement_f_yh is None:
                # default to characteristic yield of first longitudinal group
                self.confinement_f_yh = section.rebar_groups[0].rebar.f_yk

            if self.confinement_f_yh <= 0:
                raise ValueError(f"confinement_f_yh must be > 0, got {self.confinement_f_yh}")

        # Fibre mesh
        self.mesh = FibreMesh(
            section=section,
            n_fibres_width=n_fibres_width,
            n_fibres_height=n_fibres_height,
            exclude_steel_area=True,
        )

        # Geometry references
        _, min_y, _, max_y = section.get_bounding_box()
        self.section_top = max_y
        self.section_bottom = min_y
        self.section_height = max_y - min_y

        if self.section_height <= 0:
            raise ValueError("Section height must be > 0")

        # Cache fibre arrays for performance (avoid repeated allocation/copy in residual/Jacobian)
        (
            self._fibre_x,
            self._fibre_y,
            self._fibre_area,
            self._fibre_mat,
            self._fibre_mi,
            self._fibre_i,
            self._fibre_j,
        ) = self.mesh.get_fibre_arrays()

        self._fibre_mat = self._fibre_mat.astype("U8", copy=False)  # Ensure consistent dtype

        # Cache section centroid (avoid repeated Shapely geometry access)
        _, self._section_cy = self.section.get_centroid()

        # Cache diagram points to avoid repeated generation
        self._dense_diagram_points: Optional[tuple[InteractionPoint, ...]] = None
        self._dense_diagram_n: int = 0
        self._diagram_points_cache: dict[int, tuple[InteractionPoint, ...]] = {}

        # Strain-solve result cache keyed by (M_target, N_target, strict).
        # Naturally invalidated when this diagram instance is rebuilt (new object created).
        # Not keyed on tol/initial_guess — tol is always the default in practice,
        # and initial_guess is an optimizer hint that should not alter the final result.
        self._strain_cache: dict[tuple, tuple] = {}


    # ----------------------------
    # Core mechanics
    # ----------------------------

    def calculate_section_forces(
        self,
        stresses: npt.NDArray[np.float64],
        *,
        use_section_centroid: bool = True,
    ) -> Tuple[float, float]:
        """
        Calculate resultant axial force and single-axis bending moment from fibre stresses.

        Args:
            stresses: Stress at each fibre (MPa = N/mm²), same order as mesh.get_fibre_arrays()
            use_section_centroid: If True, take moments about gross concrete centroid (section.get_centroid()).
                                 This matches the rest of the solver and plotting conventions.

        Returns:
            (N, M):
                N in kN (positive compression)
                M in kN·m about the section centroid, using y-offset (single-axis)
        """
        # Use cached fibre arrays for performance
        y = self._fibre_y
        area = self._fibre_area

        # Axial force: sum(σ * A) in N, convert to kN
        N = to_kn(np.sum(stresses * area), ForceUnit.N)

        # Moment about section centroid (single axis about y-offset)
        if use_section_centroid:
            cy = self._section_cy  # Use cached centroid
        else:
            cy = float(np.sum(y * area) / np.sum(area))

        y_offset = y - cy
        M = to_knm(np.sum(stresses * area * y_offset), MomentUnit.NMM)

        return (as_float(N), as_float(M))


    def _concrete_stress_with_options(
        self,
        concrete_strains: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """
        Compute concrete stresses for the given concrete strains (compression positive),
        applying optional confinement (compression only) and optional tension stiffening (tension only).
        """
        # Base (design-level) concrete stresses from constitutive model
        concrete_stresses = self.concrete_model.get_stress_array(concrete_strains)

        # -------------------------------------
        # Confined concrete (compression only)
        # -------------------------------------
        if self.confined_concrete:
            assert self.confinement_rho_s is not None
            assert self.confinement_f_yh is not None

            rho_s = float(self.confinement_rho_s)
            f_yh_k = float(self.confinement_f_yh)  # characteristic transverse steel yield

            comp_mask = concrete_strains > 0.0
            if np.any(comp_mask):
                # Compute confinement at characteristic level, then reduce to design consistently.
                f_co_k = float(self.concrete.f_ck)
                eps_co = float(self.concrete.epsilon_c2)

                k_e = 0.75
                f_co_k_safe = max(f_co_k, 1e-6)

                f_l_k = 0.5 * k_e * rho_s * f_yh_k

                term = 1.0 + 7.94 * f_l_k / f_co_k_safe
                term = max(term, 1e-12)

                f_cc_k = f_co_k * (
                    2.254 * np.sqrt(term) - 2.0 * f_l_k / f_co_k_safe - 1.254
                )

                f_ratio = max(f_cc_k / f_co_k_safe, 1e-6)
                eps_cc = eps_co * (1.0 + 5.0 * (f_ratio - 1.0))
                eps_cc = max(eps_cc, 1e-9)

                eps_cu_conf = 0.004 + 0.14 * rho_s * f_yh_k / f_co_k_safe

                # Reduce confined characteristic stresses to design level
                design_factor = float(self.concrete.alpha_cc) / float(self.concrete.gamma_c)

                E_cm = float(self.concrete.E_cm)
                denom = E_cm - (f_cc_k / eps_cc)
                if abs(denom) < 1e-9:
                    denom = 1e-9 if denom >= 0 else -1e-9
                r = E_cm / denom

                comp_str = concrete_strains[comp_mask]
                x = comp_str / eps_cc
                x_safe = np.maximum(x, 0.0)

                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    x_pow_r = np.where(x_safe > 0.0, x_safe**r, 0.0)
                    denom_m = (r - 1.0) + x_pow_r
                    denom_m = np.where(np.abs(denom_m) < 1e-12, 1e-12, denom_m)
                    f_conf_k = f_cc_k * x_safe * r / denom_m

                f_conf_k = np.where(comp_str <= eps_cu_conf, f_conf_k, 0.0)
                f_conf_d = design_factor * f_conf_k

                concrete_stresses[comp_mask] = f_conf_d

        # ----------------------------------
        # Tension stiffening (tension only)
        # ----------------------------------
        if self.tension_stiffening:
            ten_mask = concrete_strains < 0.0
            if np.any(ten_mask):
                f_ctm = float(self.concrete.f_ctm)
                E_cm = float(self.concrete.E_cm)
                eps_cr = f_ctm / max(E_cm, 1e-9)

                beta = 0.6  # short-term
                eps_t = -concrete_strains[ten_mask]  # tension magnitude

                sigma_t = np.where(
                    eps_t <= eps_cr,
                    -E_cm * eps_t,
                    -f_ctm * np.maximum(0.0, 1.0 - beta * np.subtract(eps_t, eps_cr) / (eps_cr * 5.0)),
                )
                concrete_stresses[ten_mask] = sigma_t

        return concrete_stresses

    def _concrete_tangent_modulus_with_options(
        self,
        concrete_strains: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """
        Compute concrete tangent modulus E_t = dσ/dε for given strains,
        accounting for optional tension stiffening.

        Note: Confined concrete tangent modulus NOT implemented - requires
        complex Mander model derivative. See docs/ANALYTICAL_JACOBIAN_ENHANCEMENTS.md

        Args:
            concrete_strains: Strain array (compression positive)

        Returns:
            Tangent modulus array in MPa
        """
        # Base tangent modulus from constitutive model (parabola-rectangle)
        E_t = self.concrete_model.get_tangent_modulus_array(concrete_strains)

        # ------------------
        # Tension stiffening tangent modulus (tension only)
        # ------------------
        # Overrides base model (which has E_t=0 in tension)
        if self.tension_stiffening:
            ten_mask = concrete_strains < 0.0
            if np.any(ten_mask):
                f_ctm = float(self.concrete.f_ctm)
                E_cm = float(self.concrete.E_cm)
                eps_cr = f_ctm / max(E_cm, 1e-9)

                beta = 0.6  # short-term loading
                eps_t = -concrete_strains[ten_mask]  # tension magnitude (positive)

                # Piecewise tangent modulus:
                # 1) Pre-cracking (ε ≤ ε_cr): σ = -E_cm * ε  →  E_t = E_cm
                # 2) Post-cracking (ε > ε_cr): σ = -f_ctm * [1 - β*(ε - ε_cr)/(5*ε_cr)]
                #    →  E_t = f_ctm * β / (5*ε_cr)  (linear decay slope)
                # 3) After cutoff: σ = 0  →  E_t = 0

                # Cutoff strain where tension contribution becomes zero
                eps_cutoff = eps_cr * (1.0 + 5.0 / beta)

                E_t_tension = np.where(
                    eps_t <= eps_cr,
                    E_cm,  # Pre-cracking: elastic, dσ/dε = E_cm > 0
                    np.where(
                        eps_t < eps_cutoff,
                        -f_ctm * beta / (5.0 * eps_cr),  # Post-cracking: softening, dσ/dε < 0
                        0.0,  # After cutoff: zero stiffness
                    ),
                )
                E_t[ten_mask] = E_t_tension

        # Note: Confined concrete tangent modulus would go here if implemented
        # For now, confined concrete uses numerical Jacobian (see Jacobian selection logic)

        return E_t


    def _strain_field_from_end_strains(
        self,
        eps_top: float,
        eps_bottom: float,
    ) -> npt.NDArray[np.float64]:
        """
        Plane-sections strain field defined by strains at the extreme top/bottom fibres
        (compression positive), linear over y.

        eps_top: strain at y = section_top
        eps_bottom: strain at y = section_bottom
        """
        # Use cached fibre y-coordinates and section geometry
        y = self._fibre_y

        y_bot = float(self.section_bottom)
        h = float(self.section_height)  # Use cached value

        # linear interpolation in y
        t = (y - y_bot) / h
        # Preserve complex type for complex-step differentiation
        return eps_bottom + (eps_top - eps_bottom) * t


    def calculate_point_from_end_strains(
        self,
        eps_top: float,
        eps_bottom: float,
    ) -> InteractionPoint:
        """
        Compute (N,M) from a strain profile defined by end strains.

        Notes:
        - neutral_axis_depth and compression_from_bottom become *derived metadata* only.
        - This method is globally valid across sign changes (no branch switching).
        """
        # Use cached fibre arrays for performance
        material_type = self._fibre_mat  # Already converted to U8 in __init__
        material_index = self._fibre_mi

        strains = self._strain_field_from_end_strains(eps_top=eps_top, eps_bottom=eps_bottom)
        stresses = np.zeros_like(strains)

        # Concrete
        conc_mask = material_type == "concrete"
        if np.any(conc_mask):
            stresses[conc_mask] = self._concrete_stress_with_options(strains[conc_mask])

        # Steel
        steel_mask = material_type == "steel"
        if np.any(steel_mask):
            steel_strains = strains[steel_mask]
            steel_indices = material_index[steel_mask]
            steel_stresses = np.zeros_like(steel_strains)
            for gi, sm in enumerate(self.steel_models):
                m = steel_indices == gi
                if np.any(m):
                    steel_stresses[m] = sm.get_stress_array(steel_strains[m])
            # Zero out compression steel if flag is set (positive strain = compression)
            if self.ignore_compression_steel:
                steel_stresses[steel_strains > 0] = 0.0
            stresses[steel_mask] = steel_stresses

        N, M = self.calculate_section_forces(stresses)

        # Derived NA (optional metadata): where strain crosses 0
        y_top = float(self.section_top)
        y_bot = float(self.section_bottom)
        h = float(self.section_height)

        # For complex-step: use real part for comparisons
        eps_top_real = np.real(eps_top) if np.iscomplexobj(eps_top) else eps_top
        eps_bottom_real = np.real(eps_bottom) if np.iscomplexobj(eps_bottom) else eps_bottom

        # If eps_top == eps_bottom => no curvature => NA "at infinity"
        if abs(eps_top_real - eps_bottom_real) < 1e-18:
            na_depth = (1e6 * h) if eps_top_real > 0 else (-1e6 * h)
            comp_from_bottom = False
        else:
            # Solve eps(y)=0 for y in [y_bot, y_top]
            # eps(y)=eps_bottom + (eps_top-eps_bottom)*(y-y_bot)/h
            # Use real part for y0 calculation (geometry should be real)
            y0 = y_bot - h * (eps_bottom_real / (eps_top_real - eps_bottom_real))
            # "compression face" for metadata: side with larger strain (more compression)
            comp_from_bottom = (eps_bottom_real > eps_top_real)

            # Convert to "NA depth from compression face" only if inside height
            if y_bot - 1e-9 <= y0 <= y_top + 1e-9:
                if comp_from_bottom:
                    na_depth = float(y0 - y_bot)
                else:
                    na_depth = float(y_top - y0)
            else:
                # NA outside section => very small/large surrogate depth
                # sign indicates which side (outside) in a consistent way
                na_depth = 1e6 * h if (eps_top_real > 0 or eps_bottom_real > 0) else -1e6 * h

        # Extract real parts for metadata (strains might be complex during differentiation)
        max_conc = float(np.real(np.max(strains[conc_mask]))) if np.any(conc_mask) else 0.0
        max_steel = float(np.real(np.max(np.abs(strains[steel_mask])))) if np.any(steel_mask) else 0.0

        return InteractionPoint(
            N=as_float(N),
            M=as_float(M),
            neutral_axis_depth=float(na_depth),
            compression_from_bottom=bool(comp_from_bottom),
            max_concrete_strain=max_conc,
            max_steel_strain=max_steel,
        )


    def get_fibre_forces_from_end_strains(
        self,
        eps_top: float,
        eps_bottom: float,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Compute fibre-level forces from strain profile (public helper for external tools).

        This is a PUBLIC interface for computing detailed force distributions, intended
        for use by code checks and other analyses that need fibre-level data without
        accessing private internals.

        Args:
            eps_top: Strain at top fibre (compression positive)
            eps_bottom: Strain at bottom fibre (compression positive)

        Returns:
            Tuple of (forces, y_coords, areas):
                - forces: Force in each fibre (N), compression positive
                - y_coords: Y-coordinate of each fibre (mm)
                - areas: Area of each fibre (mm²)

        Example:
            >>> diagram = MNInteractionDiagram(section, concrete)
            >>> eps_top, eps_bottom = diagram.find_strains_for_MN(M=50.0, N=100.0)
            >>> forces, y_coords, areas = diagram.get_fibre_forces_from_end_strains(eps_top, eps_bottom)
            >>> # Compute tension/compression centroids for lever arm
            >>> tension_mask = forces < 0
            >>> y_T = np.sum(-forces[tension_mask] * y_coords[tension_mask]) / np.sum(-forces[tension_mask])
        """
        # Use cached fibre arrays for performance
        y = self._fibre_y
        area = self._fibre_area
        material_type = self._fibre_mat  # Already converted to U8 in __init__
        material_index = self._fibre_mi

        # Compute strains at all fibres
        strains = self._strain_field_from_end_strains(eps_top=eps_top, eps_bottom=eps_bottom)

        # Compute stresses
        stresses = np.zeros_like(strains)

        # Concrete fibres - use internal method that handles confinement/tension stiffening
        conc_mask = material_type == "concrete"
        if np.any(conc_mask):
            stresses[conc_mask] = self._concrete_stress_with_options(strains[conc_mask])

        # Steel fibres
        steel_mask = material_type == "steel"
        if np.any(steel_mask):
            steel_strains = strains[steel_mask]
            steel_indices = material_index[steel_mask]
            steel_stresses = np.zeros_like(steel_strains)
            for gi, sm in enumerate(self.steel_models):
                m = steel_indices == gi
                if np.any(m):
                    steel_stresses[m] = sm.get_stress_array(steel_strains[m])
            # Zero out compression steel if flag is set (positive strain = compression)
            if self.ignore_compression_steel:
                steel_stresses[steel_strains > 0] = 0.0
            stresses[steel_mask] = steel_stresses

        # Forces per fibre (compression positive): Force = stress × area
        forces = stresses * area

        return (forces, y, area)


    @staticmethod
    def _cosine_space(n: int) -> np.ndarray:
        """
        Monotonic parameter in [0,1] clustered at BOTH ends.
        Great for capturing curvature near corners without blowing up point count.
        """
        if n <= 1:
            return np.array([0.0])
        t = np.linspace(0.0, 1.0, n)
        return 0.5 * np.subtract(1.0, np.cos(np.pi * t))


    @staticmethod
    def _interp(a: float, b: float, s: np.ndarray) -> np.ndarray:
        """Linear interpolation a -> b using parameter s in [0,1]."""
        return a + (b - a) * s


    @staticmethod
    def _dedupe_pairs(
        pairs: List[Tuple[float, float]],
        tol: float = 1e-12,
    ) -> List[Tuple[float, float]]:
        """Remove consecutive near-duplicate strain pairs."""
        if not pairs:
            return pairs
        out = [pairs[0]]
        for p in pairs[1:]:
            if (abs(p[0] - out[-1][0]) > tol) or (abs(p[1] - out[-1][1]) > tol):
                out.append(p)
        return out


    @staticmethod
    def _resample_closed_polyline_by_chord(
        points: Sequence[InteractionPoint],
        n_out: int,
    ) -> List[InteractionPoint]:
        """
        Resample a CLOSED polyline (last point equals first) to n_out points, approximately
        uniform spacing in (M,N) chord length.

        Uses normalization so M and N contribute similarly to distance.
        Keeps existing InteractionPoint objects (metadata preserved, but not recomputed).
        """
        if n_out < 3:
            raise ValueError("n_out must be >= 3")
        
        pts = list(points)

        if len(pts) < 4:
            return pts

        # Ensure closed
        if (pts[0].M != pts[-1].M) or (pts[0].N != pts[-1].N):
            pts.append(pts[0])

        M = np.array([p.M for p in pts], dtype=float)
        N = np.array([p.N for p in pts], dtype=float)

        # Normalize distance so one axis doesn't dominate
        m_rng = float(np.ptp(M)) or 1.0
        n_rng = float(np.ptp(N)) or 1.0
        Mn = M / m_rng
        Nn = N / n_rng

        d = np.sqrt(np.diff(Mn)**2 + np.diff(Nn)**2)
        s = np.concatenate(([0.0], np.cumsum(d)))
        total = float(s[-1])
        if total <= 0.0:
            # all points identical
            return [pts[0]] * n_out

        # Target arc-length stations (include closure)
        s_target = np.linspace(0.0, total, n_out)

        # For each target station, find the segment index
        idx = np.searchsorted(s, s_target, side="right") - 1
        idx = np.clip(idx, 0, len(pts) - 2)

        # Interpolate (M, N) to create geometrically accurate points
        # Metadata (neutral axis, etc.) comes from the nearest dense point for approximation
        out: List[InteractionPoint] = []
        for st, i in zip(s_target, idx):
            s0, s1 = s[i], s[i + 1]
            if s1 <= s0:
                out.append(pts[i])
                continue
            t = (st - s0) / (s1 - s0)

            # Linearly interpolate M and N for geometric accuracy
            M_interp = M[i] + t * (M[i + 1] - M[i])
            N_interp = N[i] + t * (N[i + 1] - N[i])

            # Use metadata from closer endpoint (neutral axis depth, etc.)
            # This is an approximation, but sufficient for capacity checks
            source_pt = pts[i] if t < 0.5 else pts[i + 1]

            # Create new InteractionPoint with interpolated (M, N) but approximate metadata
            out.append(InteractionPoint(
                N=float(N_interp),
                M=float(M_interp),
                neutral_axis_depth=source_pt.neutral_axis_depth,
                compression_from_bottom=source_pt.compression_from_bottom,
                max_concrete_strain=source_pt.max_concrete_strain,
                max_steel_strain=source_pt.max_steel_strain,
            ))

        # Ensure closed explicitly (common expectation for your envelope)
        if (out[0].M != out[-1].M) or (out[0].N != out[-1].N):
            out[-1] = out[0]

        # Remove accidental consecutive duplicates
        cleaned = [out[0]]
        for p in out[1:]:
            if (p.M != cleaned[-1].M) or (p.N != cleaned[-1].N):
                cleaned.append(p)

        # If we lost points to de-dupe, pad by repeating last (rare)
        while len(cleaned) < n_out:
            cleaned.append(cleaned[-1])

        return cleaned[:n_out]


    @staticmethod
    def _pin_extremal_points(
        resampled: List[InteractionPoint],
        dense: tuple[InteractionPoint, ...],
    ) -> List[InteractionPoint]:
        """
        Ensure min N, max M, and min M from the dense diagram appear exactly in the
        resampled output by replacing the nearest unused resampled neighbour.

        max N is already at resampled[0] (s=0 in the chord-length resampler) and is
        left untouched.  The other three extrema replace their nearest resampled
        neighbour in normalised (M, N) space.

        The closed-loop invariant (first == last) is maintained by protecting both
        index 0 and index n-1 from replacement.
        """
        n = len(resampled)
        if n < 4:
            return resampled

        dense_M = np.array([p.M for p in dense], dtype=float)
        dense_N = np.array([p.N for p in dense], dtype=float)

        idx_min_N = int(np.argmin(dense_N))
        idx_max_M = int(np.argmax(dense_M))
        idx_min_M = int(np.argmin(dense_M))

        # Normalisation so M and N contribute equally to distance
        m_rng = float(np.ptp(dense_M)) or 1.0
        n_rng = float(np.ptp(dense_N)) or 1.0

        res_Mn = np.array([p.M for p in resampled], dtype=float) / m_rng
        res_Nn = np.array([p.N for p in resampled], dtype=float) / n_rng

        # Index 0 and n-1 are protected (max N at index 0, closure at index n-1)
        used: set[int] = {0, n - 1}

        def _replace_nearest(dense_pt: InteractionPoint) -> None:
            target_Mn = dense_pt.M / m_rng
            target_Nn = dense_pt.N / n_rng
            dist2 = (res_Mn - target_Mn) ** 2 + (res_Nn - target_Nn) ** 2
            for i in np.argsort(dist2):
                ii = int(i)
                if ii not in used:
                    resampled[ii] = dense_pt
                    used.add(ii)
                    return

        _replace_nearest(dense[idx_min_N])
        _replace_nearest(dense[idx_max_M])
        _replace_nearest(dense[idx_min_M])

        return resampled


    def _strain_limit_loop(
        self,
        n_points: int,
        eps_cu: float,
        eps_t: float,
    ) -> List[Tuple[float, float]]:
        """
        Build a CLOSED loop in (eps_top, eps_bottom) that targets the interaction envelope
        efficiently (avoids wasting points on redundant 'both-faces-tension' states).

        Convention:
        - concrete compression strain is positive
        - tension is negative
        - eps_t is a POSITIVE magnitude (we use -eps_t for tensile strain)

        Loop idea (4 segments):
        A) Pure compression -> top fixed at +eps_cu, bottom goes +eps_cu -> -eps_t
        B) Approach pure tension with bottom fixed at -eps_t, top goes +eps_cu -> -eps_t
            (keep this SHORT; it mostly collapses to the pure tension point)
        C) Mirror of B: top fixed at -eps_t, bottom goes -eps_t -> +eps_cu
            (also SHORT)
        D) Return: bottom fixed at +eps_cu, top goes -eps_t -> +eps_cu

        This hits the “useful” boundary states while keeping sampling where (N,M) changes rapidly.
        """
        n_points = int(max(n_points, 40))
        eps_cu = float(eps_cu)
        eps_t = float(abs(eps_t))  # magnitude
        eps_ten = -eps_t

        # Allocate points by "importance":
        # Most curvature/variation tends to be on the two big bending edges A and D.
        # The two short closure edges (near pure tension) get far fewer points.
        nA = int(round(0.42 * n_points))
        nD = int(round(0.42 * n_points))
        nB = int(round(0.08 * n_points))
        nC = n_points - (nA + nB + nD)
        nC = max(nC, 4)
        nB = max(nB, 4)

        # Cluster at ends for each segment to capture the knees
        sA = self._cosine_space(nA)
        sB = self._cosine_space(nB)
        sC = self._cosine_space(nC)
        sD = self._cosine_space(nD)

        pairs: List[Tuple[float, float]] = []

        # Segment A: top = +eps_cu, bottom: +eps_cu -> -eps_t
        bot_A = self._interp(eps_cu, eps_ten, sA)
        for b in bot_A[:-1]:  # endpoint handled by next segment
            pairs.append((eps_cu, float(b)))

        # Segment B: bottom = -eps_t, top: +eps_cu -> -eps_t  (short)
        top_B = self._interp(eps_cu, eps_ten, sB)
        for t in top_B[:-1]:
            pairs.append((float(t), eps_ten))

        # Segment C: top = -eps_t, bottom: -eps_t -> +eps_cu  (short)
        bot_C = self._interp(eps_ten, eps_cu, sC)
        for b in bot_C[:-1]:
            pairs.append((eps_ten, float(b)))

        # Segment D: bottom = +eps_cu, top: -eps_t -> +eps_cu
        top_D = self._interp(eps_ten, eps_cu, sD)
        for t in top_D:
            pairs.append((float(t), eps_cu))

        # Ensure closed loop (start point repeated at end)
        if pairs[0] != pairs[-1]:
            pairs.append(pairs[0])

        # Remove any consecutive duplicates (floating noise / shared endpoints)
        pairs = self._dedupe_pairs(pairs, tol=1e-15)

        return pairs


    def _eps_tension_limit(self) -> float:
        """
        Choose a tensile strain magnitude for the strain-rectangle corner.

        - If any steel model has finite ultimate strain (inclined), use max(ε_ud)
        so all groups are within the model’s intended range (clipping does the rest).
        - If all are horizontal (infinite), use a large multiple of yield strain.
        """
        ultimates = [float(sm.get_ultimate_strain()) for sm in self.steel_models]
        finite = [u for u in ultimates if np.isfinite(u)]
        if finite:
            return float(max(finite))

        eps_y_max = max(float(sm.epsilon_y) for sm in self.steel_models)
        return float(max(10.0 * eps_y_max, 0.01))


    # -----------------------------------------
    # Inverse solver (M, N) → (ε_top, ε_bottom)
    # -----------------------------------------

    def find_strains_for_MN(
        self,
        M_target: float,
        N_target: float,
        initial_guess: Optional[Tuple[float, float]] = None,
        tol: float = 1e-6,
        strict: bool = False,
    ) -> Tuple[float, float]:
        """
        Inverse solver: Find end strains that produce target (M, N).

        Uses scipy.optimize.least_squares to solve:
            calculate_point_from_end_strains(eps_top, eps_bottom) = (N_target, M_target)

        This method does NOT require generate_diagram_points() to have been called - it only
        needs the fibre mesh and constitutive models (created in __init__).

        Args:
            M_target: Target moment (kN.m)
            N_target: Target axial force (kN, compression positive)
            initial_guess: Optional (eps_top, eps_bottom) starting point for optimizer.
                          If None, automatically estimated from (M, N) quadrant.
            tol: Convergence tolerance for residual norm and parameter changes
            strict:
                If False (default), return the nearest feasible strain state when exact
                equilibrium is not achievable (e.g., target outside M-N envelope).
                If True, require residuals to meet tolerance and raise ValueError otherwise.

        Returns:
            (eps_top, eps_bottom): Tuple of end strains that produce target forces

        Raises:
            ValueError: If no numerically stable solution can be found, or if strict=True
                and the target cannot be matched within tolerance.

        Performance:
            - Typical solve time: 10-50ms per unique (M,N) point
            - Hard points (e.g. near cracking transitions) may run extra fallback
              attempts with alternative starting points and Jacobians
            - Results are cached by (M_target, N_target, strict) per diagram instance.
              Repeated calls with the same load case return immediately from cache.

        Examples:
            >>> diagram = MNInteractionDiagram(section, concrete)
            >>> eps_top, eps_bottom = diagram.find_strains_for_MN(M_target=50.0, N_target=100.0)
            >>> point = diagram.calculate_point_from_end_strains(eps_top, eps_bottom)
            >>> assert abs(point.M - 50.0) < 1e-3
            >>> assert abs(point.N - 100.0) < 1e-3
        """
        _cache_key = (M_target, N_target, strict)
        _cached = self._strain_cache.get(_cache_key)
        if _cached is not None:
            return _cached

        def residual(eps_pair: npt.NDArray) -> npt.NDArray:
            """Residual function: [N_error, M_error]."""
            point = self.calculate_point_from_end_strains(eps_pair[0], eps_pair[1])
            return np.array([point.N - N_target, point.M - M_target])

        # Estimate initial guess if not provided
        if initial_guess is None:
            initial_guess = self._estimate_initial_strains(M_target, N_target)

        # Define strain bounds to prevent solver wandering into absurd strain space
        # This prevents numerical artifacts where extreme strains "match" forces via clipping
        eps_cu = self.concrete_model.get_ultimate_strain()  # Compression limit (~0.0035)
        eps_t = self._eps_tension_limit()  # Tension limit (steel-controlled, ~0.01-0.05)
        eps_y = self.steel_models[0].epsilon_y if self.steel_models else 0.002

        # Bounds: (eps_top, eps_bottom) each in [-eps_t, +eps_cu]
        lower_bounds = np.array([-eps_t, -eps_t])  # Maximum tension (negative)
        upper_bounds = np.array([+eps_cu, +eps_cu])  # Maximum compression (positive)

        # Choose Jacobian method based on material model complexity
        # CRITICAL: Analytical Jacobian requires tangent modulus matching forward model
        #
        # Tension stiffening: NOW SUPPORTED via _concrete_tangent_modulus_with_options
        # Confined concrete: NOT SUPPORTED - requires complex Mander derivative
        #                    (see docs/ANALYTICAL_JACOBIAN_ENHANCEMENTS.md)
        jac_method: Union[Callable[[npt.NDArray], npt.NDArray], str]
        if self.confined_concrete:
            # Use numerical Jacobian for confined concrete (Mander model derivative not implemented)
            jac_method = '2-point'
            max_iterations = 200  # May need more iterations with numerical gradients
        else:
            # Use analytical Jacobian for plain concrete + tension stiffening (3-10x faster)
            def analytical_jacobian(eps_pair: npt.NDArray) -> npt.NDArray:
                """Compute analytical Jacobian at current strain pair."""
                return self._compute_analytical_jacobian(eps_pair[0], eps_pair[1])

            jac_method = analytical_jacobian
            max_iterations = 50  # Analytical Jacobian converges much faster

        def clamp_guess(guess: Tuple[float, float]) -> Tuple[float, float]:
            """Clamp guess to strain bounds before handing it to the optimizer."""
            return (
                float(np.clip(guess[0], -eps_t, +eps_cu)),
                float(np.clip(guess[1], -eps_t, +eps_cu)),
            )

        def residual_metrics(result: Any) -> Tuple[float, float]:
            """
            Return (max_abs_residual, normalized_residual_norm).

            The normalized score scales each residual by target magnitude so
            comparisons remain meaningful across low/high load levels.
            """
            fun = np.asarray(getattr(result, "fun", np.array([np.inf, np.inf])), dtype=float)
            if fun.shape != (2,) or np.any(~np.isfinite(fun)):
                return (np.inf, np.inf)

            n_err = float(fun[0])
            m_err = float(fun[1])
            abs_max = max(abs(n_err), abs(m_err))

            n_scale = max(abs(float(N_target)), 1.0)
            m_scale = max(abs(float(M_target)), 1.0)
            norm = float(np.hypot(n_err / n_scale, m_err / m_scale))
            return (abs_max, norm)

        def solve_from_guess(
            guess: Tuple[float, float],
            jac: Union[Callable[[npt.NDArray], npt.NDArray], str],
        ) -> Any:
            # Analytical Jacobian: exact derivatives, 5-10 iterations typical
            # Numerical Jacobian: finite difference, 30-50 iterations typical
            x0 = np.asarray(clamp_guess(guess), dtype=float)
            try:
                return least_squares(
                    residual,
                    x0=x0,
                    bounds=(lower_bounds, upper_bounds),
                    jac=jac,  # type: ignore[arg-type]
                    ftol=tol,
                    xtol=tol,
                    gtol=tol,
                    max_nfev=max_iterations,
                )
            except ValueError as exc:
                # SciPy/Numpy compatibility issue observed under coverage:
                # "_CopyMode.IF_NEEDED is neither True nor False."
                if "_CopyMode.IF_NEEDED" not in str(exc):
                    raise

                jacobian = jac if callable(jac) else None
                x_out, _cov_x, info, message, ier = leastsq(  # type: ignore[misc]
                    func=residual,
                    x0=x0,
                    Dfun=jacobian,
                    full_output=True,
                    maxfev=max_iterations,
                )
                x_clamped = np.clip(np.asarray(x_out, dtype=float), lower_bounds, upper_bounds)
                fun = np.asarray(residual(x_clamped), dtype=float)
                return SimpleNamespace(
                    x=x_clamped,
                    fun=fun,
                    success=bool(ier in (1, 2, 3, 4) and np.all(np.isfinite(fun))),
                    status=int(ier),
                    message=message,
                    nfev=int(info.get("nfev", 0)) if isinstance(info, dict) else 0,
                )

        # Build a compact set of branch-diverse candidate guesses. Keep the
        # existing heuristic as first choice, then add conservative alternatives.
        candidate_guesses: List[Tuple[float, float]] = [initial_guess]
        if abs(M_target) < 1e-9:
            if N_target > 0:
                candidate_guesses.extend([
                    (+eps_cu * 0.8, +eps_cu * 0.8),
                    (+eps_cu * 0.9, +eps_cu * 0.7),
                ])
            elif N_target < 0:
                candidate_guesses.extend([
                    (-eps_y * 2.0, -eps_y * 2.0),
                    (-eps_y * 3.0, -eps_y),
                ])
            else:
                candidate_guesses.append((0.0, 0.0))
        elif M_target > 0:
            candidate_guesses.extend([
                (+eps_cu * 0.8, +eps_cu * 0.2),
                (+eps_cu * 0.8, -eps_y * 1.5),
                (+eps_cu * 0.6, -eps_y * 0.5),
            ])
            if N_target > 0:
                eccentricity_mm = abs(M_target) * 1000.0 / max(abs(N_target), 1e-6)
                if eccentricity_mm > float(self.section_height) * 0.6:
                    candidate_guesses.append((+eps_cu * 0.7, -eps_y * 2.0))
        else:
            candidate_guesses.extend([
                (+eps_cu * 0.2, +eps_cu * 0.8),
                (-eps_y * 1.5, +eps_cu * 0.8),
                (-eps_y * 0.5, +eps_cu * 0.6),
            ])
            if N_target > 0:
                eccentricity_mm = abs(M_target) * 1000.0 / max(abs(N_target), 1e-6)
                if eccentricity_mm > float(self.section_height) * 0.6:
                    candidate_guesses.append((-eps_y * 2.0, +eps_cu * 0.7))

        # Deduplicate after clamping to avoid redundant solve calls.
        deduped_guesses: List[Tuple[float, float]] = []
        for guess in candidate_guesses:
            clamped = clamp_guess(guess)
            if not any(abs(clamped[0] - d[0]) < 1e-9 and abs(clamped[1] - d[1]) < 1e-9 for d in deduped_guesses):
                deduped_guesses.append(clamped)

        # Keep existing behavior for outside-envelope requests: if residual stays
        # high, still return the nearest feasible point. This threshold is only
        # used to decide if additional fallback passes are necessary.
        acceptable_abs_error = max(1.0, tol * 1e6)

        attempts: List[Tuple[Any, str, Tuple[float, float]]] = []

        # Fast path: primary guess only.
        primary_guess = deduped_guesses[0]
        primary_result = solve_from_guess(primary_guess, jac_method)
        attempts.append((primary_result, "pass1_primary", primary_guess))

        best_result = primary_result
        best_abs_error, _ = residual_metrics(best_result)
        if np.isfinite(best_abs_error) and best_abs_error <= acceptable_abs_error:
            _result = tuple(best_result.x)
            self._strain_cache[_cache_key] = _result
            return _result

        # Fallback pass 1: alternative guesses with preferred Jacobian.
        for i, guess in enumerate(deduped_guesses[1:], start=1):
            attempts.append((solve_from_guess(guess, jac_method), f"pass1_guess{i}", guess))

        best_result, _, _ = min(
            attempts,
            key=lambda item: (
                residual_metrics(item[0])[0],
                residual_metrics(item[0])[1],
                0 if bool(getattr(item[0], "success", False)) else 1,
            ),
        )

        best_abs_error, _ = residual_metrics(best_result)
        if np.isfinite(best_abs_error) and best_abs_error <= acceptable_abs_error:
            _result = tuple(best_result.x)
            self._strain_cache[_cache_key] = _result
            return _result

        # Near cracking transitions, retry all guesses with numerical Jacobian.
        if self.tension_stiffening and not self.confined_concrete:
            for i, guess in enumerate(deduped_guesses):
                attempts.append((solve_from_guess(guess, "2-point"), f"pass2_guess{i}", guess))

            best_result, _, _ = min(
                attempts,
                key=lambda item: (
                    residual_metrics(item[0])[0],
                    residual_metrics(item[0])[1],
                    0 if bool(getattr(item[0], "success", False)) else 1,
                ),
            )

        best_abs_error, _ = residual_metrics(best_result)
        finite_x = np.all(np.isfinite(np.asarray(best_result.x)))
        if not (np.isfinite(best_abs_error) and finite_x):
            raise ValueError(
                f"Inverse solver failed for M={M_target:.2f} kN.m, N={N_target:.2f} kN. "
                "All solver attempts were numerically unstable."
            )

        if strict and best_abs_error > acceptable_abs_error:
            fun = np.asarray(getattr(best_result, "fun", np.array([np.nan, np.nan])), dtype=float)
            n_err = float(fun[0]) if fun.shape == (2,) else float("nan")
            m_err = float(fun[1]) if fun.shape == (2,) else float("nan")
            raise ValueError(
                f"Inverse solver could not match M={M_target:.2f} kN.m, N={N_target:.2f} kN "
                f"within tolerance. Best residuals: dN={n_err:.3f} kN, dM={m_err:.3f} kN.m. "
                "Target may be outside section capacity envelope. "
                "Use strict=False to return the nearest feasible strain state."
            )

        _result = tuple(best_result.x)
        self._strain_cache[_cache_key] = _result
        return _result

    def _compute_analytical_jacobian(
        self,
        eps_top: float,
        eps_bottom: float
    ) -> np.ndarray:
        """
        Compute analytical Jacobian matrix for inverse M-N solver.

        Jacobian J is 2×2:
            J = [[∂N/∂eps_top,    ∂N/∂eps_bottom],
                 [∂M/∂eps_top,    ∂M/∂eps_bottom]]

        Derivation:
            For each fibre at height y:
                strain(y) = eps_bottom + (eps_top - eps_bottom) * (y - y_bot) / h

            Define:
                α(y) = (y - y_bot) / h     (linear interpolation weight)
                β(y) = 1 - α(y)            (complementary weight)

            Then:
                ∂strain/∂eps_top = α(y)
                ∂strain/∂eps_bottom = β(y)

            Force contribution from fibre i:
                F_i = σ_i * A_i = σ(ε_i) * A_i

            Derivative:
                ∂F_i/∂eps_top = (dσ/dε)|_i * A_i * α(y_i)     [E_t * A * α]
                ∂F_i/∂eps_bottom = (dσ/dε)|_i * A_i * β(y_i)  [E_t * A * β]

            Axial force:
                N = Σ F_i  →  ∂N/∂eps = Σ ∂F_i/∂eps

            Moment (about centroid c_y):
                M = Σ F_i * (y_i - c_y)  →  ∂M/∂eps = Σ [∂F_i/∂eps * (y_i - c_y)]

        Args:
            eps_top: Top fibre strain (compression positive)
            eps_bottom: Bottom fibre strain (compression positive)

        Returns:
            2×2 Jacobian matrix [[dN_deps_top, dN_deps_bottom],
                                 [dM_deps_top, dM_deps_bottom]]
        """
        # Use cached fibre arrays for performance
        y_coords = self._fibre_y
        areas = self._fibre_area
        material_type = self._fibre_mat  # Already converted to U8 in __init__
        material_index = self._fibre_mi

        y_bot = float(self.section_bottom)
        h = float(self.section_height)

        # Compute strain at each fibre
        strains = eps_bottom + (eps_top - eps_bottom) * (y_coords - y_bot) / h

        # Compute tangent modulus E_t = dσ/dε at each fibre
        E_t = np.zeros_like(strains)

        # Concrete fibres - use method with tension stiffening support
        conc_mask = material_type == "concrete"
        if np.any(conc_mask):
            E_t[conc_mask] = self._concrete_tangent_modulus_with_options(strains[conc_mask])

        # Steel fibres
        steel_mask = material_type == "steel"
        if np.any(steel_mask):
            steel_strains = strains[steel_mask]
            steel_indices = material_index[steel_mask]
            steel_E_t = np.zeros_like(steel_strains)
            for gi, sm in enumerate(self.steel_models):
                m = steel_indices == gi
                if np.any(m):
                    steel_E_t[m] = sm.get_tangent_modulus_array(steel_strains[m])
            E_t[steel_mask] = steel_E_t

        # Interpolation weights
        alpha = (y_coords - y_bot) / h      # ∂strain/∂eps_top
        beta = np.subtract(1.0, alpha)       # ∂strain/∂eps_bottom

        # Derivative of force contributions: ∂F/∂eps = E_t * A * (∂strain/∂eps)
        dF_deps_top = E_t * areas * alpha
        dF_deps_bottom = E_t * areas * beta

        # Jacobian for axial force (sum contributions, convert N→kN)
        dN_deps_top = to_kn(np.sum(dF_deps_top), ForceUnit.N)
        dN_deps_bottom = to_kn(np.sum(dF_deps_bottom), ForceUnit.N)

        # Jacobian for moment (moment arm from cached centroid, convert N·mm→kN·m)
        y_offset = y_coords - self._section_cy

        dM_deps_top = to_knm(np.sum(dF_deps_top * y_offset), MomentUnit.NMM)
        dM_deps_bottom = to_knm(np.sum(dF_deps_bottom * y_offset), MomentUnit.NMM)

        # Assemble 2×2 Jacobian
        jac = np.array([
            [dN_deps_top, dN_deps_bottom],
            [dM_deps_top, dM_deps_bottom]
        ])

        return jac

    def _estimate_initial_strains(
        self,
        M: float,
        N: float,
    ) -> Tuple[float, float]:
        """
        Heuristic initial guess for strain pair based on (M, N) quadrant.

        Strategy:
        - Use sign of M and N to determine likely loading condition
        - Place strains in range that satisfies sign conventions:
            * Compression strain > 0 (concrete model expects positive for compression)
            * Tension strain < 0 (negative for steel in tension)
        - Avoid extreme values that might be outside model validity

        Args:
            M: Target moment (kN·m)
            N: Target axial force (kN, compression positive)

        Returns:
            (ε_top, ε_bottom): Initial guess for strain pair

        Sign convention (critical!):
            - Compression strain = POSITIVE (concrete model)
            - Tension strain = NEGATIVE (steel in tension)
            - This matches the fibre-based calculation in calculate_point_from_end_strains

        Loading cases:
            - Pure compression (N>0, M≈0): Both faces compressed → (+eps, +eps)
            - Sagging (M>0): Top compressed, bottom in tension → (+eps, -eps)
            - Hogging (M<0): Bottom compressed, top in tension → (-eps, +eps)
            - Pure tension (N<0, M≈0): Both faces in tension → (-eps, -eps)
        """
        eps_cu = self.concrete_model.get_ultimate_strain()  # Typical: 0.0035
        eps_y = self.steel_models[0].epsilon_y if self.steel_models else 0.002

        # Classify loading condition
        if N > 0:  # Compression dominant
            if abs(M) < 1e-6:  # Pure compression
                # Uniform compression strain (both POSITIVE)
                return (+eps_cu * 0.8, +eps_cu * 0.8)
            elif M > 0:  # Compression + positive moment (sagging)
                # Top more compressed, bottom less compressed or in tension
                return (+eps_cu * 0.8, +eps_cu * 0.2)
            else:  # Compression + negative moment (hogging)
                # Bottom more compressed, top less compressed or in tension
                return (+eps_cu * 0.2, +eps_cu * 0.8)

        elif N < 0:  # Tension dominant
            if abs(M) < 1e-6:  # Pure tension
                # Uniform tension strain (both NEGATIVE)
                return (-eps_y * 2.0, -eps_y * 2.0)
            elif M > 0:  # Tension + positive moment
                # Bottom in more tension
                return (-eps_y, -eps_y * 3.0)
            else:  # Tension + negative moment
                # Top in more tension
                return (-eps_y * 3.0, -eps_y)

        else:  # N ≈ 0: Pure bending
            if M > 0:  # Positive moment (sagging)
                # Top compressed (+), bottom in tension (-)
                return (+eps_cu * 0.8, -eps_y * 2.0)
            elif M < 0:  # Negative moment (hogging)
                # Bottom compressed (+), top in tension (-)
                return (-eps_y * 2.0, +eps_cu * 0.8)
            else:  # M ≈ 0 and N ≈ 0: Zero force
                return (0.0, 0.0)


    # ----------------------------
    # Geometric calculations for code checks
    # ----------------------------

    def get_effective_depth(
        self,
        M_Ed: float,
        N_Ed: float,
        eps_top: Optional[float] = None,
        eps_bottom: Optional[float] = None,
    ) -> float:
        """
        Get effective depth from compression face for a given load case.

        Delegates to ``find_effective_depth_for_flexure`` (the single source of truth).
        Uses default fallback policy (d = 0.9h) for ambiguous strain states.

        Args:
            M_Ed: Design moment in kN·m
            N_Ed: Design axial force in kN (compression positive)
            eps_top: Pre-computed top strain (optional, avoids re-solving)
            eps_bottom: Pre-computed bottom strain (optional, avoids re-solving)

        Returns:
            Effective depth in mm
        """
        from materials.reinforced_concrete.code_checks.ec2_2004.flexure_utils import (
            find_effective_depth_for_flexure,
        )
        return find_effective_depth_for_flexure(
            section=self.section,
            diagram=self,
            M_Ed=M_Ed,
            N_Ed=N_Ed,
            eps_top=eps_top,
            eps_bottom=eps_bottom,
            warn_on_fallback=False,
        )

    def get_lever_arm(
        self,
        M_Ed: float,
        N_Ed: float,
        d: Optional[float] = None,
        eps_top: Optional[float] = None,
        eps_bottom: Optional[float] = None,
        *,
        prefer_rigorous: bool = True,
        cap_to_09d: bool = True,
        min_z_ratio: float = 0.10,
        warn_on_fallback: bool = True,
    ) -> tuple[float, Optional[float]]:
        """
        Returns (z_ec2, z_mech).

        z_ec2 is ALWAYS usable for design (finite, positive, conservative):
        - defaults to 0.9d if rigorous calc is not requested or not meaningful
        - optionally capped to 0.9d

        z_mech is the uncapped centroid-based lever arm if computed, else None.
        """
        # Effective depth
        if d is None:
            d = self.get_effective_depth(M_Ed, N_Ed, eps_top, eps_bottom)

        d = float(d)
        if d <= 0:
            raise ValueError(f"Effective depth d must be > 0, got {d}")

        limit_09d = 0.9 * d

        # Cheap codified lever arm
        if not prefer_rigorous:
            return (limit_09d, None)

        # Near-zero moment: centroid lever arm is ill-posed / numerically unstable
        if abs(M_Ed) < 1e-6:
            if warn_on_fallback:
                warnings.warn(
                    "Lever arm fallback to 0.9d: |M_Ed| is ~0 so centroid-based lever arm "
                    "is ill-posed (pure axial/shear state).",
                    stacklevel=2,
                )
            return (limit_09d, None)

        # Need strains to compute centroid lever arm
        if eps_top is None or eps_bottom is None:
            eps_top, eps_bottom = self.find_strains_for_MN(M_target=M_Ed, N_target=N_Ed)

        # Try rigorous centroid-based lever arm
        z_mech = self._compute_lever_arm_from_centroids(eps_top, eps_bottom)

        # If not meaningful / suspicious, fall back
        if z_mech is None or (not np.isfinite(z_mech)):
            if warn_on_fallback:
                warnings.warn(
                    "Lever arm fallback to 0.9d: unable to compute a meaningful tension/compression "
                    "centroid lever arm for this strain state.",
                    stacklevel=2,
                )
            return (limit_09d, None)

        z_mech = float(z_mech)

        # sanity: too small relative to d is almost always numerical / axial-dominated
        if z_mech < min_z_ratio * d:
            if warn_on_fallback:
                warnings.warn(
                    f"Lever arm fallback to 0.9d: computed z_mech={z_mech:.1f} mm is < "
                    f"{min_z_ratio:.2f}d={min_z_ratio*d:.1f} mm (likely axial-dominated / numerical).",
                    stacklevel=2,
                )
            return (limit_09d, z_mech)

        # Apply EC2 cap
        if cap_to_09d and z_mech > limit_09d:
            warnings.warn(
                f"Lever arm capped: z_mech={z_mech:.1f}mm > 0.9d={limit_09d:.1f}mm. "
                "Using z=0.9d for EC2 truss model.",
                stacklevel=2,
            )
            return (limit_09d, z_mech)

        return (z_mech, z_mech)

    def _compute_lever_arm_from_centroids(
        self,
        eps_top: float,
        eps_bottom: float,
    ) -> Optional[float]:
        """
        Mechanical lever arm from force resultant centroids.
        Returns None if either tension or compression resultant is absent.
        """
        forces, y_coords, _ = self.get_fibre_forces_from_end_strains(eps_top, eps_bottom)

        tension_mask = forces < 0
        compression_mask = forces > 0

        # If you don't have both resultants, "lever arm between T and C" is not defined
        if (not np.any(tension_mask)) or (not np.any(compression_mask)):
            return None

        T_total = np.sum(-forces[tension_mask])
        C_total = np.sum(forces[compression_mask])

        # Guard against pathological near-zero totals
        if T_total <= 0 or C_total <= 0:
            return None

        y_T = np.sum((-forces[tension_mask]) * y_coords[tension_mask]) / T_total
        y_C = np.sum((forces[compression_mask]) * y_coords[compression_mask]) / C_total

        z_mech = abs(float(y_T) - float(y_C))
        if not np.isfinite(z_mech):
            return None
        
        return float(z_mech)


    # ----------------------------
    # Diagram generation
    # ----------------------------

    def _get_dense_diagram_points(self, n_dense: int) -> tuple[InteractionPoint, ...]:
        if self._dense_diagram_points is not None and self._dense_diagram_n == n_dense:
            return self._dense_diagram_points
        
        # --- Compression-side strain limit (concrete-controlled)
        eps_cu = float(self.concrete_model.get_ultimate_strain())

        # --- Tension-side strain limit (steel-controlled, finite by design)
        eps_t = self._eps_tension_limit()

        # --- Build a closed loop in strain space
        #     (ε_top, ε_bottom) pairs covering:
        #     pure compression → bending → pure tension → reverse bending → closure
        # Oversample in strain space (5–10x is typical)
        strain_pairs_dense = self._strain_limit_loop(
            n_points=n_dense,
            eps_cu=eps_cu,
            eps_t=eps_t,
        )

        dense_pts = tuple(
            self.calculate_point_from_end_strains(eps_top=et, eps_bottom=eb)
            for (et, eb) in strain_pairs_dense
        )

        self._dense_diagram_points = dense_pts
        self._dense_diagram_n = int(n_dense)
        self._diagram_points_cache.clear()
        return dense_pts
    

    def generate_diagram_points(
        self,
        n_points: int = 120,
        n_dense: int = 800,
    ) -> tuple[InteractionPoint, ...]:
        """
        Generate a closed M–N interaction envelope using end-strain parameterisation.

        The envelope is traced in (ε_top, ε_bottom) strain space:
            - Compression is limited by concrete ultimate strain ε_cu
            - Tension is limited by a steel-controlled strain ε_t (via _eps_tension_limit)

        This formulation:
            - avoids neutral-axis branch switching
            - eliminates artificial kinks near pure tension
            - produces a smooth, convex interaction envelope
        """
        n_points = int(max(n_points, 40))

        cached = self._diagram_points_cache.get(n_points)
        if cached is not None:
            return cached

        dense_pts = self._get_dense_diagram_points(n_dense=n_dense)
        pts = self._resample_closed_polyline_by_chord(dense_pts, n_out=n_points)
        pts = self._pin_extremal_points(pts, dense_pts)

        out = tuple(pts)
        self._diagram_points_cache[n_points] = out
        return out


    # ----------------------------
    # Capacity checks
    # ----------------------------

    def get_capacity_vector(
        self,
        N_Ed: float,
        M_Ed: float,
        n_points: int = 120,
        return_details: bool = False,
    ) -> CapacityResult:
        """
        Get capacity point (N_Rd, M_Rd) on the M-N boundary using ray intersection (vector method).

        Ray is defined as: (M, N) = t * (M_Ed, N_Ed), t >= 0

        If intersection scale is t_cap, then:
            (M_Rd, N_Rd) = t_cap * (M_Ed, N_Ed)
            utilization = 1 / t_cap

        Args:
            N_Ed: Design axial force (kN, compression positive)
            M_Ed: Design moment (kN·m)
            n_points: Number of points to generate M-N curve (default 120)
            return_details: If True, recompute exact strain state and return detailed metadata
                           (default False for speed)

        Returns:
            If return_details=False (default):
                (N_Rd, M_Rd, is_safe, utilization)

            If return_details=True:
                (N_Rd, M_Rd, is_safe, utilization, details_dict)

                where details_dict contains exact metadata at capacity:
                    - 'eps_top': Top fibre strain
                    - 'eps_bottom': Bottom fibre strain
                    - 'neutral_axis_depth': NA depth from section bottom (mm)
                    - 'compression_from_bottom': True if compression is at bottom
                    - 'max_concrete_strain': Maximum concrete compressive strain
                    - 'max_steel_strain': Maximum steel strain (absolute value)

                Note: Exact details require solving for strain state at (M_Rd, N_Rd),
                which adds computational cost but provides accurate metadata for detailed checks.

        Note on metadata accuracy:
            - Without return_details: Fast, but no strain/stress metadata available
            - With return_details: Slower, but exact strain/stress state at capacity

            The resampled M-N curve uses interpolated (M, N) coordinates (geometrically accurate),
            but strain metadata is approximate (from nearest dense point). When you need exact
            strains, stresses, NA depth, or lever arm at capacity, use return_details=True.
        """
        diagram_points = self.generate_diagram_points(n_points=n_points)
        pts = [(p.M, p.N) for p in diagram_points]
        if len(pts) < 3:
            return CapacityResult(N_Rd=None, M_Rd=None, is_safe=False, utilization=float("inf"))

        # Special case: origin (no load)
        if abs(M_Ed) < 1e-18 and abs(N_Ed) < 1e-18:
            if return_details:
                # At origin: no strain, no stress
                zero_details = {
                    'eps_top': 0.0,
                    'eps_bottom': 0.0,
                    'neutral_axis_depth': None,
                    'compression_from_bottom': None,
                    'max_concrete_strain': 0.0,
                    'max_steel_strain': 0.0,
                }
            else:
                zero_details = None
            return CapacityResult(N_Rd=0.0, M_Rd=0.0, is_safe=True, utilization=0.0, details=zero_details)

        ray_dir = (float(M_Ed), float(N_Ed))  # IMPORTANT: do NOT normalize

        # Ensure closed (duplicate endpoint convention)
        if pts[0] != pts[-1]:
            pts = pts + [pts[0]]

        # Find all ray-segment intersections
        # NOTE: For a convex closed curve, there should be exactly ONE intersection
        # If multiple intersections found, the curve may self-intersect (non-convex)
        intersections = []
        for p1, p2 in zip(pts[:-1], pts[1:]):
            if p1 == p2:
                continue
            t = _ray_segment_intersection_alpha(ray_dir, p1, p2, tol=1e-12)
            if t is not None:
                intersections.append(t)

        # Keep only forward intersections (positive t)
        ts = [t for t in intersections if t > 1e-12]
        if not ts:
            return CapacityResult(N_Rd=None, M_Rd=None, is_safe=False, utilization=float("inf"))

        # CRITICAL: Use MINIMUM t (first boundary hit as we move outward from origin)
        # This is the correct, conservative choice for capacity checks:
        # - For convex/star-shaped curves: min(ts) == max(ts) (single intersection)
        # - For non-convex curves: min(ts) gives first boundary hit (conservative)
        # Using max(ts) would be unconservative if curve self-intersects
        t_cap = min(ts)

        # Sanity check: warn if multiple intersections found (possible self-intersection)
        if len(ts) > 2:
            # More than 2 intersections suggests self-intersection or numerical issues
            # Note: exactly 2 intersections can occur for tangent rays (entry/exit at same point)
            import warnings
            warnings.warn(
                f"Ray intersection found {len(ts)} intersections (expected 1-2). "
                f"Curve may self-intersect. Using min(ts)={t_cap:.4f} (first hit, conservative). "
                f"Consider increasing n_points or checking diagram quality.",
                stacklevel=2
            )

        M_Rd = t_cap * float(M_Ed)
        N_Rd = t_cap * float(N_Ed)

        utilization = 1.0 / t_cap
        is_safe = utilization <= 1.0

        # Fast return if details not requested
        if not return_details:
            return CapacityResult(N_Rd=float(N_Rd), M_Rd=float(M_Rd), is_safe=bool(is_safe), utilization=float(utilization))

        # Recompute exact strain state and metadata at capacity point
        try:
            # Solve for exact strains that produce (M_Rd, N_Rd)
            eps_top, eps_bottom = self.find_strains_for_MN(M_Rd, N_Rd)

            # Compute NA depth and compression direction
            h = float(self.section_height)  # Use cached value
            if abs(eps_top - eps_bottom) < 1e-12:
                # Uniform strain (pure axial) - NA is undefined
                na_depth = None
                compression_from_bottom = eps_top > 0
            else:
                # Standard case: NA at zero-strain point
                na_depth = h * abs(eps_top) / abs(eps_top - eps_bottom)
                compression_from_bottom = eps_bottom > 0

            # Get max strains - use absolute values of end strains as approximation
            # (exact fibre-level analysis would require iterating through mesh)
            max_concrete_strain = max(abs(eps_top), abs(eps_bottom))
            max_steel_strain = max_concrete_strain  # Same strain field applies to all fibres

            details = {
                'eps_top': float(eps_top),
                'eps_bottom': float(eps_bottom),
                'neutral_axis_depth': float(na_depth) if na_depth is not None else None,
                'compression_from_bottom': bool(compression_from_bottom),
                'max_concrete_strain': float(max_concrete_strain),
                'max_steel_strain': float(max_steel_strain),
            }

            return CapacityResult(N_Rd=float(N_Rd), M_Rd=float(M_Rd), is_safe=bool(is_safe), utilization=float(utilization), details=details)

        except Exception as e:
            # If exact computation fails, return None for details
            # (e.g., if solver doesn't converge at boundary)
            import warnings
            warnings.warn(
                f"Failed to compute exact strain state at capacity (M_Rd={M_Rd:.2f}, N_Rd={N_Rd:.2f}): {e}. "
                f"Returning None for details.",
                stacklevel=2
            )
            return CapacityResult(N_Rd=float(N_Rd), M_Rd=float(M_Rd), is_safe=bool(is_safe), utilization=float(utilization))


    def get_utilization_vector(
        self,
        N_Ed: float,
        M_Ed: float,
        n_points: int = 120,
    ) -> Tuple[bool, float]:
        """Convenience wrapper returning (is_safe, utilization) using vector method."""
        capacity = self.get_capacity_vector(N_Ed=N_Ed, M_Ed=M_Ed, n_points=n_points, return_details=False)
        return (bool(capacity.is_safe), float(capacity.utilization))


    @staticmethod
    def _intersections_with_horizontal(
        pts: List[Tuple[float, float]],
        N0: float,
        tol: float = 1e-9,
    ) -> List[float]:
        """
        Intersect a polyline (M,N) with the horizontal line N=N0.
        Returns list of M values where intersections occur.
        """
        Ms: List[float] = []
        if len(pts) < 2:
            return Ms

        for (M1, N1), (M2, N2) in zip(pts[:-1], pts[1:]):
            # If segment is (nearly) horizontal
            if abs(N2 - N1) <= tol:
                # If it's on the query horizontal line, take endpoints as intersections
                if abs(N1 - N0) <= tol:
                    Ms.append(float(M1))
                    Ms.append(float(M2))
                continue

            # Check if N0 is between N1 and N2 (inclusive with tol)
            if (N0 - N1) * (N0 - N2) > tol:
                continue

            # Linear interpolation parameter along segment
            t = (N0 - N1) / (N2 - N1)  # can be slightly outside due to tol, clamp
            if t < -1e-12 or t > 1.0 + 1e-12:
                continue
            t = min(max(t, 0.0), 1.0)

            Mx = M1 + t * (M2 - M1)
            Ms.append(float(Mx))

        # De-duplicate within tolerance (important around vertices)
        Ms.sort()
        out: List[float] = []
        for m in Ms:
            if not out or abs(m - out[-1]) > 1e-7:  # moment tolerance in kN·m
                out.append(m)
        return out


    def get_capacity_fixed_n(
        self,
        N_Ed: float,
        *,
        n_points: int = 160,
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Horizontal-line capacity at fixed axial force.

        "Caps the axial":
          - If N_Ed is outside the diagram's axial range, it is clamped to [N_min, N_max]
            so you still get a sensible moment capacity at the nearest achievable axial level.

        Returns:
            (N_cap, M_Rd_pos, M_Rd_neg)

        where:
            - N_cap is the clamped axial level used for the intersection
            - M_Rd_pos is the maximum positive moment capacity at N_cap
            - M_Rd_neg is the minimum (most negative) moment capacity at N_cap

        If intersections cannot be found, returns (None, None, None).
        """
        diagram_points = self.generate_diagram_points(n_points=n_points)  # should be closed already
        if len(diagram_points) < 4:
            return (None, None, None)

        pts = [(float(p.M), float(p.N)) for p in diagram_points]

        # Ensure closed
        if pts[0] != pts[-1]:
            pts = pts + [pts[0]]

        N_vals = [N for _, N in pts]
        N_min = float(min(N_vals))
        N_max = float(max(N_vals))

        # Cap axial
        N_cap = float(min(max(N_Ed, N_min), N_max))

        # Find intersections with horizontal line N=N_cap
        Ms = self._intersections_with_horizontal(pts, N0=N_cap, tol=1e-9)

        if not Ms:
            # Extremely rare if diagram is well-formed; return None to signal failure
            return (None, None, None)

        M_Rd_pos = float(max(Ms))
        M_Rd_neg = float(min(Ms))
        return (N_cap, M_Rd_pos, M_Rd_neg)

    # ----------------------------
    # Export / convenience
    # ----------------------------

    def get_diagram_arrays(
        self,
        n_points: int = 120,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        pts = self.generate_diagram_points(n_points=n_points)
        N = np.array([p.N for p in pts], dtype=float)
        M = np.array([p.M for p in pts], dtype=float)
        return (N, M)


    def export_to_json(
        self,
        file_path: str | Path,
        n_points: int = 120,
        include_metadata: bool = True,
        indent: int = 2,
    ) -> None:
        points = self.generate_diagram_points(n_points=n_points)
        data: Dict[str, Any] = {"diagram_points": [p.to_dict() for p in points]}

        if include_metadata:
            data["metadata"] = {
                "section_name": self.section.section_name,
                "concrete_grade": self.concrete.grade,
                "concrete_fck": self.concrete.f_ck,
                "concrete_fcd": self.concrete.f_cd,
                "n_rebar_groups": len(self.section.rebar_groups),
                "n_fibres": self.mesh.total_fibres,
                "concrete_model": type(self.concrete_model).__name__,
                "steel_models": [type(sm).__name__ for sm in self.steel_models],
                "tension_stiffening": self.tension_stiffening,
                "confined_concrete": self.confined_concrete,
            }

        file_path = Path(file_path)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent)


    def export_to_csv(
        self,
        file_path: str | Path,
        n_points: int = 120,
        include_strains: bool = True,
    ) -> None:
        points = self.generate_diagram_points(n_points=n_points)

        file_path = Path(file_path)
        with open(file_path, "w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "N",
                "M",
                "neutral_axis_depth",
                "compression_from_bottom",
                "max_concrete_strain",
                "max_steel_strain",
            ] if include_strains else ["N", "M"]

            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for p in points:
                row = p.to_dict()
                if not include_strains:
                    row = {"N": row["N"], "M": row["M"]}
                writer.writerow(row)


    def to_dict(
        self,
        n_points: int = 120,
        include_metadata: bool = True,
    ) -> Dict[str, Any]:
        points = self.generate_diagram_points(n_points=n_points)
        data: Dict[str, Any] = {
            "points": [p.to_dict() for p in points],
            "N_array": [p.N for p in points],
            "M_array": [p.M for p in points],
        }
        if include_metadata:
            data["metadata"] = {
                "section_name": self.section.section_name,
                "concrete_grade": self.concrete.grade,
                "concrete_fck": self.concrete.f_ck,
                "concrete_fcd": self.concrete.f_cd,
                "n_rebar_groups": len(self.section.rebar_groups),
                "n_fibres": self.mesh.total_fibres,
                "concrete_model": type(self.concrete_model).__name__,
                "steel_models": [type(sm).__name__ for sm in self.steel_models],
                "tension_stiffening": self.tension_stiffening,
                "confined_concrete": self.confined_concrete,
            }
        return data


    # -----------------------------------
    # Plotting (kept for encapsulated UX)
    # -----------------------------------
    def plot_mn(
        self,
        *,
        load_points: Optional[List[Dict[str, Any]]] = None,
        show_vectors: bool = False,
        show_metadata: bool = True,
        n_points: int = 120,
        save_path: Optional[str | Path] = None,
        show: bool = True,
        title: Optional[str] = None,
        width: int = 900,
        height: int = 700,
    ) -> Any:
        """
        Plot M-N interaction diagram with optional load points using Plotly.

        Creates an interactive plot with:
        - M-N interaction curve boundary
        - Optional load points with color-coded utilization
        - Optional vector projection rays from origin to boundary
        - Interactive hover tooltips with metadata

        Args:
            load_points: List of load case dictionaries with format:
                {
                    "N_Ed": float,      # Axial force (kN)
                    "M_Ed": float,      # Moment (kN·m)
                    "name": str,        # Load case name (optional)
                }
            show_vectors: If True, show vector projection rays from origin through
                            load points to capacity boundary
            show_metadata: If True, show metadata in hover tooltips
            n_points: Number of points to generate M-N curve
            save_path: If provided, save plot to this file path (HTML format)
            show: If True, display plot (fig.show())
            title: Custom plot title (optional)
            width: Figure width in pixels
            height: Figure height in pixels

        Returns:
            Plotly Figure object
        """
        from materials.reinforced_concrete.analysis.mn_diagram_viewer import MNDiagramViewer

        viewer = MNDiagramViewer(self)
        return viewer.plot(
            load_points=load_points,
            show_vectors=show_vectors,
            show_metadata=show_metadata,
            n_points=n_points,
            save_path=save_path,
            show=show,
            title=title,
            width=width,
            height=height,
        )


    def plot_stress_strain(
        self,
        M_Ed: float,
        N_Ed: float,
        *,
        show: bool = True,
        title: Optional[str] = None,
        width: int = 1200,
        height: int = 1000,
        section_render: Literal["points", "filled"] = "points",
    ) -> Any:
        """
        Visualize stress and strain distribution for a given load case.

        This is a thin wrapper that delegates plotting to the stress_strain_view module,
        keeping MNInteractionDiagram focused on analysis rather than plotting.
        """
        # Local import keeps plotly/shapely out of core import path unless plotting is used
        from materials.reinforced_concrete.analysis.stress_strain_viewer import StressStrainViewer

        viewer = StressStrainViewer(self)
        return viewer.plot(
            M_Ed=float(M_Ed),
            N_Ed=float(N_Ed),
            show=show,
            title=title,
            width=width,
            height=height,
            section_render=section_render,
        )


    def _compute_z_d_for_moment(
        self,
        *,
        M_Ed: float,
        N_Ed: float,
        prefer_rigorous: bool = False,
        cap_to_09d: bool = True,
        warn_on_fallback: bool = False,
    ) -> tuple[float, float]:
        """
        Compute lever arm z and effective depth d for a given moment value.

        Args:
            M_Ed: Moment for strain analysis (kN·m)
            N_Ed: Axial force (kN, positive = compression)
            prefer_rigorous: If True, attempt to compute the rigorous centroid-based
                lever arm from strain analysis. If False (default), use the simplified
                0.9d approach per EC2 §6.2.3(1).
            cap_to_09d: If True (default), cap the lever arm to 0.9d per EC2.
                Only relevant when prefer_rigorous=True.
            warn_on_fallback: If True, emit a warning when the rigorous lever arm
                calculation falls back to 0.9d (e.g., near-zero moment, numerical
                issues). Default False to avoid noise in batch calculations.

        Returns:
            (z, d) in mm, where z may be capped to 0.9d depending on settings
        """
        eps_top, eps_bottom = None, None
        if abs(M_Ed) > 1e-6:
            eps_top, eps_bottom = self.find_strains_for_MN(M_Ed, N_Ed)

        d = self.get_effective_depth(
            M_Ed=M_Ed,
            N_Ed=N_Ed,
            eps_top=eps_top,
            eps_bottom=eps_bottom,
        )
        z, _ = self.get_lever_arm(
            M_Ed=M_Ed,
            N_Ed=N_Ed,
            d=d,
            eps_top=eps_top,
            eps_bottom=eps_bottom,
            prefer_rigorous=prefer_rigorous,
            cap_to_09d=cap_to_09d,
            warn_on_fallback=warn_on_fallback,
        )
        return z, d

    def apply_tension_shift(
        self,
        *,
        M_Ed: float,
        V_Ed: float,
        N_Ed: float = 0.0,
        M_cap: Optional[float] = None,
        shear_reinforcement: Optional["ShearRebar"] = None,
        cot_theta_override: Optional[float] = None,
        use_v_rd_s_for_cot_theta: bool = False,
        cot_max_override: Optional[float] = None,
        iterate_z: bool = False,
        prefer_rigorous: bool = False,
        cap_to_09d: bool = True,
        warn_on_fallback: bool = False,
    ) -> "TensionShiftResult":
        """
        Apply EC2 §9.2.1.3 tension shift rule to a bending moment.

        This method computes z and d from the diagram's strain analysis and applies
        the tension shift rule. Use this when you want to shift load cases before
        plotting with `plot_stress_strain` or checking capacity.

        Args:
            M_Ed: Design bending moment (kN·m)
            V_Ed: Design shear force (kN)
            N_Ed: Design axial force (kN, positive = compression). Default 0.
            M_cap: Optional moment capacity cap (kN·m). Limits |M_design| ≤ |M_cap|.
            shear_reinforcement: Optional ShearRebar object. If provided, calculates
                                cot(θ) using the variable strut angle method.
                                If not provided, uses a_l = d (no shear reinforcement).
            cot_theta_override: Optional user-supplied cot(θ) value. When provided
                with shear_reinforcement, this value is used directly instead of
                calculating cot(θ) from V_Ed and V_Rd,max. Must be in the valid
                EC2 range [1.0, 2.5]. Clamped if outside range.
            use_v_rd_s_for_cot_theta: If True, determine cot(θ) from rearranged
                EC2 Eq. 6.13 (V_Rd,s = V_Ed). If False (default), determine cot(θ)
                from rearranged EC2 Eq. 6.14 / V_Rd,max.
            cot_max_override: Optional upper limit for cot(θ). When provided,
                overrides the NDP default (e.g. 1.25 for UK NA with tension).
            iterate_z: If True, iteratively recalculate z based on M_design until
                      convergence (0.5% tolerance, max 5 iterations). Only has an
                      effect when BOTH shear_reinforcement is provided (so a_l depends
                      on z) AND prefer_rigorous=True (so z depends on M). With
                      prefer_rigorous=False, z=0.9d always so iteration is skipped.
            prefer_rigorous: If True, attempt to compute the rigorous centroid-based
                lever arm from strain analysis. If False (default), use the simplified
                0.9d approach per EC2 §6.2.3(1).
            cap_to_09d: If True (default), cap the lever arm to 0.9d per EC2.
                Only relevant when prefer_rigorous=True.
            warn_on_fallback: If True, emit a warning when the rigorous lever arm
                calculation falls back to 0.9d (e.g., near-zero moment, numerical
                issues). Default False to avoid noise in batch calculations.

        Returns:
            TensionShiftResult with shifted moment and calculation details.

        Example:
            >>> diagram = MNInteractionDiagram(section, concrete)
            >>> # Shift moment for plotting
            >>> result = diagram.apply_tension_shift(M_Ed=100, V_Ed=50, N_Ed=200)
            >>> diagram.plot_stress_strain(M_Ed=result.M_design, N_Ed=200)
        """
        # Local imports to avoid circular dependencies
        from materials.reinforced_concrete.code_checks.ec2_2004.shear_utils import (
            calculate_tension_shift,
            calculate_section_breadth,
        )

        M_Ed_original = float(M_Ed)
        N_Ed = float(N_Ed)

        # Compute initial z and d
        z, d = self._compute_z_d_for_moment(
            M_Ed=M_Ed_original,
            N_Ed=N_Ed,
            prefer_rigorous=prefer_rigorous,
            cap_to_09d=cap_to_09d,
            warn_on_fallback=warn_on_fallback,
        )

        # Compute parameters needed for shear reinforcement case
        b_w: Optional[float] = None
        f_cd: Optional[float] = None
        f_ck: Optional[float] = None
        sigma_cp: float = 0.0

        if shear_reinforcement is not None:
            from materials.reinforced_concrete.code_checks.ec2_2004.shear_utils import (
                sigma_cp_from_N_and_area,
                cap_sigma_cp_upper,
            )
            b_w = calculate_section_breadth(section=self.section)
            f_cd = self.concrete.f_cd
            f_ck = self.concrete.f_ck
            A_transformed = self.section.get_transformed_area(self.concrete.E_cm)
            sigma_cp_uncapped = sigma_cp_from_N_and_area(N_Ed=N_Ed, area=A_transformed)
            sigma_cp = cap_sigma_cp_upper(sigma_cp=sigma_cp_uncapped, f_cd=f_cd)

        # Initial calculation
        shift_result = calculate_tension_shift(
            M_Ed=M_Ed_original,
            V_Ed=V_Ed,
            z=z,
            d=d,
            M_cap=M_cap,
            b_w=b_w,
            f_cd=f_cd,
            f_ck=f_ck,
            sigma_cp=sigma_cp,
            shear_reinforcement=shear_reinforcement,
            cot_theta_override=cot_theta_override,
            use_v_rd_s_for_cot_theta=use_v_rd_s_for_cot_theta,
            cot_max_override=cot_max_override,
        )

        # Iterate z if requested, shear reinforcement is provided, AND prefer_rigorous=True
        # - Without shear reinforcement: a_l = d which doesn't depend on z
        # - Without prefer_rigorous: z = 0.9d always (doesn't depend on M)
        if iterate_z and shear_reinforcement is not None and prefer_rigorous:
            MAX_ITERATIONS = 5
            CONVERGENCE_TOL = 0.005  # 0.5%

            for _ in range(MAX_ITERATIONS):
                # Recalculate z for the current M_design
                z_new, d_new = self._compute_z_d_for_moment(
                    M_Ed=shift_result.M_design,
                    N_Ed=N_Ed,
                    prefer_rigorous=prefer_rigorous,
                    cap_to_09d=cap_to_09d,
                    warn_on_fallback=warn_on_fallback,
                )

                # Check convergence
                if z > 1e-6:
                    rel_change = abs(z_new - z) / z
                    if rel_change < CONVERGENCE_TOL:
                        # Converged - update final values
                        z = z_new
                        d = d_new
                        shift_result = calculate_tension_shift(
                            M_Ed=M_Ed_original,
                            V_Ed=V_Ed,
                            z=z,
                            d=d,
                            M_cap=M_cap,
                            b_w=b_w,
                            f_cd=f_cd,
                            f_ck=f_ck,
                            sigma_cp=sigma_cp,
                            shear_reinforcement=shear_reinforcement,
                            cot_theta_override=cot_theta_override,
                            use_v_rd_s_for_cot_theta=use_v_rd_s_for_cot_theta,
                        )
                        break

                # Update for next iteration
                z = z_new
                d = d_new
                shift_result = calculate_tension_shift(
                    M_Ed=M_Ed_original,
                    V_Ed=V_Ed,
                    z=z,
                    d=d,
                    M_cap=M_cap,
                    b_w=b_w,
                    f_cd=f_cd,
                    f_ck=f_ck,
                    sigma_cp=sigma_cp,
                    shear_reinforcement=shear_reinforcement,
                    cot_theta_override=cot_theta_override,
                    use_v_rd_s_for_cot_theta=use_v_rd_s_for_cot_theta,
                )

        return shift_result

    def __repr__(self) -> str:
        return (
            f"MNInteractionDiagram("
            f"section={self.section.section_name}, "
            f"concrete={self.concrete.grade}, "
            f"fibres={self.mesh.total_fibres}, "
            f"tension_stiffening={self.tension_stiffening}, "
            f"confined={self.confined_concrete})"
        )


def create_interaction_diagram(
    section: RCSection,
    concrete: ConcreteMaterial,
    **kwargs: Any,
) -> MNInteractionDiagram:
    """
    Factory function to create M-N interaction diagram.

    Args:
        section: RC section with reinforcement
        concrete: Concrete material
        concrete_model_type: Stress-strain relationship of concrete
        steel_model_type: Stress-strain relationship of rebar
        n_fibres_width: Number of fibres to split width of section
        n_fibres_height: Number of fibres to split height of section
        tension_stiffening:
            Concrete in tension contributes post-cracking using a simplified
            EC2-style average tension stress-strain relationship
        use_characteristic: Enables characteristic strength limits for materials
        use_accidental: Enables accidental limit state factors for design strengths of materials
        confined_concrete: Enables a Mander-style confined concrete response is applied in compression
        confinement_rho_s: Must be provided when confined_concrete=True
        confinement_f_yh: Characteristic transverse steel yield strength for confinement
        ignore_compression_steel: If True, steel in compression (positive strain) contributes
            zero force.
        **kwargs: Additional arguments passed to MNInteractionDiagram

    Returns:
        MNInteractionDiagram instance
    """
    return MNInteractionDiagram(section=section, concrete=concrete, **kwargs)
