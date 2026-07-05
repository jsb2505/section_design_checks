"""
M-N interaction diagram generator using fiber-based strain compatibility.

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
- Closing points (pure compression and pure tension) are computed via the SAME fiber integration used elsewhere for consistency.

Notes on confinement:
- If confined_concrete=True, a Mander-style confined concrete response is applied in compression.
- To avoid “double factoring”, confinement is computed at characteristic level (using f_ck and f_yk) then reduced to design level
  via the same factor used for unconfined concrete (alpha_cc / gamma_c), so the solver remains consistent with design strengths.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import csv
import json
import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field
from scipy.optimize import least_squares, root_scalar

from materials.reinforced_concrete.constitutive import (
    create_concrete_stress_strain,
    create_steel_stress_strain,
)
from materials.reinforced_concrete.geometry import FiberMesh, RCSection
from materials.reinforced_concrete.materials import ConcreteMaterial


# ----------------------------
# Types / small utilities
# ----------------------------

ConcreteModelType = Literal["parabola-rectangle", "bilinear", "schematic"]
SteelBranchType = Literal["inclined", "horizontal"]


def _as_float(x: Any) -> float:
    """
    Convert numpy scalars cleanly to Python float.

    For complex-step differentiation, extracts real part.
    """
    if np.iscomplexobj(x):
        return float(np.real(x))
    return float(x)  # raises if not convertible (good)


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
            "N_kN": self.N,
            "M_kNm": self.M,
            "neutral_axis_depth_mm": self.neutral_axis_depth,
            "compression_from_bottom": self.compression_from_bottom,
            "max_concrete_strain": self.max_concrete_strain,
            "max_steel_strain": self.max_steel_strain,
        }


# ----------------------------
# Main solver
# ----------------------------

class MNInteractionDiagram:
    """
    M-N interaction diagram generator using fiber-based strain compatibility (2D single-axis).

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
        concrete_model_type: ConcreteModelType = "parabola-rectangle",
        steel_branch_type: SteelBranchType = "inclined",
        n_fibers_width: int = 20,
        n_fibers_height: int = 30,
        tension_stiffening: bool = False,
        use_characteristic: bool = False,
        use_accidental: bool = False,
        confined_concrete: bool = False,
        confinement_rho_s: Optional[float] = None,
        confinement_f_yh: Optional[float] = None,
    ):
        self.section = section
        self.concrete = concrete

        self.tension_stiffening = tension_stiffening
        self.confined_concrete = confined_concrete
        self.confinement_rho_s = confinement_rho_s

        # IMPORTANT: treat confinement_f_yh as CHARACTERISTIC if provided.
        # If None, default to the first longitudinal group's characteristic yield strength.
        self.confinement_f_yh = confinement_f_yh

        # Constitutive models for design-level capacity evaluation
        self.concrete_model = create_concrete_stress_strain(
            concrete=concrete,
            model_type=concrete_model_type,
            use_characteristic=use_characteristic,
            use_accidental=use_accidental
        )

        if len(section.rebar_groups) == 0:
            raise ValueError("Section must have at least one rebar group")

        # Steel models per group (support different grades)
        self.steel_models = [
            create_steel_stress_strain(
                steel=g.rebar,
                branch_type=steel_branch_type,
                use_characteristic=use_characteristic,
                use_accidental=use_accidental
            )
            for g in section.rebar_groups
        ]

        # Confined concrete parameter checks
        if self.confined_concrete:
            if self.confinement_rho_s is None:
                raise ValueError("confinement_rho_s must be provided when confined_concrete=True")
            if not (0.0 < self.confinement_rho_s <= 0.1):
                raise ValueError(f"confinement_rho_s must be in (0, 0.1], got {self.confinement_rho_s}")

            if self.confinement_f_yh is None:
                # default to characteristic yield of first longitudinal group
                self.confinement_f_yh = section.rebar_groups[0].rebar.f_yk

            if self.confinement_f_yh <= 0:
                raise ValueError(f"confinement_f_yh must be > 0, got {self.confinement_f_yh}")

        # Fiber mesh
        self.mesh = FiberMesh(
            section=section,
            n_fibers_width=n_fibers_width,
            n_fibers_height=n_fibers_height,
            exclude_steel_area=True,
        )

        # Geometry references
        _, min_y, _, max_y = section.get_bounding_box()
        self.section_top = max_y
        self.section_bottom = min_y
        self.section_height = max_y - min_y

        if self.section_height <= 0:
            raise ValueError("Section height must be > 0")

        # Cache fiber arrays for performance (avoid repeated allocation/copy in residual/Jacobian)
        self._fiber_x, self._fiber_y, self._fiber_area, self._fiber_mat, self._fiber_mi = self.mesh.get_fiber_arrays()
        self._fiber_mat = self._fiber_mat.astype("U8", copy=False)  # Ensure consistent dtype

        # Cache section centroid (avoid repeated Shapely geometry access)
        _, self._section_cy = self.section.get_centroid()


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
        Calculate resultant axial force and single-axis bending moment from fiber stresses.

        Args:
            stresses: Stress at each fiber (MPa = N/mm²), same order as mesh.get_fiber_arrays()
            use_section_centroid: If True, take moments about gross concrete centroid (section.get_centroid()).
                                 This matches the rest of the solver and plotting conventions.

        Returns:
            (N, M):
                N in kN (positive compression)
                M in kN·m about the section centroid, using y-offset (single-axis)
        """
        # Use cached fiber arrays for performance
        y = self._fiber_y
        area = self._fiber_area

        # Axial force: sum(σ * A) in N, convert to kN
        N = np.sum(stresses * area) / 1000.0

        # Moment about section centroid (single axis about y-offset)
        if use_section_centroid:
            cy = self._section_cy  # Use cached centroid
        else:
            cy = float(np.sum(y * area) / np.sum(area))

        y_offset = y - cy
        M = np.sum(stresses * area * y_offset) / 1_000_000.0  # N·mm -> kN·m

        return (_as_float(N), _as_float(M))


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

        # ------------------
        # Confined concrete (compression only)
        # ------------------
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

        # ------------------
        # Tension stiffening (tension only)
        # ------------------
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
                    -f_ctm * np.maximum(0.0, 1.0 - beta * (eps_t - eps_cr) / (eps_cr * 5.0)),
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
        Plane-sections strain field defined by strains at the extreme top/bottom fibers
        (compression positive), linear over y.

        eps_top: strain at y = section_top
        eps_bottom: strain at y = section_bottom
        """
        # Use cached fiber y-coordinates
        y = self._fiber_y

        y_top = float(self.section_top)
        y_bot = float(self.section_bottom)
        h = y_top - y_bot
        if h <= 0.0:
            raise ValueError("Invalid section height")

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
        # Use cached fiber arrays for performance
        material_type = self._fiber_mat  # Already converted to U8 in __init__
        material_index = self._fiber_mi

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
            N=_as_float(N),
            M=_as_float(M),
            neutral_axis_depth=float(na_depth),
            compression_from_bottom=bool(comp_from_bottom),
            max_concrete_strain=max_conc,
            max_steel_strain=max_steel,
        )


    def get_fiber_forces_from_end_strains(
        self,
        eps_top: float,
        eps_bottom: float,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Compute fiber-level forces from strain profile (public helper for external tools).

        This is a PUBLIC interface for computing detailed force distributions, intended
        for use by code checks and other analyses that need fiber-level data without
        accessing private internals.

        Args:
            eps_top: Strain at top fiber (compression positive)
            eps_bottom: Strain at bottom fiber (compression positive)

        Returns:
            Tuple of (forces, y_coords, areas):
                - forces: Force in each fiber (N), compression positive
                - y_coords: Y-coordinate of each fiber (mm)
                - areas: Area of each fiber (mm²)

        Example:
            >>> diagram = MNInteractionDiagram(section, concrete)
            >>> eps_top, eps_bottom = diagram.find_strains_for_MN(M=50.0, N=100.0)
            >>> forces, y_coords, areas = diagram.get_fiber_forces_from_end_strains(eps_top, eps_bottom)
            >>> # Compute tension/compression centroids for lever arm
            >>> tension_mask = forces < 0
            >>> y_T = np.sum(-forces[tension_mask] * y_coords[tension_mask]) / np.sum(-forces[tension_mask])
        """
        # Use cached fiber arrays for performance
        y = self._fiber_y
        area = self._fiber_area
        material_type = self._fiber_mat  # Already converted to U8 in __init__
        material_index = self._fiber_mi

        # Compute strains at all fibers
        strains = self._strain_field_from_end_strains(eps_top=eps_top, eps_bottom=eps_bottom)

        # Compute stresses
        stresses = np.zeros_like(strains)

        # Concrete fibers - use internal method that handles confinement/tension stiffening
        conc_mask = material_type == "concrete"
        if np.any(conc_mask):
            stresses[conc_mask] = self._concrete_stress_with_options(strains[conc_mask])

        # Steel fibers
        steel_mask = material_type == "steel"
        if np.any(steel_mask):
            steel_strains = strains[steel_mask]
            steel_indices = material_index[steel_mask]
            steel_stresses = np.zeros_like(steel_strains)
            for gi, sm in enumerate(self.steel_models):
                m = steel_indices == gi
                if np.any(m):
                    steel_stresses[m] = sm.get_stress_array(steel_strains[m])
            stresses[steel_mask] = steel_stresses

        # Forces per fiber (compression positive): Force = stress × area
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
        return 0.5 * (1.0 - np.cos(np.pi * t))


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
        points: List[InteractionPoint],
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
        if len(points) < 4:
            return points

        # Ensure closed
        pts = points
        if (pts[0].M != pts[-1].M) or (pts[0].N != pts[-1].N):
            pts = pts + [pts[0]]

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


    # ----------------------------
    # Inverse solver (M, N) → (ε_top, ε_bottom)
    # ----------------------------

    def find_strains_for_MN(
        self,
        M_target: float,
        N_target: float,
        initial_guess: Optional[Tuple[float, float]] = None,
        tol: float = 1e-6,
    ) -> Tuple[float, float]:
        """
        Inverse solver: Find end strains (ε_top, ε_bottom) that produce target (M, N).

        Uses scipy.optimize.least_squares with numerical Jacobian to solve the
        2D root-finding problem:
            calculate_point_from_end_strains(ε_top, ε_bottom) = (N_target, M_target)

        This method does NOT require generate_diagram() to have been called - it only
        needs the fiber mesh and constitutive models (created in __init__).

        Args:
            M_target: Target moment (kN·m)
            N_target: Target axial force (kN, compression positive)
            initial_guess: Optional (ε_top, ε_bottom) starting point for optimizer.
                          If None, automatically estimated from (M, N) quadrant.
            tol: Convergence tolerance for residual norm and parameter changes

        Returns:
            (ε_top, ε_bottom): Tuple of end strains that produce target forces

        Raises:
            ValueError: If no solution found (point may be outside diagram envelope)
                       or if solver fails to converge

        Performance:
            - Typical solve time: 10-50ms per unique (M,N) point
            - No caching - solves fresh each time per design

        Examples:
            >>> diagram = MNInteractionDiagram(section, concrete)
            >>> # Find strains for sagging moment with compression
            >>> eps_top, eps_bottom = diagram.find_strains_for_MN(M_target=50.0, N_target=100.0)
            >>> # Verify round-trip
            >>> point = diagram.calculate_point_from_end_strains(eps_top, eps_bottom)
            >>> assert abs(point.M - 50.0) < 1e-3
            >>> assert abs(point.N - 100.0) < 1e-3
        """
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

        # Solve using least squares
        # Analytical Jacobian: exact derivatives, 5-10 iterations typical
        # Numerical Jacobian: finite difference, 30-50 iterations typical
        result = least_squares(
            residual,
            x0=np.array(initial_guess),
            bounds=(lower_bounds, upper_bounds),  # Prevent absurd strains
            jac=jac_method,  # type: ignore[arg-type]  # Analytical (plain) or numerical (confined/tension stiffening)
            ftol=tol,
            xtol=tol,
            gtol=tol,  # Gradient tolerance (helps convergence for awkward load points)
            max_nfev=max_iterations,
        )

        if not result.success:
            raise ValueError(
                f"Inverse solver failed for M={M_target:.2f} kN·m, N={N_target:.2f} kN. "
                f"Reason: {result.message}. "
                "Point may be outside section capacity envelope. "
                f"Final residual: {result.fun}"
            )

        return tuple(result.x)

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
            For each fiber at height y:
                strain(y) = eps_bottom + (eps_top - eps_bottom) * (y - y_bot) / h

            Define:
                α(y) = (y - y_bot) / h     (linear interpolation weight)
                β(y) = 1 - α(y)            (complementary weight)

            Then:
                ∂strain/∂eps_top = α(y)
                ∂strain/∂eps_bottom = β(y)

            Force contribution from fiber i:
                F_i = σ_i * A_i = σ(ε_i) * A_i

            Derivative:
                ∂F_i/∂eps_top = (dσ/dε)|_i * A_i * α(y_i)     [E_t * A * α]
                ∂F_i/∂eps_bottom = (dσ/dε)|_i * A_i * β(y_i)  [E_t * A * β]

            Axial force:
                N = Σ F_i  →  ∂N/∂eps = Σ ∂F_i/∂eps

            Moment (about centroid c_y):
                M = Σ F_i * (y_i - c_y)  →  ∂M/∂eps = Σ [∂F_i/∂eps * (y_i - c_y)]

        Args:
            eps_top: Top fiber strain (compression positive)
            eps_bottom: Bottom fiber strain (compression positive)

        Returns:
            2×2 Jacobian matrix [[dN_deps_top, dN_deps_bottom],
                                 [dM_deps_top, dM_deps_bottom]]
        """
        # Use cached fiber arrays for performance
        y_coords = self._fiber_y
        areas = self._fiber_area
        material_type = self._fiber_mat  # Already converted to U8 in __init__
        material_index = self._fiber_mi

        y_top = float(self.section_top)
        y_bot = float(self.section_bottom)
        h = float(self.section_height)

        # Compute strain at each fiber
        strains = eps_bottom + (eps_top - eps_bottom) * (y_coords - y_bot) / h

        # Compute tangent modulus E_t = dσ/dε at each fiber
        E_t = np.zeros_like(strains)

        # Concrete fibers - use method with tension stiffening support
        conc_mask = material_type == "concrete"
        if np.any(conc_mask):
            E_t[conc_mask] = self._concrete_tangent_modulus_with_options(strains[conc_mask])

        # Steel fibers
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
        beta = 1.0 - alpha                   # ∂strain/∂eps_bottom

        # Derivative of force contributions: ∂F/∂eps = E_t * A * (∂strain/∂eps)
        dF_deps_top = E_t * areas * alpha
        dF_deps_bottom = E_t * areas * beta

        # Jacobian for axial force (sum contributions, convert N→kN)
        dN_deps_top = np.sum(dF_deps_top) / 1000.0
        dN_deps_bottom = np.sum(dF_deps_bottom) / 1000.0

        # Jacobian for moment (moment arm from cached centroid, convert N·mm→kN·m)
        y_offset = y_coords - self._section_cy

        dM_deps_top = np.sum(dF_deps_top * y_offset) / 1_000_000.0
        dM_deps_bottom = np.sum(dF_deps_bottom * y_offset) / 1_000_000.0

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
            - This matches the fiber-based calculation in calculate_point_from_end_strains

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
                # TODO: Could be improved by checking eccentricity e = M/N vs h
                # to determine if tension develops (e > h/6 typically indicates tension).
                # However, changing this requires extensive testing to avoid wrong local minima.
                # See docs/INITIAL_GUESS_HEURISTIC.md for details and attempted implementation.
                # Current simple guess works reliably for most cases.
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
    # Diagram generation
    # ----------------------------

    def generate_diagram(
        self,
        n_points: int = 120,
    ) -> List[InteractionPoint]:
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
        # --- Compression-side strain limit (concrete-controlled)
        eps_cu = float(self.concrete_model.get_ultimate_strain())

        # --- Tension-side strain limit (steel-controlled, finite by design)
        eps_t = self._eps_tension_limit()

        # --- Build a closed loop in strain space
        #     (ε_top, ε_bottom) pairs covering:
        #     pure compression → bending → pure tension → reverse bending → closure
        # Oversample in strain space (5–10x is typical)
        n_dense = int(max(5 * n_points, 300))
        strain_pairs_dense = self._strain_limit_loop(
            n_points=n_dense,
            eps_cu=eps_cu,
            eps_t=eps_t,
        )

        dense_pts: List[InteractionPoint] = [
            self.calculate_point_from_end_strains(eps_top=et, eps_bottom=eb)
            for (et, eb) in strain_pairs_dense
        ]

        # Resample to uniform chord-length in (M,N)
        pts = self._resample_closed_polyline_by_chord(dense_pts, n_out=int(max(n_points, 40)))

        return pts


    # ----------------------------
    # Capacity checks
    # ----------------------------

    def get_capacity_vector(
        self,
        N_Ed: float,
        M_Ed: float,
        n_points: int = 120,
    ) -> Tuple[Optional[float], Optional[float], bool, float]:
        """
        Get capacity point (N_Rd, M_Rd) on the M-N boundary using ray intersection (vector method).

        Ray is defined as: (M, N) = t * (M_Ed, N_Ed), t >= 0

        If intersection scale is t_cap, then:
            (M_Rd, N_Rd) = t_cap * (M_Ed, N_Ed)
            utilization = 1 / t_cap
        """
        diagram = self.generate_diagram(n_points=n_points)
        pts = [(p.M, p.N) for p in diagram]
        if len(pts) < 3:
            return (None, None, False, float("inf"))

        # Special case: origin (no load)
        if abs(M_Ed) < 1e-18 and abs(N_Ed) < 1e-18:
            return (0.0, 0.0, True, 0.0)

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

        if len(intersections) == 0:
            return (None, None, False, float("inf"))

        # Take maximum t (farthest intersection from origin)
        # This is correct for convex curves; for self-intersecting curves, could be unconservative
        max_t = max(intersections)

        # Sanity check: warn if multiple intersections found (possible self-intersection)
        if len(intersections) > 2:
            # More than 2 intersections suggests self-intersection or numerical issues
            # Note: exactly 2 intersections can occur for tangent rays (entry/exit at same point)
            import warnings
            warnings.warn(
                f"Ray intersection found {len(intersections)} intersections (expected 1-2). "
                f"Curve may self-intersect. Using max_t={max_t:.4f}. "
                f"Consider increasing n_points or checking diagram quality.",
                stacklevel=2
            )

        if max_t <= 1e-12:
            return (None, None, False, float("inf"))

        M_Rd = max_t * float(M_Ed)
        N_Rd = max_t * float(N_Ed)

        utilization = 1.0 / max_t
        is_safe = utilization <= 1.0

        return (float(N_Rd), float(M_Rd), bool(is_safe), float(utilization))


    def get_utilization_vector(
        self,
        N_Ed: float,
        M_Ed: float,
        n_points: int = 120,
    ) -> Tuple[bool, float]:
        """Convenience wrapper returning (is_safe, utilization) using vector method."""
        _, _, is_safe, util = self.get_capacity_vector(N_Ed=N_Ed, M_Ed=M_Ed, n_points=n_points)
        return (bool(is_safe), float(util))


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
        diagram = self.generate_diagram(n_points=n_points)  # should be closed already
        if len(diagram) < 4:
            return (None, None, None)

        pts = [(float(p.M), float(p.N)) for p in diagram]

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
        pts = self.generate_diagram(n_points=n_points)
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
        points = self.generate_diagram(n_points=n_points)
        data: Dict[str, Any] = {"diagram_points": [p.to_dict() for p in points]}

        if include_metadata:
            data["metadata"] = {
                "section_name": self.section.section_name,
                "concrete_grade": self.concrete.grade,
                "concrete_fck": self.concrete.f_ck,
                "concrete_fcd": self.concrete.f_cd,
                "n_rebar_groups": len(self.section.rebar_groups),
                "n_fibers": self.mesh.total_fibers,
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
        points = self.generate_diagram(n_points=n_points)

        file_path = Path(file_path)
        with open(file_path, "w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "N_kN",
                "M_kNm",
                "neutral_axis_depth_mm",
                "compression_from_bottom",
                "max_concrete_strain",
                "max_steel_strain",
            ] if include_strains else ["N_kN", "M_kNm"]

            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for p in points:
                row = p.to_dict()
                if not include_strains:
                    row = {"N_kN": row["N_kN"], "M_kNm": row["M_kNm"]}
                writer.writerow(row)


    def to_dict(
        self,
        n_points: int = 120,
        include_metadata: bool = True,
    ) -> Dict[str, Any]:
        points = self.generate_diagram(n_points=n_points)
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
                "n_fibers": self.mesh.total_fibers,
                "concrete_model": type(self.concrete_model).__name__,
                "steel_models": [type(sm).__name__ for sm in self.steel_models],
                "tension_stiffening": self.tension_stiffening,
                "confined_concrete": self.confined_concrete,
            }
        return data


    # ----------------------------
    # Plotting (kept for encapsulated UX)
    # ----------------------------

    def plot(
        self,
        *,
        load_points: Optional[List[Dict[str, Any]]] = None,
        show_vectors: bool = False,
        show_metadata: bool = True,
        n_points: int = 120,
        save_path: Optional[str | Path] = None,
        show: bool = True,
        title: Optional[str] = None,
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

        Returns:
            Plotly Figure object
        """
        try:
            import plotly.graph_objects as go
        except ImportError as e:
            raise ImportError(
                "Plotly is required for plotting. Install with: pip install plotly"
            ) from e

        # Generate diagram
        diagram_points = self.generate_diagram(n_points=n_points)
        M_curve = [p.M for p in diagram_points]
        N_curve = [p.N for p in diagram_points]

        fig = go.Figure()

        # Capacity curve
        if show_metadata:
            hover = (
                "M: %{x:.3g} kN·m<br>"
                "N: %{y:.3g} kN<br>"
                "<extra></extra>"
            )
        else:
            hover = "M: %{x:.3g}<br>N: %{y:.3g}<extra></extra>"

        fig.add_trace(go.Scatter(
            x=M_curve,
            y=N_curve,
            mode="lines",
            name="M-N Capacity",
            line=dict(color="black", width=2),
            hovertemplate=hover,
        ))

        # Origin
        fig.add_trace(go.Scatter(
            x=[0.0],
            y=[0.0],
            mode="markers",
            name="Origin",
            marker=dict(color="black", size=4, symbol="circle"),
            hovertemplate="Origin<extra></extra>",
        ))

        # Load points
        if load_points:
            for idx, lp in enumerate(load_points):
                N_Ed = float(lp.get("N_Ed", 0.0))
                M_Ed = float(lp.get("M_Ed", 0.0))
                name_lp = str(lp.get("name", f"Load Case {idx + 1}"))

                N_Rd, M_Rd, is_safe, utilization = self.get_capacity_vector(
                    N_Ed=N_Ed, M_Ed=M_Ed, n_points=n_points
                )

                if utilization <= 0.8:
                    color = "green"
                elif utilization <= 1.0:
                    color = "orange"
                else:
                    color = "red"

                if show_metadata:
                    hover_text = (
                        f"<b>{name_lp}</b><br>"
                        f"N_Ed: {N_Ed:.3g} kN<br>"
                        f"M_Ed: {M_Ed:.3g} kN·m<br>"
                    )
                    if N_Rd is not None and M_Rd is not None:
                        hover_text += (
                            f"N_Rd: {N_Rd:.3g} kN<br>"
                            f"M_Rd: {M_Rd:.3g} kN·m<br>"
                            f"Utilization: {utilization:.1%}<br>"
                            f"Status: {'✓ PASS' if is_safe else '✗ FAIL'}"
                        )
                    else:
                        hover_text += "Status: Outside boundary"
                else:
                    hover_text = name_lp

                fig.add_trace(go.Scatter(
                    x=[M_Ed],
                    y=[N_Ed],
                    mode="markers",
                    name=name_lp,
                    marker=dict(color=color, size=7, symbol="circle", line=dict(color="black", width=1)),
                    hovertemplate=hover_text + "<extra></extra>",
                    showlegend=True,
                ))

                if show_vectors and (N_Rd is not None) and (M_Rd is not None):
                    # Origin -> load
                    fig.add_trace(go.Scatter(
                        x=[0.0, M_Ed],
                        y=[0.0, N_Ed],
                        mode="lines",
                        line=dict(color=color, width=1.5, dash="solid"),
                        showlegend=False,
                        hoverinfo="skip",
                    ))
                    # Load -> capacity point
                    fig.add_trace(go.Scatter(
                        x=[M_Ed, M_Rd],
                        y=[N_Ed, N_Rd],
                        mode="lines",
                        line=dict(color=color, width=1.5, dash="dash"),
                        showlegend=False,
                        hoverinfo="skip",
                    ))

        plot_title = title if title else "M-N Interaction Diagram"
        fig.update_layout(
            title=dict(text=plot_title, font=dict(size=16, color="black")),
            xaxis_title="Moment M (kN·m)",
            yaxis_title="Axial Force N (kN)",
            hovermode="closest",
            template="plotly_white",
            showlegend=True,
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
            width=900,
            height=700,
        )
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")

        # ---- Force axes to include everything (capacity + loads + vectors) ----
        # Collect all x/y values you actually plotted
        xs = list(M_curve) + [0.0]
        ys = list(N_curve) + [0.0]

        if load_points:
            for lp in load_points:
                xs.append(float(lp.get("M_Ed", 0.0)))
                ys.append(float(lp.get("N_Ed", 0.0)))

                if show_vectors:
                    N_Rd, M_Rd, _, _ = self.get_capacity_vector(
                        N_Ed=float(lp.get("N_Ed", 0.0)),
                        M_Ed=float(lp.get("M_Ed", 0.0)),
                        n_points=n_points
                    )
                    if (M_Rd is not None) and (N_Rd is not None):
                        xs.append(float(M_Rd))
                        ys.append(float(N_Rd))

        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)

        # Pad by 5% (fallback to 1.0 if range is tiny)
        xpad = 0.05 * (xmax - xmin) if xmax > xmin else 1.0
        ypad = 0.05 * (ymax - ymin) if ymax > ymin else 1.0

        fig.update_xaxes(range=[xmin - xpad, xmax + xpad], autorange=False)
        fig.update_yaxes(range=[ymin - ypad, ymax + ypad], autorange=False)

        if save_path:
            fig.write_html(str(save_path))

        if show:
            fig.show()

        return fig

    def __repr__(self) -> str:
        return (
            f"MNInteractionDiagram("
            f"section={self.section.section_name}, "
            f"concrete={self.concrete.grade}, "
            f"fibers={self.mesh.total_fibers}, "
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
        **kwargs: Additional arguments passed to MNInteractionDiagram

    Returns:
        MNInteractionDiagram instance
    """
    return MNInteractionDiagram(section=section, concrete=concrete, **kwargs)
