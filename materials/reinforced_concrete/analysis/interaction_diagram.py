"""
M-N interaction diagram generator using fiber-based strain compatibility.

Implements EC2 ultimate limit state analysis for combined axial force and bending
about a single axis (major axis in this 2D implementation).

Key modelling choices / conventions
-----------------------------------
Sign convention (global):
- Axial force N > 0 => compression
- Axial force N < 0 => tension
- Concrete constitutive models expect compression strain > 0 and return compression stress > 0
- Steel constitutive models return stress with the same sign as strain (tension positive)

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
from typing import Any, Dict, List, Literal, Optional, Tuple

import csv
import json
import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field
from scipy.optimize import root_scalar

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
    """Convert numpy scalars cleanly to Python float."""
    return float(x)  # raises if not convertible (good)


def _unit_vector(m: float, n: float) -> Tuple[float, float]:
    """Return unit vector in direction (m, n); if zero, returns (0,0)."""
    norm = (m * m + n * n) ** 0.5
    if norm == 0.0:
        return (0.0, 0.0)
    return (m / norm, n / norm)


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
            use_characteristic=False,  # design (f_cd etc.)
        )

        if len(section.rebar_groups) == 0:
            raise ValueError("Section must have at least one rebar group")

        # Steel models per group (support different grades)
        self.steel_models = [
            create_steel_stress_strain(
                steel=g.rebar,
                branch_type=steel_branch_type,
                use_characteristic=False,  # design (f_yd)
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
        _, y, area, _, _ = self.mesh.get_fiber_arrays()

        # Axial force: sum(σ * A) in N, convert to kN
        N = np.sum(stresses * area) / 1000.0

        # Moment about section centroid (single axis about y-offset)
        if use_section_centroid:
            _, cy = self.section.get_centroid()
        else:
            cy = float(np.sum(y * area) / np.sum(area))

        y_offset = y - cy
        M = np.sum(stresses * area * y_offset) / 1_000_000.0  # N·mm -> kN·m

        return (_as_float(N), _as_float(M))

    def _strain_field(
        self,
        neutral_axis_depth: float,
        max_concrete_strain: float,
        *,
        compression_from_bottom: bool,
    ) -> npt.NDArray[np.float64]:
        """
        Build the strain field (compression positive) for all fibers.

        For NA depth > 0:
            ε(z) = ε_c,max * (NA - z) / NA
        where z is distance from compression face.

        This ensures:
        - ε = ε_c,max at the compression face (z=0)
        - ε = 0 at z=NA (neutral axis)
        - ε < 0 (tension) for z > NA
        """
        _, y, _, _, _ = self.mesh.get_fiber_arrays()

        if compression_from_bottom:
            z = y - self.section_bottom
        else:
            z = self.section_top - y

        na = float(neutral_axis_depth)
        if na <= 0.0:
            raise ValueError("neutral_axis_depth must be > 0 for this solver branch")

        return max_concrete_strain * (na - z) / na

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

    def calculate_point(
        self,
        neutral_axis_depth: float,
        max_concrete_strain: Optional[float] = None,
        *,
        compression_from_bottom: bool = False,
    ) -> InteractionPoint:
        """
        Calculate a single point on the interaction diagram for a given neutral axis depth.

        Args:
            neutral_axis_depth: NA depth from compression face (mm). Must be > 0 in this implementation.
            max_concrete_strain: Extreme compression strain at the compression face. Defaults to concrete model ε_cu.
            compression_from_bottom: If True, compression face is bottom; else top.

        Returns:
            InteractionPoint with N, M and strain metadata.
        """
        if max_concrete_strain is None:
            max_concrete_strain = float(self.concrete_model.get_ultimate_strain())

        if neutral_axis_depth <= 0.0:
            raise ValueError("neutral_axis_depth must be > 0 (solver uses explicit tension point instead)")

        # Fiber arrays
        _, _, _, material_type, material_index = self.mesh.get_fiber_arrays()

        # Ensure robust dtype for comparisons (avoids numpy object surprises)
        material_type = material_type.astype("U8", copy=False)

        # Strain field
        strains = self._strain_field(
            neutral_axis_depth=neutral_axis_depth,
            max_concrete_strain=max_concrete_strain,
            compression_from_bottom=compression_from_bottom,
        )

        stresses = np.zeros_like(strains)

        # Concrete stresses
        concrete_mask = material_type == "concrete"
        if np.any(concrete_mask):
            conc_strains = strains[concrete_mask]
            stresses[concrete_mask] = self._concrete_stress_with_options(conc_strains)

        # Steel stresses (per group)
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

        # Resultants
        N, M = self.calculate_section_forces(stresses)

        max_conc = float(np.max(strains[concrete_mask])) if np.any(concrete_mask) else 0.0
        max_steel = float(np.max(np.abs(strains[steel_mask]))) if np.any(steel_mask) else 0.0

        return InteractionPoint(
            N=N,
            M=M,
            neutral_axis_depth=float(neutral_axis_depth),
            compression_from_bottom=bool(compression_from_bottom),
            max_concrete_strain=max_conc,
            max_steel_strain=max_steel,
        )

    # ----------------------------
    # Closing points (consistent fiber evaluation)
    # ----------------------------

    def _pure_compression_point(self) -> InteractionPoint:
        """
        Pure compression closing point computed by fiber integration.

        Uses uniform compressive strain = ε_cu everywhere (no curvature).
        This is consistent with the fiber system and avoids analytic area inconsistencies.
        """
        eps_cu = float(self.concrete_model.get_ultimate_strain())

        _, y, _, material_type, material_index = self.mesh.get_fiber_arrays()
        material_type = material_type.astype("U8", copy=False)

        strains = np.full_like(y, eps_cu, dtype=float)
        stresses = np.zeros_like(strains)

        concrete_mask = material_type == "concrete"
        if np.any(concrete_mask):
            stresses[concrete_mask] = self._concrete_stress_with_options(strains[concrete_mask])

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

        return InteractionPoint(
            N=N,
            M=M,
            neutral_axis_depth=self.section_height * 1e6,  # effectively "infinite" NA
            compression_from_bottom=False,
            max_concrete_strain=eps_cu,
            max_steel_strain=float(np.max(np.abs(strains[steel_mask]))) if np.any(steel_mask) else 0.0,
        )

    def _pure_tension_point(self) -> InteractionPoint:
        """
        Pure tension closing point computed by fiber integration.

        Concrete in tension:
        - returns 0 if tension_stiffening=False
        - returns some negative stress if tension_stiffening=True (by design choice)

        Steel:
        - driven to yield by applying a sufficiently large uniform tensile strain.
        """
        eps_y_max = max(float(sm.epsilon_y) for sm in self.steel_models)
        eps_t = -10.0 * eps_y_max if eps_y_max > 0 else -0.01

        _, y, _, material_type, material_index = self.mesh.get_fiber_arrays()
        material_type = material_type.astype("U8", copy=False)

        strains = np.full_like(y, eps_t, dtype=float)
        stresses = np.zeros_like(strains)

        # Concrete
        concrete_mask = material_type == "concrete"
        if np.any(concrete_mask):
            stresses[concrete_mask] = self._concrete_stress_with_options(strains[concrete_mask])

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

        return InteractionPoint(
            N=N,
            M=M,
            neutral_axis_depth=-self.section_height * 1e6,  # effectively "infinite" NA on tension side
            compression_from_bottom=False,
            max_concrete_strain=float(np.max(strains[concrete_mask])) if np.any(concrete_mask) else 0.0,
            max_steel_strain=float(np.max(np.abs(strains[steel_mask]))) if np.any(steel_mask) else 0.0,
        )

    # ----------------------------
    # NA sampling (more sensible)
    # ----------------------------

    def _na_depths(self, n: int) -> npt.NDArray[np.float64]:
        h = float(self.section_height)

        na_max = 50.0 * h
        na_mid_hi = 2.5 * h
        na_mid = 0.60 * h
        na_bal_lo = 0.12 * h
        na_min = max(0.02 * h, 1.0)

        n = max(int(n), 30)

        n1 = max(8, n // 5)
        n2 = max(10, n // 4)
        n3 = max(10, n // 3)
        n4 = max(8, n - (n1 + n2 + n3))

        # Key change: seg1 endpoint=False so na_mid_hi isn't duplicated
        seg1 = np.logspace(np.log10(na_max), np.log10(na_mid_hi), num=n1, endpoint=False)
        seg2 = np.linspace(na_mid_hi, na_mid, num=n2, endpoint=False)
        seg3 = np.linspace(na_mid, na_bal_lo, num=n3, endpoint=False)
        seg4 = np.logspace(np.log10(na_bal_lo), np.log10(na_min), num=n4, endpoint=True)

        na = np.concatenate([seg1, seg2, seg3, seg4]).astype(np.float64)
        na = np.maximum(na, 1e-6)

        # Remove near-duplicates with tolerance (preserves current order)
        # (tolerance in mm; choose small but > float noise)
        tol = 1e-9 * h  # e.g. 5e-7 mm for h=500
        out = [na[0]]
        for v in na[1:]:
            if abs(v - out[-1]) > tol:
                out.append(v)

        # You want descending from large->small
        out = np.array(out, dtype=np.float64)
        out = out[::-1]
        return out

    # ----------------------------
    # Diagram generation
    # ----------------------------

    def generate_diagram(
        self,
        n_points: int = 120,
        include_tension: bool = True,
    ) -> List[InteractionPoint]:
        """
        Generate a closed M-N interaction envelope.

        Args:
            n_points: Total sampling density. Internally split across top/bottom branches.
            include_tension: Include pure tension closing point (recommended True for closed loop).

        Returns:
            Ordered closed loop of InteractionPoint objects.
        """
        n_branch = max(30, n_points // 2)
        na_depths = self._na_depths(n_branch)

        p_comp = self._pure_compression_point()
        p_tens = self._pure_tension_point() if include_tension else None

        top_branch: List[InteractionPoint] = [
            self.calculate_point(float(na), compression_from_bottom=False)
            for na in na_depths
        ]

        bottom_branch: List[InteractionPoint] = [
            self.calculate_point(float(na), compression_from_bottom=True)
            for na in na_depths
        ]

        envelope: List[InteractionPoint] = []
        envelope.append(p_comp)
        envelope.extend(top_branch)

        if p_tens is not None:
            envelope.append(p_tens)

        envelope.extend(reversed(bottom_branch))
        envelope.append(p_comp)  # explicit closure

        return envelope

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
        diagram = self.generate_diagram(n_points=n_points, include_tension=True)
        pts = [(p.M, p.N) for p in diagram]
        if len(pts) < 3:
            return (None, None, False, float("inf"))

        # Special case: origin (no load)
        if abs(M_Ed) < 1e-12 and abs(N_Ed) < 1e-12:
            return (0.0, 0.0, True, 0.0)

        ray_dir = (float(M_Ed), float(N_Ed))  # IMPORTANT: do NOT normalize

        max_t = 0.0
        for i in range(len(pts) - 1):
            t = _ray_segment_intersection_alpha(ray_dir, pts[i], pts[i + 1], tol=1e-12)
            if t is not None:
                max_t = max(max_t, t)

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

    # ----------------------------
    # Balanced point (improved guards)
    # ----------------------------

    def find_balanced_point(
        self,
        max_concrete_strain: Optional[float] = None,
    ) -> Tuple[InteractionPoint, float]:
        """
        Find balanced failure point (ε_c at extreme compression = ε_cu, tensile steel reaches ε_y).

        Returns:
            (balanced_point, neutral_axis_depth_mm)
        """
        if max_concrete_strain is None:
            max_concrete_strain = float(self.concrete_model.get_ultimate_strain())

        _, y, _, material_type, material_index = self.mesh.get_fiber_arrays()
        material_type = material_type.astype("U8", copy=False)

        steel_mask = material_type == "steel"
        if not np.any(steel_mask):
            raise ValueError("Cannot find balanced point - no steel fibers")

        steel_y = y[steel_mask]
        steel_idx = material_index[steel_mask]

        # For top compression, tension steel is near bottom => min y
        i_min = int(np.argmin(steel_y))
        y_tension = float(steel_y[i_min])
        group_idx = int(steel_idx[i_min])

        eps_y = float(self.steel_models[group_idx].epsilon_y)
        y_from_top = self.section_top - y_tension

        x_bal_est = max_concrete_strain * y_from_top / (max_concrete_strain + eps_y)

        def objective(na: float) -> float:
            if na <= 0.0:
                return 1e9
            eps_s = max_concrete_strain * (na - y_from_top) / na
            return abs(eps_s) - eps_y

        lo = max(0.02 * self.section_height, 1e-3)
        hi = 5.0 * self.section_height

        try:
            sol = root_scalar(objective, bracket=[lo, hi], method="brentq", xtol=1e-3)
            na_bal = float(sol.root) if sol.converged else float(x_bal_est)
        except Exception:
            na_bal = float(x_bal_est)

        p = self.calculate_point(na_bal, max_concrete_strain=max_concrete_strain, compression_from_bottom=False)
        return (p, na_bal)

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
        diagram_points = self.generate_diagram(n_points=n_points, include_tension=True)
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
