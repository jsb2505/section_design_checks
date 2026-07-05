"""
Biaxial M-M-N interaction surface generator using fibre-based strain compatibility.

Implements EC2 ultimate limit state analysis for combined axial force and biaxial bending.

Key modelling choices / conventions
-----------------------------------
Sign convention (global):
- Axial force N > 0 => compression
- Axial force N < 0 => tension
- Strain: compression positive, tension negative (consistent with concrete convention)
- Concrete constitutive models expect compression strain > 0 and return compression stress > 0
- Steel constitutive models return stress with the same sign as strain

Strain compatibility:
- Plane sections remain plane.
- A neutral axis depth and angle are assumed.
- The EC2 pivot method determines strain limits (Zone A/B/C).

The surface represents all combinations of axial force (N) and biaxial moments (My, Mz)
that bring the section to its ultimate limit state per EC2.

Axis Convention (3D FEA standard):
- x: longitudinal axis (along member)
- y: horizontal axis in cross-section (minor axis, width)
- z: vertical axis in cross-section (major axis, height)
- My: moment about y-axis (major axis bending, from z-forces)
- Mz: moment about z-axis (minor axis bending, from y-forces)
"""

from __future__ import annotations

from typing import List, Optional, Dict, Any, Tuple
import json
import csv
from pathlib import Path
import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, Field, ConfigDict
from scipy.optimize import brentq
from scipy.spatial import ConvexHull

from materials.utils.helpers import as_float
from materials.core.units import ForceUnit, MomentUnit, to_kn, to_knm

from materials.reinforced_concrete.geometry import RCSection, FibreMesh
from materials.reinforced_concrete.constitutive import (
    create_concrete_stress_strain,
    create_steel_stress_strain,
    SteelModelType,
    ConcreteModelType,
)
from materials.reinforced_concrete.materials import ConcreteMaterial


class BiaxialInteractionPoint(BaseModel):
    """
    Single point on biaxial M-M-N interaction surface.

    Uses 3D FEA axis convention:
    - x: longitudinal axis (along member)
    - y: horizontal axis in cross-section (minor axis, width)
    - z: vertical axis in cross-section (major axis, height)
    - My: moment about y-axis (major axis bending, from z-forces)
    - Mz: moment about z-axis (minor axis bending, from y-forces)
    """

    model_config = ConfigDict(frozen=True)

    N: float = Field(..., description="Axial force in kN (positive = compression)")
    My: float = Field(..., description="Moment about y-axis (major axis) in kN·m")
    Mz: float = Field(..., description="Moment about z-axis (minor axis) in kN·m")
    neutral_axis_depth: float = Field(..., description="Neutral axis depth from centroid (mm)")
    neutral_axis_angle: float = Field(..., description="Neutral axis angle from y-axis (degrees)")
    max_concrete_strain: float = Field(..., description="Maximum concrete strain")
    max_steel_strain: float = Field(..., description="Maximum steel strain")

    def __repr__(self) -> str:
        return f"BiaxialPoint(N={self.N:.1f} kN, My={self.My:.1f} kN·m, Mz={self.Mz:.1f} kN·m)"

    def to_dict(self) -> Dict[str, Any]:
        """Export biaxial interaction point to dictionary."""
        return {
            "N_kN": self.N,
            "My_kNm": self.My,
            "Mz_kNm": self.Mz,
            "neutral_axis_depth_mm": self.neutral_axis_depth,
            "neutral_axis_angle_deg": self.neutral_axis_angle,
            "max_concrete_strain": self.max_concrete_strain,
            "max_steel_strain": self.max_steel_strain,
        }


class BiaxialMNInteractionSurface:
    """
    Biaxial M-M-N interaction surface generator using fibre-based strain compatibility.

    Method:
    1. Assume a neutral axis depth and angle
    2. Calculate strain distribution (plane sections remain plane)
    3. Strains perpendicular to the neutral axis
    4. Get stresses from constitutive models
    5. Integrate forces over fibres
    6. Result is one (N, My, Mz) point on the surface
    7. Repeat for different neutral axis depths and angles
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
    ):
        """
        Initialize biaxial M-M-N surface generator.

        Args:
            section: RC section with reinforcement
            concrete: Concrete material properties
            concrete_model_type: Stress-strain model for concrete
            steel_model_type: Stress-strain model for steel
            n_fibres_width: Fibre mesh resolution (width)
            n_fibres_height: Fibre mesh resolution (height)
            tension_stiffening: If True, include post-cracking concrete tension contribution
            use_characteristic: If True, use characteristic strengths instead of design
            use_accidental: If True, use accidental combination factors
            confined_concrete: If True, apply Mander-style confinement model
            confinement_rho_s: Volumetric ratio of transverse reinforcement (required if confined_concrete=True)
            confinement_f_yh: Characteristic yield strength of transverse steel (MPa)
            ignore_compression_steel: If True, zero out compression steel contribution
        """
        self.section = section
        self.concrete = concrete
        self.tension_stiffening = tension_stiffening
        self.confined_concrete = confined_concrete
        self.ignore_compression_steel = ignore_compression_steel
        self.confinement_rho_s = confinement_rho_s
        self.confinement_f_yh = confinement_f_yh

        # Create constitutive models
        self.concrete_model = create_concrete_stress_strain(
            concrete=concrete,
            model_type=concrete_model_type,
            use_characteristic=use_characteristic,
            use_accidental=use_accidental,
        )

        if len(section.rebar_groups) == 0:
            raise ValueError("Section must have at least one rebar group")

        # Steel models per group (support different grades)
        self.steel_models = [
            create_steel_stress_strain(
                steel=g.rebar,
                branch_type=steel_model_type,
                use_characteristic=use_characteristic,
                use_accidental=use_accidental,
            )
            for g in section.rebar_groups
        ]

        # Confined concrete parameter checks
        _model_has_ec2_confinement = getattr(self.concrete_model, 'is_ec2_confined', False)

        if self.confined_concrete:
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
                self.confinement_f_yh = section.rebar_groups[0].rebar.f_yk

            if self.confinement_f_yh <= 0:
                raise ValueError(f"confinement_f_yh must be > 0, got {self.confinement_f_yh}")

        # Generate fibre mesh
        self.mesh = FibreMesh(
            section=section,
            n_fibres_width=n_fibres_width,
            n_fibres_height=n_fibres_height,
            exclude_steel_area=True,
        )

        # Cache fibre arrays for performance (avoid repeated allocation/copy)
        (
            self._fibre_x,
            self._fibre_y,
            self._fibre_area,
            self._fibre_mat,
            self._fibre_mi,
            self._fibre_i,
            self._fibre_j,
        ) = self.mesh.get_fibre_arrays()

        self._fibre_mat = self._fibre_mat.astype("U8", copy=False)

        # Get section properties
        self.section_centroid_x, self.section_centroid_y = section.get_centroid()
        min_x, min_y, max_x, max_y = section.get_bounding_box()
        self.section_width = max_x - min_x
        self.section_height = max_y - min_y

        # Multi-tier cache for generated surfaces
        self._dense_surface_points: Optional[tuple[BiaxialInteractionPoint, ...]] = None
        self._dense_params: Optional[tuple[int, int]] = None
        self._surface_cache: dict[tuple[int, int], tuple[BiaxialInteractionPoint, ...]] = {}
        self._hull_cache: dict[tuple[int, int], ConvexHull] = {}

    # ----------------------------
    # Tension limit (derived from steel models)
    # ----------------------------

    def _eps_tension_limit(self) -> float:
        """
        Choose a tensile strain magnitude for the strain limit.

        - If any steel model has finite ultimate strain (inclined), use max(ε_ud)
        - If all are horizontal (infinite), use a large multiple of yield strain.
        """
        ultimates = [float(sm.get_ultimate_strain()) for sm in self.steel_models]
        finite = [u for u in ultimates if np.isfinite(u)]
        if finite:
            return float(max(finite))

        eps_y_max = max(float(sm.epsilon_y) for sm in self.steel_models)
        return float(max(10.0 * eps_y_max, 0.01))

    # ----------------------------
    # Concrete stress with options
    # ----------------------------

    def _concrete_stress_with_options(
        self,
        concrete_strains: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """
        Compute concrete stresses for the given concrete strains (compression positive),
        applying optional confinement (compression only) and optional tension stiffening (tension only).
        """
        concrete_stresses = self.concrete_model.get_stress_array(concrete_strains)

        # Confined concrete (compression only)
        if self.confined_concrete:
            assert self.confinement_rho_s is not None
            assert self.confinement_f_yh is not None

            rho_s = float(self.confinement_rho_s)
            f_yh_k = float(self.confinement_f_yh)

            comp_mask = concrete_strains > 0.0
            if np.any(comp_mask):
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

        # Tension stiffening (tension only)
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

    # ----------------------------
    # Convex hull utilities
    # ----------------------------

    def _build_convex_hull(self, surface_points: tuple[BiaxialInteractionPoint, ...]) -> ConvexHull:
        """Build a convex hull in (N, My, Mz) space from surface points."""
        pts = np.array([[p.N, p.My, p.Mz] for p in surface_points], dtype=float)
        if pts.shape[0] < 4:
            raise ValueError("At least 4 points are required to build a convex hull")
        return ConvexHull(pts)

    def _get_hull(self, n_angles: int, n_axial_levels: int) -> ConvexHull:
        """Get or build convex hull for given resolution."""
        key = (n_angles, n_axial_levels)
        if key not in self._hull_cache:
            pts = self.generate_surface_pivot(n_angles=n_angles, n_axial_levels=n_axial_levels)
            self._hull_cache[key] = self._build_convex_hull(pts)
        return self._hull_cache[key]

    # ----------------------------
    # Capacity checks
    # ----------------------------

    def get_utilization_vector(
        self,
        N_Ed: float,
        My_Ed: float,
        Mz_Ed: float,
        surface_points: Optional[List[BiaxialInteractionPoint]] = None,
        hull: Optional[ConvexHull] = None,
        n_angles: int = 72,
        n_axial_levels: int = 30,
    ) -> Tuple[bool, float]:
        """
        Check capacity using exact ray-to-surface intersection via convex hull.

        Returns:
            (is_safe, utilization) tuple
        """
        _, _, _, is_safe, utilization = self.get_capacity_vector_exact(
            N_Ed=N_Ed,
            My_Ed=My_Ed,
            Mz_Ed=Mz_Ed,
            surface_points=surface_points,
            hull=hull,
            n_angles=n_angles,
            n_axial_levels=n_axial_levels,
        )
        return (bool(is_safe), float(utilization))

    def get_capacity_vector_exact(
        self,
        N_Ed: float,
        My_Ed: float,
        Mz_Ed: float,
        surface_points: Optional[List[BiaxialInteractionPoint]] = None,
        hull: Optional[ConvexHull] = None,
        n_angles: int = 72,
        n_axial_levels: int = 30,
    ) -> Tuple[Optional[float], Optional[float], Optional[float], bool, float]:
        """
        Get exact capacity point using convex hull intersection (ray-plane with triangles).
        """
        # Special case: origin point (no load)
        if abs(N_Ed) < 1e-6 and abs(My_Ed) < 1e-6 and abs(Mz_Ed) < 1e-6:
            return (0.0, 0.0, 0.0, True, 0.0)

        # Use provided hull or generate/cached hull
        try:
            if hull is not None:
                hull_obj = hull
            elif surface_points is not None:
                hull_obj = self._build_convex_hull(tuple(surface_points))
            else:
                hull_obj = self._get_hull(n_angles, n_axial_levels)
        except Exception:
            return (None, None, None, False, float("inf"))

        load_vec = np.array([N_Ed, My_Ed, Mz_Ed], dtype=float)
        load_mag = float(np.linalg.norm(load_vec))

        if load_mag < 1e-12:
            return (0.0, 0.0, 0.0, True, 0.0)

        ray_dir = load_vec / load_mag

        equations = hull_obj.equations  # shape (n_facets, 4) -> normals | offsets
        normals = equations[:, :3]
        offsets = equations[:, 3]
        denom = normals @ ray_dir

        forward_mask = denom > 1e-12
        if not np.any(forward_mask):
            return (None, None, None, False, float("inf"))

        t_candidates = -offsets[forward_mask] / denom[forward_mask]
        t_candidates = t_candidates[t_candidates > 1e-12]

        if t_candidates.size == 0:
            return (None, None, None, False, float("inf"))

        t_min = float(np.min(t_candidates))

        capacity_vec = t_min * ray_dir
        utilization = load_mag / t_min

        return (
            float(capacity_vec[0]),
            float(capacity_vec[1]),
            float(capacity_vec[2]),
            bool(utilization <= 1.0 + 1e-6),
            float(utilization),
        )

    def get_capacity_vector(
        self,
        N_Ed: float,
        My_Ed: float,
        Mz_Ed: float,
        surface_points: Optional[List[BiaxialInteractionPoint]] = None,
        hull: Optional[ConvexHull] = None,
        n_angles: int = 72,
        n_axial_levels: int = 30,
    ) -> Tuple[Optional[float], Optional[float], Optional[float], bool, float]:
        """
        Backwards-compatible wrapper for get_capacity_vector_exact.
        """
        return self.get_capacity_vector_exact(
            N_Ed=N_Ed,
            My_Ed=My_Ed,
            Mz_Ed=Mz_Ed,
            surface_points=surface_points,
            hull=hull,
            n_angles=n_angles,
            n_axial_levels=n_axial_levels,
        )

    # ========================================================================
    # Surface generation - EC2 Pivot Method
    # ========================================================================

    def calculate_axial_limits(self) -> tuple[float, float]:
        """
        Calculate the absolute theoretical N_min (pure tension) and N_max (pure compression).

        Returns:
            Tuple of (N_min, N_max) in kN
        """
        area = self._fibre_area
        mat_type = self._fibre_mat
        mat_idx = self._fibre_mi

        eps_cu2 = self.concrete.epsilon_cu2
        eps_ud = self._eps_tension_limit()

        # 1. PURE TENSION (N_min)
        n_min = 0.0
        steel_mask = (mat_type == 'steel')
        if np.any(steel_mask):
            n_steel_tension = 0.0
            unique_steel_groups = np.unique(mat_idx[steel_mask])
            for g_idx in unique_steel_groups:
                group_mask = steel_mask & (mat_idx == g_idx)
                stress_tension = self.steel_models[int(g_idx)].get_stress(-eps_ud)
                n_steel_tension += stress_tension * np.sum(area[group_mask])
            n_min = as_float(to_kn(n_steel_tension, ForceUnit.N))

        # 2. PURE COMPRESSION (N_max)
        n_max = 0.0
        conc_mask = (mat_type == 'concrete')
        if np.any(conc_mask):
            stress_c = self.concrete_model.get_stress(eps_cu2)
            n_max += to_kn(stress_c * np.sum(area[conc_mask]), ForceUnit.N)

        if np.any(steel_mask):
            n_steel_compression = 0.0
            unique_steel_groups = np.unique(mat_idx[steel_mask])
            for g_idx in unique_steel_groups:
                group_mask = steel_mask & (mat_idx == g_idx)
                stress_comp = self.steel_models[int(g_idx)].get_stress(eps_cu2)
                n_steel_compression += stress_comp * np.sum(area[group_mask])
            n_max += to_kn(n_steel_compression, ForceUnit.N)

        return (as_float(n_min), as_float(n_max))

    def _get_strain_at_y_pivot(
        self,
        y: float,
        na_depth: float,
        y_max: float,
        y_min: float,
        h: float,
        rebar_y_min: float,
        d_eff: float,
    ) -> float:
        """
        Calculate strain at coordinate y using EC2 Pivot Method with balanced depth.

        Three zones per EC2:
        - Zone A: Tension failure (pivot at extreme rebar ε_ud) when na_depth <= x_bal
        - Zone B: Bending failure (pivot at extreme concrete fibre ε_cu2) when x_bal < na_depth <= h
        - Zone C: Compression failure (pivot at depth z_p with ε_c2) when na_depth > h
        """
        eps_cu2 = self.concrete.epsilon_cu2
        eps_c2 = self.concrete.epsilon_c2
        eps_ud = self._eps_tension_limit()

        x_bal = (eps_cu2 / (eps_cu2 + eps_ud)) * d_eff
        z_p = (1.0 - eps_c2 / eps_cu2) * h
        y_na = y_max - na_depth

        # ZONE A: Tension Failure
        if na_depth <= x_bal:
            slope = -eps_ud / (rebar_y_min - y_na)
            return slope * (y - y_na)

        # ZONE B: Bending Failure
        elif na_depth <= h:
            slope = eps_cu2 / na_depth
            return slope * (y - y_na)

        # ZONE C: Compression Failure
        else:
            slope = eps_c2 / (na_depth - z_p)
            return slope * (y - y_na)

    def calculate_point_pivot(
        self,
        na_depth: float,
        neutral_axis_angle: float = 0.0,
    ) -> BiaxialInteractionPoint:
        """
        Calculate point on M-M-N surface using PIVOT METHOD (vectorized).

        Uses the EC2 pivot method to ensure strains always touch ultimate limits.

        Args:
            na_depth: Neutral axis depth from top fibre (mm, positive = deeper)
            neutral_axis_angle: Angle of neutral axis from horizontal (degrees)

        Returns:
            Point on the failure surface
        """
        # Use cached fibre arrays
        x = self._fibre_x
        y = self._fibre_y
        area = self._fibre_area
        material_type = self._fibre_mat
        material_index = self._fibre_mi

        # Fibre positions relative to centroid
        x_rel = x - self.section_centroid_x
        y_rel = y - self.section_centroid_y

        # Rotate neutral axis angle to radians
        angle_rad = np.radians(neutral_axis_angle)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)

        # Distance perpendicular to neutral axis
        dist_perp = y_rel * cos_a + x_rel * sin_a

        # Find extreme coordinates for pivot logic
        y_max = float(np.max(dist_perp))
        y_min = float(np.min(dist_perp))
        h = y_max - y_min

        # Find extreme rebar position for tension pivot
        steel_mask = material_type == 'steel'
        if np.any(steel_mask):
            rebar_y_min = float(np.min(dist_perp[steel_mask]))
        else:
            rebar_y_min = y_min

        d_eff = y_max - rebar_y_min

        # Vectorized strain calculation
        eps_cu2 = self.concrete.epsilon_cu2
        eps_c2 = self.concrete.epsilon_c2
        eps_ud = self._eps_tension_limit()

        x_bal = (eps_cu2 / (eps_cu2 + eps_ud)) * d_eff
        z_p = (1.0 - eps_c2 / eps_cu2) * h
        y_na = y_max - na_depth

        if na_depth <= x_bal:
            slope = -eps_ud / (rebar_y_min - y_na)
        elif na_depth <= h:
            slope = eps_cu2 / na_depth
        else:
            slope = eps_c2 / (na_depth - z_p)

        strains = slope * (dist_perp - y_na)

        # Get stresses from constitutive models
        concrete_mask = material_type == 'concrete'
        stresses = np.zeros_like(strains)

        # Concrete stresses (with confinement/tension stiffening support)
        if np.any(concrete_mask):
            stresses[concrete_mask] = self._concrete_stress_with_options(strains[concrete_mask])

        # Steel stresses
        if np.any(steel_mask):
            steel_strains = strains[steel_mask]
            steel_indices = material_index[steel_mask]
            steel_stresses = np.zeros_like(steel_strains)
            for gi in np.unique(steel_indices):
                m = steel_indices == gi
                if np.any(m):
                    steel_stresses[m] = self.steel_models[gi].get_stress_array(steel_strains[m])
            # Zero out compression steel if flag is set
            if self.ignore_compression_steel:
                steel_stresses[steel_strains > 0] = 0.0
            stresses[steel_mask] = steel_stresses

        # Calculate forces
        N = as_float(to_kn(np.sum(stresses * area), ForceUnit.N))
        My = as_float(to_knm(np.sum(stresses * area * x_rel), MomentUnit.NMM))
        Mz = as_float(to_knm(np.sum(stresses * area * y_rel), MomentUnit.NMM))

        # Track maximum strains
        max_conc_strain = float(np.max(np.abs(strains[concrete_mask]))) if np.any(concrete_mask) else 0.0
        max_steel_strain = float(np.max(np.abs(strains[steel_mask]))) if np.any(steel_mask) else 0.0

        return BiaxialInteractionPoint(
            N=N,
            My=My,
            Mz=Mz,
            neutral_axis_depth=na_depth,
            neutral_axis_angle=neutral_axis_angle,
            max_concrete_strain=max_conc_strain,
            max_steel_strain=max_steel_strain,
        )

    # ----------------------------
    # Dense generation + cache
    # ----------------------------

    def _get_dense_surface_points(
        self,
        n_dense_angles: int,
        n_dense_axial: int,
    ) -> tuple[BiaxialInteractionPoint, ...]:
        """
        Generate dense surface points (oversampled) and cache the result.
        Subsequent calls with the same params return the cached result.
        """
        params = (n_dense_angles, n_dense_axial)
        if self._dense_surface_points is not None and self._dense_params == params:
            return self._dense_surface_points

        pts = self._generate_surface_raw(
            n_angles=n_dense_angles,
            n_axial_levels=n_dense_axial,
        )

        self._dense_surface_points = pts
        self._dense_params = params
        # Invalidate downstream caches
        self._surface_cache.clear()
        self._hull_cache.clear()
        return pts

    def _generate_surface_raw(
        self,
        n_angles: int,
        n_axial_levels: int,
    ) -> tuple[BiaxialInteractionPoint, ...]:
        """
        Generate M-M-N surface using PIVOT METHOD with uniform N-level spacing.

        1. Calculate N_max and N_min using theoretical limits
        2. Create uniform N levels
        3. For each (N_target, angle), solve for NA depth using tangent mapping
        4. Force N to exact target for perfect uniformity
        """
        N_min, N_max = self.calculate_axial_limits()

        max_dim = max(self.section_width, self.section_height)

        N_levels = np.linspace(N_min * 0.98, N_max * 0.98, n_axial_levels)
        angles = np.linspace(0, 360, n_angles, endpoint=False)

        points: list[BiaxialInteractionPoint] = []

        for N_target in N_levels:
            for angle_deg in angles:
                def objective_tangent(phi: float) -> float:
                    na_depth = max_dim * np.tan(phi)
                    point = self.calculate_point_pivot(na_depth, angle_deg)
                    return point.N - N_target

                try:
                    phi_bound = 1.5
                    f_min = objective_tangent(-phi_bound)
                    f_max = objective_tangent(phi_bound)

                    if f_min * f_max > 0:
                        for phi_bound in (1.55, 1.56, 1.569):
                            f_min = objective_tangent(-phi_bound)
                            f_max = objective_tangent(phi_bound)
                            if f_min * f_max <= 0:
                                break

                    if f_min * f_max <= 0:
                        phi_solution = float(brentq(objective_tangent, -phi_bound, phi_bound, xtol=1e-5))  # type: ignore[arg-type]
                        na_depth_solution = max_dim * np.tan(phi_solution)
                        calc_point = self.calculate_point_pivot(na_depth_solution, angle_deg)

                        point = BiaxialInteractionPoint(
                            N=N_target,
                            My=calc_point.My,
                            Mz=calc_point.Mz,
                            neutral_axis_depth=calc_point.neutral_axis_depth,
                            neutral_axis_angle=calc_point.neutral_axis_angle,
                            max_concrete_strain=calc_point.max_concrete_strain,
                            max_steel_strain=calc_point.max_steel_strain,
                        )
                        points.append(point)
                    else:
                        if abs(N_target - N_max) < abs(N_target - N_min):
                            pole_point = self.calculate_point_pivot(max_dim * 10, angle_deg)
                        else:
                            pole_point = self.calculate_point_pivot(-max_dim * 2, angle_deg)

                        point = BiaxialInteractionPoint(
                            N=N_target,
                            My=pole_point.My,
                            Mz=pole_point.Mz,
                            neutral_axis_depth=pole_point.neutral_axis_depth,
                            neutral_axis_angle=pole_point.neutral_axis_angle,
                            max_concrete_strain=pole_point.max_concrete_strain,
                            max_steel_strain=pole_point.max_steel_strain,
                        )
                        points.append(point)
                except Exception:
                    continue

        return tuple(points)

    @staticmethod
    def _downsample_surface(
        dense_points: tuple[BiaxialInteractionPoint, ...],
        n_dense_angles: int,
        n_dense_axial: int,
        n_out_angles: int,
        n_out_axial: int,
    ) -> tuple[BiaxialInteractionPoint, ...]:
        """
        Downsample dense surface grid to requested resolution by taking evenly spaced indices.

        Uses integer step sizes (floor division) to ensure uniform spacing and preserve
        any symmetry present in the dense grid.
        """
        if n_out_angles >= n_dense_angles and n_out_axial >= n_dense_axial:
            return dense_points

        # Points are in order: for N_level in levels: for angle in angles
        total_dense = n_dense_axial * n_dense_angles
        if len(dense_points) < total_dense:
            # Some points failed; can't reliably index, return as-is
            return dense_points

        # Use exact step-based indexing (dense sizes are exact multiples of output)
        axial_step = max(1, n_dense_axial // n_out_axial)
        angle_step = max(1, n_dense_angles // n_out_angles)
        axial_indices = list(range(0, n_dense_axial, axial_step))[:n_out_axial]
        angle_indices = list(range(0, n_dense_angles, angle_step))[:n_out_angles]

        result: list[BiaxialInteractionPoint] = []
        for ai in axial_indices:
            for aj in angle_indices:
                idx = ai * n_dense_angles + aj
                if idx < len(dense_points):
                    result.append(dense_points[idx])

        return tuple(result)

    def generate_surface_pivot(
        self,
        n_angles: int = 36,
        n_axial_levels: int = 20,
        n_dense_angles: Optional[int] = None,
        n_dense_axial: Optional[int] = None,
    ) -> tuple[BiaxialInteractionPoint, ...]:
        """
        Generate M-M-N surface using PIVOT METHOD with oversample + downsample.

        Generates a dense grid at (n_dense_angles × n_dense_axial) resolution,
        then downsamples to the requested (n_angles × n_axial_levels).

        Args:
            n_angles: Number of neutral axis angles for output (longitude lines)
            n_axial_levels: Number of uniform N levels for output (latitude rings)
            n_dense_angles: Number of angles for dense generation (default: max(4*n_angles, 144))
            n_dense_axial: Number of N levels for dense generation (default: max(4*n_axial_levels, 80))

        Returns:
            Tuple of points forming the interaction surface
        """
        key = (n_angles, n_axial_levels)
        cached = self._surface_cache.get(key)
        if cached is not None:
            return cached

        # Default dense resolution: at least 4x oversample, exact multiple of output
        if n_dense_angles is None:
            factor_a = max(4, 144 // max(n_angles, 1))
            n_dense_angles = factor_a * n_angles
        if n_dense_axial is None:
            factor_n = max(4, 80 // max(n_axial_levels, 1))
            n_dense_axial = factor_n * n_axial_levels

        dense_pts = self._get_dense_surface_points(
            n_dense_angles=n_dense_angles,
            n_dense_axial=n_dense_axial,
        )

        result = self._downsample_surface(
            dense_points=dense_pts,
            n_dense_angles=n_dense_angles,
            n_dense_axial=n_dense_axial,
            n_out_angles=n_angles,
            n_out_axial=n_axial_levels,
        )

        self._surface_cache[key] = result
        return result

    def _prepare_surface_matrices(
        self,
        surface_pts: tuple[BiaxialInteractionPoint, ...],
        n_axial_levels: int,
        n_angles: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare surface data as 2D matrices for go.Surface plotting.

        Returns:
            Tuple of (My_matrix, Mz_matrix, N_matrix) shaped (n_axial_levels+2, n_angles+1)
        """
        # Extract arrays and reshape to grid
        N_raw = np.array([p.N for p in surface_pts]).reshape((n_axial_levels, n_angles))
        My_raw = np.array([p.My for p in surface_pts]).reshape((n_axial_levels, n_angles))
        Mz_raw = np.array([p.Mz for p in surface_pts]).reshape((n_axial_levels, n_angles))

        # Close the longitude loop
        N_grid = np.hstack([N_raw, N_raw[:, :1]])
        My_grid = np.hstack([My_raw, My_raw[:, :1]])
        Mz_grid = np.hstack([Mz_raw, Mz_raw[:, :1]])

        # Add pole points
        n_cols = n_angles + 1
        angles = np.linspace(0, 360, n_angles, endpoint=False)
        max_dim = max(self.section_width, self.section_height)

        tension_poles = [self.calculate_point_pivot(-max_dim * 2, ang) for ang in angles]
        compression_poles = [self.calculate_point_pivot(max_dim * 10, ang) for ang in angles]

        bot_pole_N = np.array([p.N for p in tension_poles] + [tension_poles[0].N]).reshape(1, n_cols)
        bot_pole_My = np.array([p.My for p in tension_poles] + [tension_poles[0].My]).reshape(1, n_cols)
        bot_pole_Mz = np.array([p.Mz for p in tension_poles] + [tension_poles[0].Mz]).reshape(1, n_cols)

        top_pole_N = np.array([p.N for p in compression_poles] + [compression_poles[0].N]).reshape(1, n_cols)
        top_pole_My = np.array([p.My for p in compression_poles] + [compression_poles[0].My]).reshape(1, n_cols)
        top_pole_Mz = np.array([p.Mz for p in compression_poles] + [compression_poles[0].Mz]).reshape(1, n_cols)

        N_final = np.vstack([bot_pole_N, N_grid, top_pole_N])
        My_final = np.vstack([bot_pole_My, My_grid, top_pole_My])
        Mz_final = np.vstack([bot_pole_Mz, Mz_grid, top_pole_Mz])

        return My_final, Mz_final, N_final

    def plot(
        self,
        load_points: Optional[List[Dict[str, Any]]] = None,
        show_vectors: bool = False,
        show_metadata: bool = True,
        n_angles: int = 36,
        n_axial_levels: int = 20,
        save_path: Optional[str] = None,
        show: bool = True,
        title: Optional[str] = None,
    ) -> Any:
        """
        Plot biaxial M-M-N interaction surface with optional load points using Plotly.

        Args:
            load_points: List of load case dictionaries with format:
                {"N_Ed": float, "My_Ed": float, "Mz_Ed": float, "name": str}
            show_vectors: If True, show vector projection rays
            show_metadata: If True, show metadata in hover tooltips
            n_angles: Number of angles for surface generation
            n_axial_levels: Number of N levels for surface generation
            save_path: If provided, save plot to this file path (HTML format)
            show: If True, display plot in browser
            title: Custom plot title (optional)

        Returns:
            Plotly Figure object
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            raise ImportError(
                "Plotly is required for plotting. Install with: pip install plotly"
            )

        surface_pts = self.generate_surface_pivot(
            n_angles=n_angles,
            n_axial_levels=n_axial_levels,
        )

        My_mat, Mz_mat, N_mat = self._prepare_surface_matrices(
            surface_pts, n_axial_levels, n_angles
        )

        fig = go.Figure()

        fig.add_trace(go.Surface(
            x=My_mat,
            y=Mz_mat,
            z=N_mat,
            colorscale='Viridis',
            opacity=0.5,
            name='M-M-N Surface',
            showlegend=True,
            showscale=False,
            hoverinfo='skip',
        ))

        fig.add_trace(go.Scatter3d(
            x=[0],
            y=[0],
            z=[0],
            mode='markers',
            name='Origin',
            marker=dict(color='black', size=3, symbol='circle'),
            hovertemplate='Origin<extra></extra>',
        ))

        if load_points:
            for idx, lp in enumerate(load_points):
                N_Ed = lp.get("N_Ed", 0.0)
                My_Ed = lp.get("My_Ed", 0.0)
                Mz_Ed = lp.get("Mz_Ed", 0.0)
                name = lp.get("name", f"Load Case {idx + 1}")

                N_Rd, My_Rd, Mz_Rd, is_safe, utilization = self.get_capacity_vector(
                    N_Ed=N_Ed, My_Ed=My_Ed, Mz_Ed=Mz_Ed,
                    surface_points=list(surface_pts),
                    n_angles=n_angles,
                    n_axial_levels=n_axial_levels
                )

                if utilization <= 0.8:
                    color = 'green'
                elif utilization <= 1.0:
                    color = 'orange'
                else:
                    color = 'red'

                if show_metadata:
                    hover_text = (
                        f"<b>{name}</b><br>"
                        f"N_Ed: {N_Ed:.1f} kN<br>"
                        f"My_Ed: {My_Ed:.1f} kN·m<br>"
                        f"Mz_Ed: {Mz_Ed:.1f} kN·m<br>"
                    )
                    if N_Rd is not None and My_Rd is not None and Mz_Rd is not None:
                        hover_text += (
                            f"N_Rd: {N_Rd:.1f} kN<br>"
                            f"My_Rd: {My_Rd:.1f} kN·m<br>"
                            f"Mz_Rd: {Mz_Rd:.1f} kN·m<br>"
                            f"Utilization: {utilization:.1%}<br>"
                            f"Status: {'✓ PASS' if is_safe else '✗ FAIL'}"
                        )
                    else:
                        hover_text += "Status: Outside boundary"
                else:
                    hover_text = name

                legend_grp = f"lc_{idx}"
                fig.add_trace(go.Scatter3d(
                    x=[My_Ed],
                    y=[Mz_Ed],
                    z=[N_Ed],
                    mode='markers',
                    name=name,
                    legendgroup=legend_grp,
                    marker=dict(
                        color=color,
                        size=5,
                        symbol='circle',
                        line=dict(color='black', width=1)
                    ),
                    hovertemplate=hover_text + '<extra></extra>',
                    showlegend=True,
                ))

                if show_vectors and N_Rd is not None and My_Rd is not None and Mz_Rd is not None:
                    fig.add_trace(go.Scatter3d(
                        x=[0, My_Ed],
                        y=[0, Mz_Ed],
                        z=[0, N_Ed],
                        mode='lines',
                        line=dict(color=color, width=3, dash='solid'),
                        legendgroup=legend_grp,
                        showlegend=False,
                        hoverinfo='skip',
                    ))

                    fig.add_trace(go.Scatter3d(
                        x=[My_Ed, My_Rd],
                        y=[Mz_Ed, Mz_Rd],
                        z=[N_Ed, N_Rd],
                        mode='lines',
                        line=dict(color=color, width=3, dash='dash'),
                        legendgroup=legend_grp,
                        showlegend=False,
                        hoverinfo='skip',
                    ))

        plot_title = title if title else "Biaxial M-M-N Interaction Surface"
        fig.update_layout(
            title=dict(text=plot_title, font=dict(size=16, color='black')),
            scene=dict(
                xaxis_title="My - Major Axis Moment (kN·m)",
                yaxis_title="Mz - Minor Axis Moment (kN·m)",
                zaxis_title="N - Axial Force (kN)",
                xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
                yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
                zaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
                aspectmode='cube',
            ),
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99
            ),
            width=1000,
            height=800,
        )

        if save_path:
            fig.write_html(save_path)

        if show:
            fig.show()

        return fig

    def export_to_json(
        self,
        file_path: str | Path,
        n_angles: int = 36,
        n_axial_levels: int = 20,
        include_metadata: bool = True,
        indent: int = 2,
    ) -> None:
        """Export biaxial M-M-N surface to JSON file."""
        points = self.generate_surface_pivot(n_angles=n_angles, n_axial_levels=n_axial_levels)

        data: Dict[str, Any] = {
            "surface_points": [p.to_dict() for p in points],
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
                "n_angles": n_angles,
                "n_axial_levels": n_axial_levels,
                "total_points": len(points),
            }

        file_path = Path(file_path)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent)

    def export_to_csv(
        self,
        file_path: str | Path,
        n_angles: int = 36,
        n_axial_levels: int = 20,
    ) -> None:
        """Export biaxial M-M-N surface to CSV file."""
        points = self.generate_surface_pivot(n_angles=n_angles, n_axial_levels=n_axial_levels)

        file_path = Path(file_path)
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = [
                'N_kN',
                'My_kNm',
                'Mz_kNm',
                'neutral_axis_depth_mm',
                'neutral_axis_angle_deg',
                'max_concrete_strain',
                'max_steel_strain',
            ]

            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for point in points:
                writer.writerow(point.to_dict())

    def __repr__(self) -> str:
        return (
            f"BiaxialMNInteractionSurface("
            f"section={self.section.section_name}, "
            f"concrete={self.concrete.grade})"
        )


def create_biaxial_interaction_surface(
    section: RCSection,
    concrete: ConcreteMaterial,
    **kwargs: Any,
) -> BiaxialMNInteractionSurface:
    """
    Factory function to create biaxial M-M-N interaction surface.

    Args:
        section: RC section with reinforcement
        concrete: Concrete material
        **kwargs: Additional arguments passed to BiaxialMNInteractionSurface

    Returns:
        BiaxialMNInteractionSurface instance
    """
    return BiaxialMNInteractionSurface(section=section, concrete=concrete, **kwargs)
