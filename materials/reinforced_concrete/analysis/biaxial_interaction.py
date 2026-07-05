"""
Biaxial M-M-N interaction surface generator using fiber-based strain compatibility.

Implements EC2 ultimate limit state analysis for combined axial force and biaxial bending.
"""

from typing import List, Optional, Literal, Dict, Any, Tuple
import json
import csv
from pathlib import Path
import numpy as np
from pydantic import BaseModel, Field, ConfigDict
from scipy.optimize import brentq
from scipy.spatial import ConvexHull

from materials.reinforced_concrete.geometry import RCSection, FiberMesh
from materials.reinforced_concrete.constitutive import (
    create_concrete_stress_strain,
    create_steel_stress_strain,
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
    Biaxial M-M-N interaction surface generator using fiber-based strain compatibility.

    The surface represents all combinations of axial force (N) and biaxial moments (My, Mz)
    that bring the section to its ultimate limit state per EC2.

    Axis Convention (3D FEA standard):
    - x: longitudinal axis (along member)
    - y: horizontal axis in cross-section (minor axis, width)
    - z: vertical axis in cross-section (major axis, height)
    - My: moment about y-axis (major axis bending)
    - Mz: moment about z-axis (minor axis bending)

    Method:
    1. Assume a neutral axis depth and angle
    2. Calculate strain distribution (plane sections remain plane)
    3. Strains perpendicular to the neutral axis
    4. Get stresses from constitutive models
    5. Integrate forces over fibers
    6. Result is one (N, My, Mz) point on the surface
    7. Repeat for different neutral axis depths and angles
    """

    def __init__(
        self,
        section: RCSection,
        concrete: ConcreteMaterial,
        concrete_model_type: Literal["parabola-rectangle", "bilinear"] = "parabola-rectangle",
        steel_branch_type: Literal["inclined", "horizontal"] = "inclined",
        n_fibers_width: int = 20,
        n_fibers_height: int = 30,
    ):
        """
        Initialize biaxial M-M-N surface generator.

        Args:
            section: RC section with reinforcement
            concrete: Concrete material properties
            concrete_model_type: Stress-strain model for concrete
            steel_branch_type: Stress-strain model for steel
            n_fibers_width: Fiber mesh resolution (width)
            n_fibers_height: Fiber mesh resolution (height)
        """
        self.section = section
        self.concrete = concrete

        # Create constitutive models
        self.concrete_model = create_concrete_stress_strain(
            concrete=concrete,
            model_type=concrete_model_type,
            use_characteristic=False,
        )

        if len(section.rebar_groups) == 0:
            raise ValueError("Section must have at least one rebar group")

        # Create steel models for each rebar group (to support different steel grades)
        self.steel_models = []
        for group in section.rebar_groups:
            steel_model = create_steel_stress_strain(
                steel=group.rebar,
                branch_type=steel_branch_type,
                use_characteristic=False,  # Use f_yd
            )
            self.steel_models.append(steel_model)

        # Generate fiber mesh
        self.mesh = FiberMesh(
            section=section,
            n_fibers_width=n_fibers_width,
            n_fibers_height=n_fibers_height,
            exclude_steel_area=True,
        )

        # Get section properties
        self.section_centroid_x, self.section_centroid_y = section.get_centroid()
        min_x, min_y, max_x, max_y = section.get_bounding_box()
        self.section_width = max_x - min_x
        self.section_height = max_y - min_y
        # Cache for generated surfaces and convex hulls to reuse across load checks
        self._surface_cache: Optional[Dict[str, Any]] = None

    def _build_convex_hull(self, surface_points: List["BiaxialInteractionPoint"]) -> ConvexHull:
        """Build a convex hull in (N, My, Mz) space from surface points."""
        pts = np.array([[p.N, p.My, p.Mz] for p in surface_points], dtype=float)
        if pts.shape[0] < 4:
            raise ValueError("At least 4 points are required to build a convex hull")
        return ConvexHull(pts)

    def _get_surface_and_hull(
        self,
        surface_points: Optional[List["BiaxialInteractionPoint"]],
        hull: Optional[ConvexHull],
        n_angles: int,
        n_axial_levels: int,
    ) -> tuple[List["BiaxialInteractionPoint"], ConvexHull]:
        """
        Return a tuple of (surface_points, convex_hull), caching both together.

        - If caller supplies points (and optionally a hull), reuse them, building the hull once.
        - Otherwise, reuse cached points/hull if generation parameters match.
        - If cache is absent or stale, regenerate points and hull and refresh the cache.
        """
        if surface_points is not None:
            pts = surface_points
            hull_obj = hull or self._build_convex_hull(pts)
            return pts, hull_obj

        params = (n_angles, n_axial_levels)
        if self._surface_cache and self._surface_cache.get("params") == params:
            cached_pts = self._surface_cache["points"]
            cached_hull = self._surface_cache["hull"]
            return cached_pts, cached_hull

        pts = self.generate_surface_pivot(
            n_angles=n_angles,
            n_axial_levels=n_axial_levels,
        )
        hull_obj = self._build_convex_hull(pts)
        self._surface_cache = {"params": params, "points": pts, "hull": hull_obj}
        return pts, hull_obj

    def get_utilization_vector(
        self,
        N_Ed: float,
        My_Ed: float,
        Mz_Ed: float,
        surface_points: Optional[List] = None,
        hull: Optional[ConvexHull] = None,
        n_angles: int = 72,
        n_axial_levels: int = 30,
    ) -> Tuple[bool, float]:
        """
        Check capacity using exact ray-to-surface intersection via convex hull.

        Projects a ray from origin through (N_Ed, My_Ed, Mz_Ed) and intersects it with
        the convex hull of the generated surface points.
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
        surface_points: Optional[List] = None,
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

        # Use provided surface/hull or generate/cached pair
        try:
            points, hull_obj = self._get_surface_and_hull(
                surface_points=surface_points,
                hull=hull,
                n_angles=n_angles,
                n_axial_levels=n_axial_levels,
            )
        except Exception:
            return (None, None, None, False, float("inf"))

        load_vec = np.array([N_Ed, My_Ed, Mz_Ed], dtype=float)
        load_mag = np.linalg.norm(load_vec)

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
        surface_points: Optional[List] = None,
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
    # NEW SURFACE GENERATION - EC2 Pivot Method (code review Fixes)
    # ========================================================================

    def calculate_axial_limits(self) -> tuple[float, float]:
        """
        Calculate the absolute theoretical N_min (pure tension) and N_max (pure compression).

        These values bound the interaction surface:
        - N_min: Concrete fully cracked, all steel at -eps_ud (tension)
        - N_max: Entire section at uniform strain eps_c2 (compression)

        Returns:
            Tuple of (N_min, N_max) in kN
        """
        # Get fiber data
        _, _, area, mat_type, mat_idx = self.mesh.get_fiber_arrays()

        # EC2 strain limits
        eps_cu2 = self.concrete.epsilon_cu2  # Use ultimate compression strain for pole
        eps_ud = 0.02  # Design ultimate tension strain (assumed design limit for reinforcement)

        # 1. PURE TENSION (N_min)
        # Concrete contributes 0, all steel at -eps_ud
        n_min = 0.0

        steel_mask = (mat_type == 'steel')
        if np.any(steel_mask):
            # Tension strain is negative
            n_steel_tension = 0.0
            unique_steel_groups = np.unique(mat_idx[steel_mask])

            for g_idx in unique_steel_groups:
                group_mask = steel_mask & (mat_idx == g_idx)

                # Get stress at -eps_ud from constitutive model
                stress_tension = self.steel_models[int(g_idx)].get_stress(-eps_ud)
                n_steel_tension += stress_tension * np.sum(area[group_mask])

            n_min = n_steel_tension / 1000.0  # Convert to kN

        # 2. PURE COMPRESSION (N_max)
        # Uniform strain eps_c2 across entire section
        n_max = 0.0

        # Concrete contribution
        conc_mask = (mat_type == 'concrete')
        if np.any(conc_mask):
            stress_c = self.concrete_model.get_stress(eps_cu2)
            n_max += stress_c * np.sum(area[conc_mask]) / 1000.0

        # Steel contribution at eps_c2 compression
        if np.any(steel_mask):
            n_steel_compression = 0.0
            unique_steel_groups = np.unique(mat_idx[steel_mask])

            for g_idx in unique_steel_groups:
                group_mask = steel_mask & (mat_idx == g_idx)
                stress_comp = self.steel_models[int(g_idx)].get_stress(eps_cu2)
                n_steel_compression += stress_comp * np.sum(area[group_mask])

            n_max += n_steel_compression / 1000.0

        return (n_min, n_max)

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

        CRITICAL FIX: The transition between Zone A and Zone B happens at the
        BALANCED DEPTH (x_bal), NOT at x=0. Using x=0 creates a discontinuity
        that causes "divots" in the surface.

        Three zones per EC2:
        - Zone A: Tension failure (pivot at extreme rebar ε_ud) when na_depth <= x_bal
        - Zone B: Bending failure (pivot at extreme concrete fiber ε_cu2) when x_bal < na_depth <= h
        - Zone C: Compression failure (pivot at depth z_p with ε_c2) when na_depth > h

        Args:
            y: Coordinate to evaluate strain at
            na_depth: Neutral axis depth from y_max (positive downward)
            y_max: Maximum y coordinate (top fiber, compression side)
            y_min: Minimum y coordinate (bottom fiber, tension side)
            h: Total height (y_max - y_min)
            rebar_y_min: Position of extreme tension rebar
            d_eff: Effective depth to extreme tension rebar (y_max - rebar_y_min)

        Returns:
            Strain at y (positive = compression)
        """
        # EC2 strain limits
        eps_cu2 = self.concrete.epsilon_cu2  # Ultimate compression strain (0.0035)
        eps_c2 = self.concrete.epsilon_c2    # Parabola-rectangle transition (0.0020)
        eps_ud = 0.02  # Design ultimate tension strain for reinforcement

        # Calculate balanced depth - where concrete reaches eps_cu2 at same time
        # as extreme rebar reaches -eps_ud
        x_bal = (eps_cu2 / (eps_cu2 + eps_ud)) * d_eff

        # Pivot depth for Zone C (pure compression)
        z_p = (1.0 - eps_c2 / eps_cu2) * h

        # Neutral axis position (y-coordinate)
        y_na = y_max - na_depth

        # ZONE A: Tension Failure
        # Pivot at extreme rebar with -ε_ud
        if na_depth <= x_bal:
            # Strain profile: -ε_ud at rebar_y_min, 0 at y_na
            slope = -eps_ud / (rebar_y_min - y_na)
            return slope * (y - y_na)

        # ZONE B: Bending Failure (most common)
        # Pivot at top fiber with ε_cu2
        elif na_depth <= h:
            # Strain profile: ε_cu2 at y_max, 0 at y_na
            slope = eps_cu2 / na_depth
            return slope * (y - y_na)

        # ZONE C: Compression Failure
        # Pivot at z_p from top fiber with ε_c2
        else:  # na_depth > h
            # Strain profile: ε_c2 at (y_max - z_p), 0 at y_na
            slope = eps_c2 / (na_depth - z_p)
            return slope * (y - y_na)

    def calculate_point_pivot(
        self,
        na_depth: float,
        neutral_axis_angle: float = 0.0,
    ) -> BiaxialInteractionPoint:
        """
        Calculate point on M-M-N surface using PIVOT METHOD (vectorized).

        This uses the EC2 pivot method to ensure strains always touch ultimate limits.
        Includes vectorization for 10-100x speedup over loop-based approach.

        Args:
            na_depth: Neutral axis depth from top fiber (mm, positive = deeper)
            neutral_axis_angle: Angle of neutral axis from horizontal (degrees)

        Returns:
            Point on the failure surface
        """
        # Get fiber coordinates
        x, y, area, material_type, material_index = self.mesh.get_fiber_arrays()

        # Fiber positions relative to centroid
        x_rel = x - self.section_centroid_x
        y_rel = y - self.section_centroid_y

        # Rotate neutral axis angle to radians
        angle_rad = np.radians(neutral_axis_angle)

        # Rotation matrix for neutral axis angle
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)

        # Rotate to align neutral axis with horizontal
        # Distance perpendicular to neutral axis
        dist_perp = y_rel * cos_a + x_rel * sin_a

        # Find extreme coordinates for pivot logic
        y_max = np.max(dist_perp)
        y_min = np.min(dist_perp)
        h = y_max - y_min

        # Find extreme rebar position for tension pivot
        steel_mask = material_type == 'steel'
        if np.any(steel_mask):
            rebar_y_min = np.min(dist_perp[steel_mask])
        else:
            rebar_y_min = y_min

        # Calculate effective depth for balanced depth calculation
        d_eff = y_max - rebar_y_min

        # VECTORIZED strain calculation (much faster than loop)
        # Determine which pivot zone based on na_depth
        eps_cu2 = self.concrete.epsilon_cu2  # 0.0035
        eps_c2 = self.concrete.epsilon_c2    # 0.0020
        eps_ud = 0.02

        # Calculate balanced depth and pivot depth
        x_bal = (eps_cu2 / (eps_cu2 + eps_ud)) * d_eff
        z_p = (1.0 - eps_c2 / eps_cu2) * h
        y_na = y_max - na_depth

        # Determine slope based on zone (same logic as _get_strain_at_y_pivot but vectorized)
        if na_depth <= x_bal:
            # ZONE A: Tension Failure
            slope = -eps_ud / (rebar_y_min - y_na)
        elif na_depth <= h:
            # ZONE B: Bending Failure
            slope = eps_cu2 / na_depth
        else:
            # ZONE C: Compression Failure
            slope = eps_c2 / (na_depth - z_p)

        # Vectorized strain calculation (all fibers at once!)
        strains = slope * (dist_perp - y_na)

        # Get stresses from constitutive models
        concrete_mask = material_type == 'concrete'
        stresses = np.zeros_like(strains)

        # Concrete stresses
        if np.any(concrete_mask):
            stresses[concrete_mask] = self.concrete_model.get_stress_array(strains[concrete_mask])

        # Steel stresses
        if np.any(steel_mask):
            for group_idx in np.unique(material_index[steel_mask]):
                group_mask = (material_type == 'steel') & (material_index == group_idx)
                stresses[group_mask] = self.steel_models[group_idx].get_stress_array(strains[group_mask])

        # Calculate forces
        # Axial force (N = sum(stress * A))
        N = np.sum(stresses * area) / 1000.0  # kN

        # Moments about centroid (axis convention: x along member, My from z-forces, Mz from y-forces)
        # My (about y-axis): M = sum(stress * A * x_offset)
        My = np.sum(stresses * area * x_rel) / 1e6  # kN·m

        # Mz (about z-axis): M = sum(stress * A * y_offset)
        Mz = np.sum(stresses * area * y_rel) / 1e6  # kN·m

        # Track maximum strains
        max_conc_strain = np.max(np.abs(strains[concrete_mask])) if np.any(concrete_mask) else 0.0
        max_steel_strain = np.max(np.abs(strains[steel_mask])) if np.any(steel_mask) else 0.0

        return BiaxialInteractionPoint(
            N=N,
            My=My,
            Mz=Mz,
            neutral_axis_depth=na_depth,
            neutral_axis_angle=neutral_axis_angle,
            max_concrete_strain=max_conc_strain,
            max_steel_strain=max_steel_strain,
        )

    def generate_surface_pivot(
        self,
        n_angles: int = 36,
        n_axial_levels: int = 20,
    ) -> List[BiaxialInteractionPoint]:
        """
        Generate M-M-N surface using PIVOT METHOD with uniform N-level spacing.

        This is the CORRECT implementation per code review's advice:
        1. Calculate N_max and N_min using theoretical limits
        2. Create uniform N levels
        3. For each (N_target, angle), solve for NA depth using tangent mapping
        4. Force N to exact target for perfect uniformity
        5. This guarantees points on the failure surface, no interior points

        Args:
            n_angles: Number of neutral axis angles (longitude lines)
            n_axial_levels: Number of uniform N levels (latitude rings)

        Returns:
            List of points forming the interaction surface
        """
        print(f"Generating surface using PIVOT METHOD: {n_angles} angles × {n_axial_levels} N levels...")

        # Step 1: Calculate N_max (pure compression) and N_min (pure tension)
        # Use theoretical limits for stability
        print("  Calculating axial force limits...")
        N_min, N_max = self.calculate_axial_limits()

        print(f"  N range: {N_min:.1f} to {N_max:.1f} kN")

        # Get section dimensions for solver bounds
        max_dim = max(self.section_width, self.section_height)

        # Step 2: Create uniform N levels (always include full range, tension to compression)
        N_levels = np.linspace(N_min * 0.98, N_max * 0.98, n_axial_levels)

        # Step 3: Create uniform angles
        angles = np.linspace(0, 360, n_angles, endpoint=False)

        print(f"  Solving for {n_axial_levels} × {n_angles} = {n_axial_levels * n_angles} points...")

        points = []

        # Step 4: For each (N_target, angle), solve for NA depth using TANGENT MAPPING
        # This maps na_depth from [-∞, ∞] to phi in [-π/2, π/2] for stability
        for N_target in N_levels:
            for angle_deg in angles:
                # Tangent mapping: na_depth = h * tan(phi)
                # This ensures solver doesn't fail at extreme poles
                def objective_tangent(phi: float) -> float:
                    na_depth = max_dim * np.tan(phi)
                    point = self.calculate_point_pivot(na_depth, angle_deg)
                    return point.N - N_target

                try:
                    # Search in finite bounds [-1.5, 1.5] which covers most of [-∞, ∞]
                    # when mapped through tan(). Expand slightly if not bracketed.
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
                        # Target is bracketed, solve for phi
                        phi_solution = brentq(objective_tangent, -phi_bound, phi_bound, xtol=1e-5)

                        # Convert back to na_depth
                        na_depth_solution = max_dim * np.tan(phi_solution)

                        # Calculate point with solved NA depth
                        calc_point = self.calculate_point_pivot(na_depth_solution, angle_deg)

                        # Force N to exact target for uniform grid
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
                        # Target not bracketed - use pole point
                        # This happens at extreme N values (near N_min or N_max)
                        if abs(N_target - N_max) < abs(N_target - N_min):
                            # Closer to compression pole
                            pole_point = self.calculate_point_pivot(max_dim * 10, angle_deg)
                        else:
                            # Closer to tension pole
                            pole_point = self.calculate_point_pivot(-max_dim * 2, angle_deg)

                        # Use actual pole moments (non-zero for asymmetric sections)
                        point = BiaxialInteractionPoint(
                            N=N_target,
                            My=pole_point.My,  # Use actual My (may be non-zero for asymmetric sections)
                            Mz=pole_point.Mz,  # Use actual Mz (may be non-zero for asymmetric sections)
                            neutral_axis_depth=pole_point.neutral_axis_depth,
                            neutral_axis_angle=pole_point.neutral_axis_angle,
                            max_concrete_strain=pole_point.max_concrete_strain,
                            max_steel_strain=pole_point.max_steel_strain,
                        )
                        points.append(point)
                except Exception:
                    # Solver failed - skip this point
                    continue

        print(f"  [OK] Generated {len(points)} surface points")
        print(f"  Success rate: {len(points)}/{n_axial_levels * n_angles} = {100*len(points)/(n_axial_levels * n_angles):.1f}%")

        return points

    def _prepare_surface_matrices(
        self,
        surface_pts: List[BiaxialInteractionPoint],
        n_axial_levels: int,
        n_angles: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare surface data as 2D matrices for go.Surface plotting.

        This eliminates "interior lines" by structuring the data as a proper grid
        and closing the loops at the poles and at the 0°/360° seam.

        Args:
            surface_pts: List of surface points (must be in order: for N in levels: for angle in angles)
            n_axial_levels: Number of N levels in the grid
            n_angles: Number of angles in the grid

        Returns:
            Tuple of (My_matrix, Mz_matrix, N_matrix) shaped (n_axial_levels+2, n_angles+1)
            The +2 is for pole points, +1 is for closing the angular seam
        """
        # Extract arrays and reshape to grid
        # Assumes points were generated as: for N in N_levels: for ang in angles
        N_raw = np.array([p.N for p in surface_pts]).reshape((n_axial_levels, n_angles))
        My_raw = np.array([p.My for p in surface_pts]).reshape((n_axial_levels, n_angles))
        Mz_raw = np.array([p.Mz for p in surface_pts]).reshape((n_axial_levels, n_angles))

        # Close the longitude loop (0° to 360°)
        # Append first column to end so surface connects back to itself
        N_grid = np.hstack([N_raw, N_raw[:, :1]])
        My_grid = np.hstack([My_raw, My_raw[:, :1]])
        Mz_grid = np.hstack([Mz_raw, Mz_raw[:, :1]])

        # Add pole points to close the tips using actual fiber integration (non-zero for asymmetric sections)
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

        # Stack: [Bottom Pole] -> [Surface Grid] -> [Top Pole]
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

        Creates an interactive 3D plot with:
        - Translucent M-M-N interaction surface
        - Latitude rings (constant N contours) and longitude lines (constant angle rays)
        - Optional load points with color-coded utilization
        - Optional vector projection rays from origin to surface
        - Interactive hover tooltips with metadata

        Args:
            load_points: List of load case dictionaries with format:
                {
                    "N_Ed": float,      # Axial force (kN)
                    "My_Ed": float,     # Moment about y-axis (kN·m)
                    "Mz_Ed": float,     # Moment about z-axis (kN·m)
                    "name": str,        # Load case name (optional)
                }
            show_vectors: If True, show vector projection rays from origin through
                         load points to capacity surface
            show_metadata: If True, show metadata in hover tooltips
            n_angles: Number of angles for surface generation (longitude lines)
            n_axial_levels: Number of N levels for surface generation (latitude rings)
            save_path: If provided, save plot to this file path (HTML format)
            show: If True, display plot in browser
            title: Custom plot title (optional)

        Returns:
            Plotly Figure object

        Example:
            >>> surface = create_biaxial_interaction_surface(section, concrete)
            >>> surface.plot(
            ...     load_points=[
            ...         {"N_Ed": 1000, "My_Ed": 150, "Mz_Ed": 100, "name": "LC1: DL+LL"},
            ...         {"N_Ed": 800, "My_Ed": 200, "Mz_Ed": 80, "name": "LC2: DL+Wind"}
            ...     ],
            ...     show_vectors=True,
            ...     save_path="biaxial_surface.html"
            ... )
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            raise ImportError(
                "Plotly is required for plotting. Install with: pip install plotly"
            )

        # Generate surface points using EC2 pivot method (guarantees no interior points)
        surface_pts = self.generate_surface_pivot(
            n_angles=n_angles,
            n_axial_levels=n_axial_levels,
        )

        # Prepare matrices for go.Surface (eliminates interior lines)
        print("Preparing surface matrices for go.Surface...")
        My_mat, Mz_mat, N_mat = self._prepare_surface_matrices(
            surface_pts, n_axial_levels, n_angles
        )
        print(f"[OK] Prepared surface matrix with shape {N_mat.shape}")

        # Create figure
        fig = go.Figure()

        # Add the M-M-N surface using go.Surface (watertight, no interior lines)
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

        # Add origin point
        fig.add_trace(go.Scatter3d(
            x=[0],
            y=[0],
            z=[0],
            mode='markers',
            name='Origin',
            marker=dict(color='black', size=3, symbol='circle'),
            hovertemplate='Origin<extra></extra>',
        ))

        # Process load points if provided
        if load_points:
            for idx, lp in enumerate(load_points):
                N_Ed = lp.get("N_Ed", 0.0)
                My_Ed = lp.get("My_Ed", 0.0)
                Mz_Ed = lp.get("Mz_Ed", 0.0)
                name = lp.get("name", f"Load Case {idx + 1}")

                # Get capacity and utilization using surface points
                N_Rd, My_Rd, Mz_Rd, is_safe, utilization = self.get_capacity_vector(
                    N_Ed=N_Ed, My_Ed=My_Ed, Mz_Ed=Mz_Ed,
                    surface_points=surface_pts,
                    n_angles=n_angles,
                    n_axial_levels=n_axial_levels
                )

                # Color coding based on utilization
                if utilization <= 0.8:
                    color = 'green'
                elif utilization <= 1.0:
                    color = 'orange'
                else:
                    color = 'red'

                # Build hover text with metadata
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

                # Plot load point (smaller markers)
                fig.add_trace(go.Scatter3d(
                    x=[My_Ed],
                    y=[Mz_Ed],
                    z=[N_Ed],
                    mode='markers',
                    name=name,
                    marker=dict(
                        color=color,
                        size=5,  # Reduced from 8
                        symbol='circle',
                        line=dict(color='black', width=1)
                    ),
                    hovertemplate=hover_text + '<extra></extra>',
                    showlegend=True,
                ))

                # Add vector projection rays if requested
                if show_vectors and N_Rd is not None and My_Rd is not None and Mz_Rd is not None:
                    # Solid line from origin to load point
                    fig.add_trace(go.Scatter3d(
                        x=[0, My_Ed],
                        y=[0, Mz_Ed],
                        z=[0, N_Ed],
                        mode='lines',
                        line=dict(color=color, width=3, dash='solid'),
                        showlegend=False,
                        hoverinfo='skip',
                    ))

                    # Dashed line from load point to capacity boundary
                    fig.add_trace(go.Scatter3d(
                        x=[My_Ed, My_Rd],
                        y=[Mz_Ed, Mz_Rd],
                        z=[N_Ed, N_Rd],
                        mode='lines',
                        line=dict(color=color, width=3, dash='dash'),
                        showlegend=False,
                        hoverinfo='skip',
                    ))

        # Update layout
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
                aspectmode='cube',  # Balanced visual proportions (height ≈ width)
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

        # Save if requested
        if save_path:
            fig.write_html(save_path)
            print(f"[OK] Saved plot to {save_path}")

        # Show if requested
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
                "n_fibers": self.mesh.total_fibers,
                "concrete_model": type(self.concrete_model).__name__,
                "steel_models": [type(sm).__name__ for sm in self.steel_models],
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
    **kwargs,
) -> BiaxialMNInteractionSurface:
    """
    Factory function to create biaxial M-M-N interaction surface.

    Args:
        section: RC section with reinforcement
        concrete: Concrete material
        **kwargs: Additional arguments passed to BiaxialMNInteractionSurface

    Returns:
        BiaxialMNInteractionSurface instance

    Example:
        >>> from materials.reinforced_concrete.geometry import create_rectangular_section
        >>> from materials.reinforced_concrete.materials import ConcreteMaterial
        >>>
        >>> section = create_rectangular_section(300, 500)
        >>> # ... add reinforcement ...
        >>> concrete = ConcreteMaterial(grade="C30/37")
        >>>
        >>> surface = create_biaxial_interaction_surface(section, concrete)
        >>> points = surface.generate_surface_pivot(n_angles=60, n_axial_levels=50)
    """
    return BiaxialMNInteractionSurface(section=section, concrete=concrete, **kwargs)
