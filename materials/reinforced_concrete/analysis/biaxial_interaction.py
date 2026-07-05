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
from scipy.optimize import brentq, OptimizeWarning
import warnings

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

    def find_neutral_axis_depth_for_axial_force(
        self,
        target_N: float,
        neutral_axis_angle: float = 0.0,
        max_concrete_strain: Optional[float] = None,
        tol: float = 0.1,
    ) -> Optional[float]:
        """
        Find the neutral axis depth that produces a target axial force N.

        This is the inverse problem: given (target_N, angle) → find NA_depth.
        Uses root finding to solve: N(NA_depth, angle) - target_N = 0

        Args:
            target_N: Target axial force (kN)
            neutral_axis_angle: Neutral axis angle (degrees)
            max_concrete_strain: Maximum concrete strain
            tol: Tolerance for N convergence (kN)

        Returns:
            Neutral axis depth (mm) or None if not found
        """
        max_depth = max(self.section_width, self.section_height)

        # Define objective function: N(depth) - target_N
        def objective(depth: float) -> float:
            point = self.calculate_point(depth, neutral_axis_angle, max_concrete_strain)
            return point.N - target_N

        # Search bounds: from deep tension to deep compression
        depth_min = -max_depth * 3
        depth_max = max_depth * 20

        try:
            # Check if target is bracketed
            f_min = objective(depth_min)
            f_max = objective(depth_max)

            if f_min * f_max > 0:
                # Target not bracketed - may be outside valid range
                return None

            # Use Brent's method for root finding
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=OptimizeWarning)
                # brentq returns a scalar (not tuple) when full_output=False (default)
                result: float = brentq(objective, depth_min, depth_max, xtol=tol)  # type: ignore[assignment]
                # Convert to Python float (brentq returns numpy scalar)
                na_depth = float(result)

            return na_depth

        except (ValueError, RuntimeError):
            # Root finding failed
            return None

    def calculate_point(
        self,
        neutral_axis_depth: float,
        neutral_axis_angle: float = 0.0,
        max_concrete_strain: Optional[float] = None,
    ) -> BiaxialInteractionPoint:
        """
        Calculate single point on biaxial interaction surface.

        Uses strain compatibility with rotated neutral axis:
        - Plane sections remain plane
        - Strain varies linearly perpendicular to neutral axis
        - Maximum concrete strain at extreme fiber

        Args:
            neutral_axis_depth: Distance from centroid to neutral axis (mm)
                                Positive = compression side
            neutral_axis_angle: Angle of neutral axis (degrees)
                                0° = NA horizontal → vertical forces → My (major axis)
                                90° = NA vertical → horizontal forces → Mz (minor axis)
            max_concrete_strain: Maximum concrete compressive strain (default: ε_cu2)

        Returns:
            BiaxialInteractionPoint with N, My, Mz, and strain information
        """
        if max_concrete_strain is None:
            max_concrete_strain = self.concrete_model.get_ultimate_strain()

        # Get fiber coordinates relative to section centroid
        x, y, area, material_type, material_index = self.mesh.get_fiber_arrays()

        # Fiber positions relative to centroid
        x_rel = x - self.section_centroid_x
        y_rel = y - self.section_centroid_y

        # Neutral axis angle in radians
        theta = np.radians(neutral_axis_angle)

        # Perpendicular distance from each fiber to the neutral axis
        # NA passes through a point at distance 'neutral_axis_depth' from centroid
        # in the direction perpendicular to the NA (angle theta + 90°)
        # Distance = x*sin(theta) - y*cos(theta) + neutral_axis_depth
        # (using signed distance, positive = compression side)

        # Normal vector to neutral axis (pointing toward compression zone)
        nx = np.sin(theta)
        ny = -np.cos(theta)

        # Perpendicular distance from each fiber to neutral axis
        # The neutral axis is defined by: nx*x + ny*y = neutral_axis_depth
        # So distance from a point (x_rel, y_rel) to this line is:
        distance_from_na = nx * x_rel + ny * y_rel - neutral_axis_depth

        # Find the extreme fiber distances for scaling strains
        # The extreme compression fiber has maximum positive distance
        # The extreme tension fiber has maximum negative distance
        d_max = np.max(distance_from_na)  # Extreme compression fiber
        d_min = np.min(distance_from_na)  # Extreme tension fiber

        # Calculate strains using plane sections remain plane
        # Strain varies linearly from neutral axis
        # At extreme compression fiber: ε = ε_cu (ultimate compression strain)
        # At neutral axis: ε = 0
        # In tension zone: ε < 0 (tensile strain)

        if d_max > 0:
            # There is a compression zone
            # Scale strains so that extreme compression fiber reaches ε_cu
            strains = max_concrete_strain * distance_from_na / d_max
        elif d_min < 0:
            # Pure tension (all fibers in tension)
            # Scale by most tensioned fiber
            strains = max_concrete_strain * distance_from_na / abs(d_min)
        else:
            # All fibers at neutral axis (shouldn't happen)
            strains = np.zeros_like(distance_from_na)

        # Calculate stresses from constitutive models
        stresses = np.zeros_like(strains)

        # Concrete fibers
        concrete_mask = material_type == 'concrete'
        stresses[concrete_mask] = self.concrete_model.get_stress_array(strains[concrete_mask])

        # Steel fibers - apply correct steel model for each rebar group
        steel_mask = material_type == 'steel'
        steel_strains = strains[steel_mask]
        steel_indices = material_index[steel_mask]

        # Calculate stresses for each rebar group separately
        steel_stresses = np.zeros_like(steel_strains)
        for group_idx in range(len(self.steel_models)):
            # Find fibers belonging to this rebar group
            group_mask = steel_indices == group_idx
            if np.any(group_mask):
                # Apply the steel model for this specific group
                steel_stresses[group_mask] = self.steel_models[group_idx].get_stress_array(
                    steel_strains[group_mask]
                )

        stresses[steel_mask] = steel_stresses

        # Calculate resultant forces
        # N = Σ(σ · A)
        N = np.sum(stresses * area) / 1000.0  # kN

        # Moments about centroid using 3D FEA convention
        # Section coords: x_rel = horizontal (width), y_rel = vertical (height)
        # FEA coords: y = horizontal (minor), z = vertical (major)
        # My = Σ(σ · A · z_rel) = Σ(σ · A · y_rel_section)  (major axis moment)
        # Mz = Σ(σ · A · y_rel) = Σ(σ · A · x_rel_section)  (minor axis moment)
        My = np.sum(stresses * area * y_rel) / 1e6  # kN·m (major axis)
        Mz = np.sum(stresses * area * x_rel) / 1e6  # kN·m (minor axis)

        # Get maximum strains for reporting
        max_conc_strain = np.max(strains[concrete_mask]) if np.any(concrete_mask) else 0.0
        max_steel_strain = np.max(np.abs(strains[steel_mask])) if np.any(steel_mask) else 0.0

        return BiaxialInteractionPoint(
            N=N,
            My=My,
            Mz=Mz,
            neutral_axis_depth=neutral_axis_depth,
            neutral_axis_angle=neutral_axis_angle,
            max_concrete_strain=max_conc_strain,
            max_steel_strain=max_steel_strain,
        )

    def generate_surface(
        self,
        n_angles: int = 36,
        n_axial_levels: int = 20,
        include_tension: bool = True,
    ) -> List[BiaxialInteractionPoint]:
        """
        Generate biaxial M-M-N interaction surface using constant N contours.

        This method creates horizontal slices through the surface at constant N levels,
        producing the proper rugby ball / ellipsoidal shape.

        For each N level:
        1. Sweep through neutral axis angles (0° to 360°)
        2. Solve for NA depth that gives the target N
        3. Creates a closed contour (ellipse) at that N level

        Args:
            n_angles: Number of neutral axis angles per contour (recommended: 36-60)
            n_axial_levels: Number of N levels to generate (recommended: 20-50)
            include_tension: Include tension branch (negative N)

        Returns:
            List of BiaxialInteractionPoint with proper ellipsoidal shape

        Example:
            >>> surface = create_biaxial_interaction_surface(section, concrete)
            >>> points = surface.generate_surface(n_angles=60, n_axial_levels=50)
            >>> # Creates ~3000 points in rugby ball shape
        """
        points: List[BiaxialInteractionPoint] = []

        # Calculate pure compression and tension capacities
        pure_compression_N, pure_tension_N = self._get_axial_capacity_range()

        # Define N levels from tension to compression
        if include_tension:
            N_levels = np.linspace(pure_tension_N * 0.95, pure_compression_N * 0.95, n_axial_levels)
        else:
            N_levels = np.linspace(0, pure_compression_N * 0.95, n_axial_levels)

        # Generate points for each N level
        for target_N in N_levels:
            # Angles for this contour (0° to 360°)
            angles = np.linspace(0, 360, n_angles, endpoint=False)

            for angle in angles:
                # Find NA depth that gives target N at this angle
                na_depth = self.find_neutral_axis_depth_for_axial_force(
                    target_N=target_N,
                    neutral_axis_angle=angle,
                )

                if na_depth is not None:
                    point = self.calculate_point(na_depth, angle)
                    points.append(point)

        # Add pure compression point
        pure_comp_point = self._get_pure_compression_point()
        points.append(pure_comp_point)

        # Add pure tension point if requested
        if include_tension:
            pure_tens_point = self._get_pure_tension_point()
            points.append(pure_tens_point)

        return points

    def _get_axial_capacity_range(self) -> Tuple[float, float]:
        """Get pure compression and pure tension capacities."""
        # Pure compression
        concrete_area = self.section.outline.area
        f_c = self.concrete_model.get_yield_stress()
        N_concrete = concrete_area * f_c / 1000.0

        N_steel_comp = 0.0
        N_steel_tens = 0.0

        for group_idx, group in enumerate(self.section.rebar_groups):
            A_s = group.rebar.area
            f_yd = self.steel_models[group_idx].get_yield_stress()
            n_bars = len(group.positions)

            N_steel_comp += n_bars * A_s * f_yd / 1000.0
            N_steel_tens += n_bars * A_s * f_yd / 1000.0

        pure_compression_N = N_concrete + N_steel_comp
        pure_tension_N = -N_steel_tens

        return pure_compression_N, pure_tension_N

    def _get_pure_compression_point(self) -> BiaxialInteractionPoint:
        """Calculate pure compression point."""
        section_cx, section_cy = self.section.get_centroid()

        concrete_area = self.section.outline.area
        f_c = self.concrete_model.get_yield_stress()
        N_concrete = concrete_area * f_c / 1000.0

        N_steel = 0.0
        My_steel = 0.0
        Mz_steel = 0.0
        max_steel_strain = 0.0

        for group_idx, group in enumerate(self.section.rebar_groups):
            A_s = group.rebar.area
            f_yd = self.steel_models[group_idx].get_yield_stress()

            for pos in group.positions:
                bar_force = A_s * f_yd / 1000.0
                N_steel += bar_force

                x_offset = pos.x - section_cx
                y_offset = pos.y - section_cy

                My_steel += bar_force * y_offset / 1000.0
                Mz_steel += bar_force * x_offset / 1000.0

            max_steel_strain = max(max_steel_strain, self.steel_models[group_idx].epsilon_y)

        max_depth = max(self.section_width, self.section_height)

        return BiaxialInteractionPoint(
            N=N_concrete + N_steel,
            My=My_steel,
            Mz=Mz_steel,
            neutral_axis_depth=max_depth * 1000,
            neutral_axis_angle=0.0,
            max_concrete_strain=self.concrete_model.get_ultimate_strain(),
            max_steel_strain=max_steel_strain,
        )

    def _get_pure_tension_point(self) -> BiaxialInteractionPoint:
        """Calculate pure tension point."""
        section_cx, section_cy = self.section.get_centroid()

        N_steel = 0.0
        My_steel = 0.0
        Mz_steel = 0.0
        max_steel_strain = 0.0

        for group_idx, group in enumerate(self.section.rebar_groups):
            A_s = group.rebar.area
            f_yd = self.steel_models[group_idx].get_yield_stress()

            for pos in group.positions:
                bar_force = -A_s * f_yd / 1000.0  # Negative for tension
                N_steel += bar_force

                x_offset = pos.x - section_cx
                y_offset = pos.y - section_cy

                My_steel += bar_force * y_offset / 1000.0
                Mz_steel += bar_force * x_offset / 1000.0

            max_steel_strain = max(max_steel_strain, self.steel_models[group_idx].epsilon_y)

        max_depth = max(self.section_width, self.section_height)

        return BiaxialInteractionPoint(
            N=N_steel,
            My=My_steel,
            Mz=Mz_steel,
            neutral_axis_depth=-max_depth * 10,
            neutral_axis_angle=0.0,
            max_concrete_strain=0.0,
            max_steel_strain=max_steel_strain,
        )

    def get_utilization_vector(
        self,
        N_Ed: float,
        My_Ed: float,
        Mz_Ed: float,
        surface_points: Optional[List] = None,
        n_angles: int = 72,
        n_axial_levels: int = 30,
    ) -> Tuple[bool, float]:
        """
        Check capacity using vector projection method for biaxial bending.

        Projects a vector from the origin through the applied load point (N_Ed, My_Ed, Mz_Ed)
        in 3D space and finds where it intersects the M-M-N interaction surface. The
        utilization ratio is the ratio of the distance to the applied load vs. distance
        to the capacity boundary.

        This is the geometrically correct method for biaxial M-M-N interaction checking
        as it properly accounts for the interaction between axial force and both moments.

        Method: Projects a ray from origin through (N_Ed, My_Ed, Mz_Ed) to find the
        intersection with the surface boundary (N_Rd, My_Rd, Mz_Rd).

        Args:
            N_Ed: Applied axial force in kN (positive = compression)
            My_Ed: Applied moment about y-axis (major axis) in kN·m
            Mz_Ed: Applied moment about z-axis (minor axis) in kN·m
            surface_points: Pre-generated surface points (optional). If provided, uses these
                           instead of regenerating the surface. Pass the result from
                           generate_surface() to avoid recomputation when checking multiple
                           load cases. If None, generates surface with n_angles and n_axial_levels.
            n_angles: Number of angles for surface generation (default: 72, ignored if surface_points provided)
            n_axial_levels: Number of N levels for surface generation (default: 30, ignored if surface_points provided)

        Returns:
            Tuple of (is_safe, utilization)
            - is_safe: True if utilization <= 1.0
            - utilization: ||(N_Ed, My_Ed, Mz_Ed)|| / ||(N_Rd, My_Rd, Mz_Rd)||
                          where (N_Rd, My_Rd, Mz_Rd) is the intersection point

        Example:
            >>> # Single load case (surface generated automatically)
            >>> surface = create_biaxial_interaction_surface(section, concrete)
            >>> is_safe, util = surface.get_utilization_vector(
            ...     N_Ed=1500,  # kN
            ...     My_Ed=100,  # kN·m
            ...     Mz_Ed=50    # kN·m
            ... )
            >>> print(f"Safe: {is_safe}, Utilization: {util:.1%}")

            >>> # Multiple load cases (reuse surface for efficiency)
            >>> surface_pts = surface.generate_surface(n_angles=72, n_axial_levels=30)
            >>> for load_case in load_cases:
            ...     is_safe, util = surface.get_utilization_vector(
            ...         N_Ed=load_case.N,
            ...         My_Ed=load_case.My,
            ...         Mz_Ed=load_case.Mz,
            ...         surface_points=surface_pts  # Reuse pre-generated surface
            ...     )
        """
        # Special case: origin point (no load)
        if abs(N_Ed) < 1e-6 and abs(My_Ed) < 1e-6 and abs(Mz_Ed) < 1e-6:
            return (True, 0.0)

        # Use provided surface or generate new one
        if surface_points is None:
            points = self.generate_surface(
                n_angles=n_angles,
                n_axial_levels=n_axial_levels,
                include_tension=True,
            )
        else:
            points = surface_points

        # Extract coordinates
        N_values = np.array([p.N for p in points])
        My_values = np.array([p.My for p in points])
        Mz_values = np.array([p.Mz for p in points])

        # Direction vector of the applied load
        load_direction = np.array([N_Ed, My_Ed, Mz_Ed])
        load_magnitude = np.linalg.norm(load_direction)

        if load_magnitude < 1e-10:
            return (True, 0.0)

        load_direction_unit = load_direction / load_magnitude

        # Find the maximum scaling factor alpha such that
        # alpha * (N_Ed, My_Ed, Mz_Ed) is still on the boundary surface
        #
        # We check all points on the surface and find which ones are roughly
        # aligned with the load direction, then find the maximum alpha

        max_alpha = 0.0

        # For each point on the surface, check if it's aligned with load direction
        for i in range(len(points)):
            surface_point = np.array([N_values[i], My_values[i], Mz_values[i]])
            surface_magnitude = np.linalg.norm(surface_point)

            if surface_magnitude < 1e-10:
                continue

            surface_direction_unit = surface_point / surface_magnitude

            # Check if directions are aligned (dot product close to 1)
            dot_product = np.dot(load_direction_unit, surface_direction_unit)

            # If aligned (within tolerance), calculate scaling factor
            if dot_product > 0.999:  # ~2.5 degree tolerance
                # Calculate alpha: surface_point = alpha * load_direction
                # alpha = ||surface_point|| / ||load_direction||
                alpha = surface_magnitude / load_magnitude
                max_alpha = max(max_alpha, alpha)

        # If we didn't find any aligned points, use a more robust search
        # by checking the nearest neighbor approach
        if max_alpha < 1e-10:
            # Project all surface points onto the load direction
            # and find the one with maximum projection
            projections = (
                N_values * N_Ed + My_values * My_Ed + Mz_values * Mz_Ed
            ) / (load_magnitude ** 2)

            # Filter to positive projections (same direction as load)
            positive_mask = projections > 0
            if np.any(positive_mask):
                max_alpha = np.max(projections[positive_mask])

        # If still no intersection found, point is likely outside
        if max_alpha < 1e-10:
            return (False, float('inf'))

        # Utilization ratio
        utilization = 1.0 / max_alpha
        is_safe = utilization <= 1.0

        # Convert to Python types (numpy operations may return numpy scalars)
        return (bool(is_safe), float(utilization))

    def get_capacity_vector(
        self,
        N_Ed: float,
        My_Ed: float,
        Mz_Ed: float,
        surface_points: Optional[List] = None,
        n_angles: int = 72,
        n_axial_levels: int = 30,
    ) -> Tuple[Optional[float], Optional[float], Optional[float], bool, float]:
        """
        Get capacity point (N_Rd, My_Rd, Mz_Rd) on the M-M-N surface using vector projection.

        This method finds where the ray from origin through (N_Ed, My_Ed, Mz_Ed) intersects
        the M-M-N interaction surface boundary, returning the capacity coordinates.

        Args:
            N_Ed: Applied axial force in kN (positive = compression)
            My_Ed: Applied moment about y-axis (major axis) in kN·m
            Mz_Ed: Applied moment about z-axis (minor axis) in kN·m
            surface_points: Pre-generated surface points (optional)
            n_angles: Number of angles for surface generation (if surface_points not provided)
            n_axial_levels: Number of N levels for surface generation (if surface_points not provided)

        Returns:
            Tuple of (N_Rd, My_Rd, Mz_Rd, is_safe, utilization)
            - N_Rd: Design axial capacity at intersection (kN) or None if no intersection
            - My_Rd: Design moment capacity about y-axis (kN·m) or None if no intersection
            - Mz_Rd: Design moment capacity about z-axis (kN·m) or None if no intersection
            - is_safe: True if utilization <= 1.0
            - utilization: ||(N_Ed, My_Ed, Mz_Ed)|| / ||(N_Rd, My_Rd, Mz_Rd)||

        Example:
            >>> surface = create_biaxial_interaction_surface(section, concrete)
            >>> N_Rd, My_Rd, Mz_Rd, is_safe, util = surface.get_capacity_vector(
            ...     N_Ed=1000, My_Ed=150, Mz_Ed=100
            ... )
            >>> print(f"Capacity: N_Rd={N_Rd:.1f} kN, My_Rd={My_Rd:.1f} kN·m, Mz_Rd={Mz_Rd:.1f} kN·m")
            >>> print(f"Utilization: {util:.1%}")
        """
        # Special case: origin point (no load)
        if abs(N_Ed) < 1e-6 and abs(My_Ed) < 1e-6 and abs(Mz_Ed) < 1e-6:
            return (0.0, 0.0, 0.0, True, 0.0)

        # Use provided surface or generate new one
        if surface_points is None:
            points = self.generate_surface(
                n_angles=n_angles,
                n_axial_levels=n_axial_levels,
                include_tension=True,
            )
        else:
            points = surface_points

        # Extract coordinates
        N_values = np.array([p.N for p in points])
        My_values = np.array([p.My for p in points])
        Mz_values = np.array([p.Mz for p in points])

        # Direction vector of the applied load
        load_direction = np.array([N_Ed, My_Ed, Mz_Ed])
        load_magnitude = np.linalg.norm(load_direction)

        if load_magnitude < 1e-10:
            return (0.0, 0.0, 0.0, True, 0.0)

        load_direction_unit = load_direction / load_magnitude

        # Find the maximum scaling factor alpha
        max_alpha = 0.0

        # For each point on the surface, check if it's aligned with load direction
        for i in range(len(points)):
            surface_point = np.array([N_values[i], My_values[i], Mz_values[i]])
            surface_magnitude = np.linalg.norm(surface_point)

            if surface_magnitude < 1e-10:
                continue

            surface_direction_unit = surface_point / surface_magnitude

            # Check if directions are aligned (dot product close to 1)
            dot_product = np.dot(load_direction_unit, surface_direction_unit)

            # If aligned (within tolerance), calculate scaling factor
            if dot_product > 0.999:  # ~2.5 degree tolerance
                alpha = surface_magnitude / load_magnitude
                max_alpha = max(max_alpha, alpha)

        # If we didn't find any aligned points, use projection approach
        if max_alpha < 1e-10:
            projections = (
                N_values * N_Ed + My_values * My_Ed + Mz_values * Mz_Ed
            ) / (load_magnitude ** 2)

            positive_mask = projections > 0
            if np.any(positive_mask):
                max_alpha = np.max(projections[positive_mask])

        # If still no intersection found
        if max_alpha < 1e-10:
            return (None, None, None, False, float('inf'))

        # Calculate capacity point coordinates
        N_Rd = max_alpha * N_Ed
        My_Rd = max_alpha * My_Ed
        Mz_Rd = max_alpha * Mz_Ed

        # Utilization ratio
        utilization = 1.0 / max_alpha
        is_safe = utilization <= 1.0

        # Convert to Python types
        return (float(N_Rd), float(My_Rd), float(Mz_Rd), bool(is_safe), float(utilization))

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

        # Generate surface points
        print(f"Generating surface with {n_angles} angles and {n_axial_levels} N levels...")
        surface_pts = self.generate_surface(
            n_angles=n_angles,
            n_axial_levels=n_axial_levels,
            include_tension=True
        )
        print(f"[OK] Generated {len(surface_pts)} surface points")

        # Extract coordinates
        N_surf = np.array([p.N for p in surface_pts])
        My_surf = np.array([p.My for p in surface_pts])
        Mz_surf = np.array([p.Mz for p in surface_pts])
        angles = np.array([p.neutral_axis_angle for p in surface_pts])

        # Filter out interior points at each N level using convex hull
        print("Filtering interior points to ensure clean surface...")
        from scipy.spatial import ConvexHull

        filtered_indices = []
        unique_N = np.unique(N_surf.round(decimals=1))

        for N_level in unique_N:
            # Find points at this N level
            mask = np.abs(N_surf - N_level) < 10.0
            indices = np.where(mask)[0]

            if len(indices) < 3:
                continue

            # Get My, Mz coordinates for this slice
            My_slice = My_surf[indices]
            Mz_slice = Mz_surf[indices]

            # Create 2D points array
            points_2d = np.column_stack([My_slice, Mz_slice])

            # Compute convex hull if we have enough points
            if len(points_2d) >= 3:
                try:
                    hull = ConvexHull(points_2d)
                    # Keep only hull vertices
                    hull_indices = indices[hull.vertices]
                    filtered_indices.extend(hull_indices)
                except:
                    # If convex hull fails, keep all points
                    filtered_indices.extend(indices)
            else:
                filtered_indices.extend(indices)

        # Use filtered points
        filtered_indices = np.array(filtered_indices)
        N_surf = N_surf[filtered_indices]
        My_surf = My_surf[filtered_indices]
        Mz_surf = Mz_surf[filtered_indices]
        angles = angles[filtered_indices]
        surface_pts_filtered = [surface_pts[i] for i in filtered_indices]

        print(f"[OK] Filtered to {len(filtered_indices)} boundary points")

        # Find pole points (pure compression and pure tension)
        N_max_idx = np.argmax(N_surf)
        N_min_idx = np.argmin(N_surf)

        compression_pole = (My_surf[N_max_idx], Mz_surf[N_max_idx], N_surf[N_max_idx])
        tension_pole = (My_surf[N_min_idx], Mz_surf[N_min_idx], N_surf[N_min_idx])

        # Create figure
        fig = go.Figure()

        # Add latitude rings (constant N contours) with faint black lines FIRST (so they don't block)
        # Group points by N level
        unique_N_filtered = np.unique(N_surf.round(decimals=1))

        for N_level in unique_N_filtered:
            # Find points at this N level (with tolerance)
            mask = np.abs(N_surf - N_level) < 10.0  # 10 kN tolerance
            if np.sum(mask) < 3:  # Need at least 3 points for a ring
                continue

            My_ring = My_surf[mask]
            Mz_ring = Mz_surf[mask]
            N_ring = N_surf[mask]
            angles_ring = angles[mask]

            # Sort by angle to create continuous ring
            sort_idx = np.argsort(angles_ring)
            My_ring = My_ring[sort_idx]
            Mz_ring = Mz_ring[sort_idx]
            N_ring = N_ring[sort_idx]

            # Close the ring
            My_ring = np.append(My_ring, My_ring[0])
            Mz_ring = np.append(Mz_ring, Mz_ring[0])
            N_ring = np.append(N_ring, N_ring[0])

            fig.add_trace(go.Scatter3d(
                x=My_ring,
                y=Mz_ring,
                z=N_ring,
                mode='lines',
                line=dict(color='black', width=1),
                showlegend=False,
                hoverinfo='skip',
            ))

        # Add longitude lines (constant angle rays from tension pole through to compression pole)
        unique_angles_filtered = np.unique(angles.round(decimals=1))

        for angle in unique_angles_filtered[::max(1, len(unique_angles_filtered) // 36)]:  # Limit number of longitude lines
            # Find points at this angle
            mask = np.abs(angles - angle) < 1.0  # 1 degree tolerance
            indices_at_angle = np.where(mask)[0]

            if len(indices_at_angle) < 1:
                continue

            My_line = My_surf[mask]
            Mz_line = Mz_surf[mask]
            N_line = N_surf[mask]

            # Sort by N to create continuous line from tension to compression
            sort_idx = np.argsort(N_line)
            My_line = My_line[sort_idx]
            Mz_line = Mz_line[sort_idx]
            N_line = N_line[sort_idx]

            # Add tension pole at the start
            My_line = np.insert(My_line, 0, tension_pole[0])
            Mz_line = np.insert(Mz_line, 0, tension_pole[1])
            N_line = np.insert(N_line, 0, tension_pole[2])

            # Add compression pole at the end
            My_line = np.append(My_line, compression_pole[0])
            Mz_line = np.append(Mz_line, compression_pole[1])
            N_line = np.append(N_line, compression_pole[2])

            fig.add_trace(go.Scatter3d(
                x=My_line,
                y=Mz_line,
                z=N_line,
                mode='lines',
                line=dict(color='black', width=1),
                showlegend=False,
                hoverinfo='skip',
            ))

        # Add origin point (smaller size)
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

                # Get capacity and utilization using FILTERED surface points
                N_Rd, My_Rd, Mz_Rd, is_safe, utilization = self.get_capacity_vector(
                    N_Ed=N_Ed, My_Ed=My_Ed, Mz_Ed=Mz_Ed,
                    surface_points=surface_pts_filtered,  # Use filtered points!
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

        # Add translucent surface LAST (so it doesn't block hover on load points)
        fig.add_trace(go.Mesh3d(
            x=My_surf,
            y=Mz_surf,
            z=N_surf,
            opacity=0.3,
            color='lightblue',
            alphahull=0,  # Use Delaunay triangulation
            name='M-M-N Surface',
            showlegend=True,
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
        points = self.generate_surface(n_angles=n_angles, n_axial_levels=n_axial_levels)

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
        points = self.generate_surface(n_angles=n_angles, n_axial_levels=n_axial_levels)

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
        >>> points = surface.generate_surface(n_angles=60, n_axial_levels=50)
    """
    return BiaxialMNInteractionSurface(section=section, concrete=concrete, **kwargs)
