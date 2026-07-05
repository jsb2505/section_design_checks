"""
M-N interaction diagram generator using fiber-based strain compatibility.

Implements EC2 ultimate limit state analysis for combined axial force and bending.
"""

from typing import List, Tuple, Optional, Literal, Dict, Any
import json
import csv
from pathlib import Path
import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, Field, ConfigDict
from scipy.optimize import root_scalar

from materials.reinforced_concrete.geometry import RCSection, FiberMesh
from materials.reinforced_concrete.constitutive import (
    create_concrete_stress_strain,
    create_steel_stress_strain,
)
from materials.reinforced_concrete.materials import ConcreteMaterial


class InteractionPoint(BaseModel):
    """Single point on M-N interaction diagram."""

    model_config = ConfigDict(frozen=True)

    N: float = Field(..., description="Axial force in kN (positive = compression)")
    M: float = Field(..., description="Moment about centroid in kN·m")
    neutral_axis_depth: float = Field(..., description="Neutral axis depth from top (mm)")
    max_concrete_strain: float = Field(..., description="Maximum concrete strain")
    max_steel_strain: float = Field(..., description="Maximum steel strain")

    def __repr__(self) -> str:
        return f"InteractionPoint(N={self.N:.1f} kN, M={self.M:.1f} kN·m)"

    def to_dict(self) -> Dict[str, Any]:
        """
        Export interaction point to dictionary.

        Returns:
            Dictionary with all point properties
        """
        return {
            "N_kN": self.N,
            "M_kNm": self.M,
            "neutral_axis_depth_mm": self.neutral_axis_depth,
            "max_concrete_strain": self.max_concrete_strain,
            "max_steel_strain": self.max_steel_strain,
        }


class MNInteractionDiagram:
    """
    M-N interaction diagram generator using fiber-based strain compatibility.

    The diagram represents all combinations of axial force (N) and moment (M)
    that bring the section to its ultimate limit state per EC2.

    Method:
    1. Assume a neutral axis depth
    2. Calculate strain distribution (plane sections remain plane)
    3. Get stresses from constitutive models
    4. Integrate forces over fibers
    5. Result is one (N, M) point on the diagram
    6. Repeat for different neutral axis depths

    Special cases handled:
    - Pure compression (NA at infinity)
    - Pure tension (NA at -infinity)
    - Balanced failure
    - Small/large eccentricity
    """

    def __init__(
        self,
        section: RCSection,
        concrete: ConcreteMaterial,
        concrete_model_type: Literal["parabola-rectangle", "bilinear"] = "parabola-rectangle",
        steel_branch_type: Literal["inclined", "horizontal"] = "inclined",
        n_fibers_width: int = 20,
        n_fibers_height: int = 30,
        tension_stiffening: bool = False,
        confined_concrete: bool = False,
        confinement_rho_s: Optional[float] = None,
        confinement_f_yh: Optional[float] = None,
    ):
        """
        Initialize M-N diagram generator.

        Args:
            section: RC section with reinforcement
            concrete: Concrete material properties
            concrete_model_type: Stress-strain model for concrete
            steel_branch_type: Stress-strain model for steel
            n_fibers_width: Fiber mesh resolution (width)
            n_fibers_height: Fiber mesh resolution (height)
            tension_stiffening: Include tension stiffening effects (default: False)
                If True, concrete in tension contributes using EC2 average stress-strain
            confined_concrete: Use Mander confined concrete model (default: False)
                If True, enhances strength and ductility in compression zone
            confinement_rho_s: Volumetric ratio of transverse reinforcement (optional)
                Required if confined_concrete=True. Typically 0.01-0.03
            confinement_f_yh: Yield strength of transverse reinforcement in MPa (optional)
                Defaults to same as longitudinal steel if not provided
        """
        self.section = section
        self.concrete = concrete
        self.tension_stiffening = tension_stiffening
        self.confined_concrete = confined_concrete
        self.confinement_rho_s = confinement_rho_s
        self.confinement_f_yh = confinement_f_yh

        # Create constitutive models (using design strengths)
        self.concrete_model = create_concrete_stress_strain(
            concrete=concrete,
            model_type=concrete_model_type,
            use_characteristic=False,  # Use f_cd
        )

        # Create steel models for each rebar group (to support different steel grades)
        if len(section.rebar_groups) == 0:
            raise ValueError("Section must have at least one rebar group")

        # Create a steel model for each rebar group
        self.steel_models = []
        for group in section.rebar_groups:
            steel_model = create_steel_stress_strain(
                steel=group.rebar,
                branch_type=steel_branch_type,
                use_characteristic=False,  # Use f_yd
            )
            self.steel_models.append(steel_model)

        # Validate confined concrete parameters
        if self.confined_concrete:
            if self.confinement_rho_s is None:
                raise ValueError(
                    "confinement_rho_s must be provided when confined_concrete=True"
                )
            if self.confinement_rho_s <= 0 or self.confinement_rho_s > 0.1:
                raise ValueError(
                    f"confinement_rho_s must be between 0 and 0.1, got {self.confinement_rho_s}"
                )
            # Default to longitudinal steel yield strength if not provided
            if self.confinement_f_yh is None:
                self.confinement_f_yh = section.rebar_groups[0].rebar.f_yd

        # Generate fiber mesh
        self.mesh = FiberMesh(
            section=section,
            n_fibers_width=n_fibers_width,
            n_fibers_height=n_fibers_height,
            exclude_steel_area=True,
        )

        # Get section properties
        _, self.section_centroid_y = section.get_centroid()
        _, min_y, _, max_y = section.get_bounding_box()
        self.section_height = max_y - min_y
        self.section_top = max_y
        self.section_bottom = min_y

    def calculate_point(
        self,
        neutral_axis_depth: float,
        max_concrete_strain: Optional[float] = None,
        compression_from_bottom: bool = False,
    ) -> InteractionPoint:
        """
        Calculate single point on interaction diagram.

        Uses strain compatibility:
        - Plane sections remain plane
        - Strain varies linearly from neutral axis
        - Maximum concrete strain at compression face

        Args:
            neutral_axis_depth: Depth from compression face to neutral axis (mm)
                                Positive = NA inside section
                                Negative = NA beyond section (opposite side in tension)
                                Very large = NA beyond section (pure compression)
            max_concrete_strain: Maximum concrete compressive strain (default: ε_cu2)
            compression_from_bottom: If True, compression is applied from bottom (creates negative moments)
                                     If False, compression is applied from top (creates positive moments)

        Returns:
            InteractionPoint with N, M, and strain information
        """
        if max_concrete_strain is None:
            max_concrete_strain = self.concrete_model.get_ultimate_strain()

        # Get fiber coordinates
        x, y, area, material_type, material_index = self.mesh.get_fiber_arrays()

        # Calculate strain at each fiber using plane sections remain plane
        # ε = ε_max * (NA_depth - distance_from_compression_face) / NA_depth

        if compression_from_bottom:
            # Compression applied from BOTTOM (creates negative moments)
            # Distance from bottom of section to each fiber
            distance_from_compression_face = y - self.section_bottom
        else:
            # Compression applied from TOP (creates positive moments)
            # Distance from top of section to each fiber
            distance_from_compression_face = self.section_top - y

        # Strain distribution (compression positive)
        if neutral_axis_depth > 0:
            # NA inside or beyond section on tension side
            strains = max_concrete_strain * (neutral_axis_depth - distance_from_compression_face) / neutral_axis_depth
        else:
            # NA beyond section on compression side (entire section in tension)
            # Use similar triangles with NA beyond compression face
            # Avoid division by zero when NA is at section boundary
            na_abs = abs(neutral_axis_depth)
            if na_abs < 1e-6:
                # NA at boundary: assume uniform tension strain
                strains = np.full_like(distance_from_compression_face, -max_concrete_strain)
            else:
                strains = -max_concrete_strain * distance_from_compression_face / na_abs

        # Calculate stresses from constitutive models
        stresses = np.zeros_like(strains)

        # Concrete fibers
        concrete_mask = material_type == 'concrete'
        concrete_strains = strains[concrete_mask]
        concrete_stresses = self.concrete_model.get_stress_array(concrete_strains)

        # Apply confined concrete model if enabled (Mander model)
        if self.confined_concrete:
            # Type narrowing: confined_concrete=True guarantees these are not None (validated in __init__)
            assert self.confinement_rho_s is not None
            assert self.confinement_f_yh is not None

            # Store in local variables for type clarity and to avoid repeated property access
            rho_s = self.confinement_rho_s
            f_yh = self.confinement_f_yh

            # Mander confined concrete model for compression
            # Enhances strength and ductility based on transverse reinforcement
            compression_mask = concrete_strains > 0  # Positive strain = compression

            if np.any(compression_mask):
                # Mander model parameters
                f_co = self.concrete.f_cd  # Unconfined strength
                epsilon_co = self.concrete.epsilon_c2  # Unconfined strain at peak

                # Confinement effectiveness coefficient (typically 0.95 for circular, 0.75 for rectangular)
                k_e = 0.75  # Conservative for rectangular sections

                # Effective lateral confining pressure
                f_l = 0.5 * k_e * rho_s * f_yh

                # Confined strength (Mander model)
                # f_cc = f_co * (1 + 5 * (f_cc/f_co - 1))
                # Simplified: f_cc = f_co * (2.254 * sqrt(1 + 7.94*f_l/f_co) - 2*f_l/f_co - 1.254)
                f_cc = f_co * (
                    2.254 * np.sqrt(1 + 7.94 * f_l / f_co) - 2 * f_l / f_co - 1.254
                )

                # Confined strain at peak stress
                epsilon_cc = epsilon_co * (
                    1 + 5 * (f_cc / f_co - 1)
                )

                # Ultimate confined strain (Mander)
                epsilon_cu_confined = 0.004 + 0.14 * rho_s * f_yh / f_co

                # Mander stress-strain relationship for confined concrete
                compression_strains = concrete_strains[compression_mask]

                # Normalized variables
                x = compression_strains / epsilon_cc
                r = self.concrete.E_cm / (self.concrete.E_cm - f_cc / epsilon_cc)

                # Mander equation: f_c = f_cc * x * r / (r - 1 + x^r)
                confined_stresses = np.where(
                    compression_strains <= epsilon_cu_confined,
                    f_cc * x * r / (r - 1 + x**r),
                    0.0  # Zero stress beyond ultimate strain
                )

                # Update concrete stresses in compression
                concrete_stresses[compression_mask] = confined_stresses

        # Apply tension stiffening if enabled
        if self.tension_stiffening:
            # EC2 average tension stress-strain for cracked concrete
            # Convention: tension = negative stress, compression = positive stress
            # Uses simplified approach: σ_ct = -f_ctm * (1 - β * (ε_t - ε_cr) / ε_cr)
            # where β ≈ 0.6 for short-term loading, ε_cr = f_ctm / E_cm

            tension_mask = concrete_strains < 0  # Negative strain = tension
            if np.any(tension_mask):
                # Tensile strength (mean)
                f_ctm = self.concrete.f_ctm
                E_cm = self.concrete.E_cm

                # Cracking strain (positive value)
                epsilon_cr = f_ctm / E_cm

                # Tension stiffening factor (0.6 for short-term, 0.4 for sustained)
                beta = 0.6

                # Get absolute values of tension strains (make positive)
                tension_strains_abs = -concrete_strains[tension_mask]

                # Linear elastic up to cracking, then tension stiffening decay
                # Before cracking: stress = -E_cm * ε_t (linear elastic tension)
                # After cracking: stress = -f_ctm * (1 - β * (ε_t - ε_cr) / ε_cr)
                tension_stresses = np.where(
                    tension_strains_abs <= epsilon_cr,
                    # Linear elastic (negative stress = tension)
                    -E_cm * tension_strains_abs,
                    # Tension stiffening (decreasing tension contribution)
                    -f_ctm * np.maximum(0, 1.0 - beta * (tension_strains_abs - epsilon_cr) / (epsilon_cr * 5.0))
                )

                # Update concrete stresses for fibers in tension
                concrete_stresses[tension_mask] = tension_stresses

        stresses[concrete_mask] = concrete_stresses

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
        N, M = self.mesh.calculate_section_forces(stresses)

        # Get maximum strains for reporting
        max_conc_strain = np.max(strains[concrete_mask]) if np.any(concrete_mask) else 0.0
        max_steel_strain = np.max(np.abs(strains[steel_mask])) if np.any(steel_mask) else 0.0

        return InteractionPoint(
            N=N,
            M=M,
            neutral_axis_depth=neutral_axis_depth,
            max_concrete_strain=max_conc_strain,
            max_steel_strain=max_steel_strain,
        )

    def generate_diagram(
        self,
        n_points: int = 100,
        include_tension: bool = True,
    ) -> List[InteractionPoint]:
        """
        Generate complete M-N interaction diagram.

        Creates TWO curves and combines them:
        1. Compression from TOP (positive moments) - top in compression, bottom in tension
        2. Compression from BOTTOM (negative moments) - bottom in compression, top in tension

        This ensures the full M-N envelope is captured for asymmetric sections.

        Args:
            n_points: Number of points on diagram (divided between both curves)
            include_tension: Include pure tension branch

        Returns:
            List of InteractionPoint ordered by N (descending) for continuous plotting
        """
        points: List[InteractionPoint] = []

        # Use half the points for each curve direction
        points_per_curve = n_points // 2

        # =======================================================================
        # CURVE 1: Compression from TOP (creates positive moments)
        # =======================================================================

        # Use denser, more uniform sampling for smoother curves
        # NA sweep from deep compression through to tension

        # For a proper convex M-N diagram, only include points where there IS a compression zone
        # Pure tension points (NA < 0) create concave regions and are not part of ULS envelope

        # Symmetric sampling around peak moment region + dense sampling at poles
        # Peak M typically occurs around NA depth ~0.25h to 0.35h (balanced failure, N ~ 30% of N_max)
        # Need equal point density above and below peak for symmetric M-N curve
        # ALSO need dense sampling at poles (pure compression/tension) where curves join with sharp curvature change

        # For poles: use LOGARITHMIC spacing in NA depth to get better distribution in N
        # (since N vs NA depth is asymptotic near poles)
        na_depths_top = np.concatenate([
            # AT pure compression pole - logarithmic spacing for uniform N distribution
            np.logspace(np.log10(self.section_height * 50), np.log10(self.section_height * 3), num=max(15, points_per_curve // 4)),
            # Deep compression (moment increasing, approaching peak from above)
            np.linspace(self.section_height * 3, self.section_height * 0.8, num=max(10, points_per_curve // 6)),
            # Upper balanced region (high curvature, approaching peak M from compression side)
            np.linspace(self.section_height * 0.8, self.section_height * 0.25, num=max(16, points_per_curve // 4)),
            # Core balanced region (peak moment zone - highest curvature)
            np.linspace(self.section_height * 0.25, self.section_height * 0.12, num=max(20, points_per_curve // 3)),
            # Lower balanced region (high curvature, leaving peak M toward tension side)
            np.linspace(self.section_height * 0.12, self.section_height * 0.04, num=max(16, points_per_curve // 4)),
            # Tension controlled (moment decreasing, mirroring deep compression)
            np.linspace(self.section_height * 0.04, self.section_height * 0.01, num=max(10, points_per_curve // 6)),
            # AT pure tension pole - logarithmic spacing for uniform N distribution
            # Need many points here to bridge to pure tension smoothly
            np.logspace(np.log10(self.section_height * 0.01), np.log10(self.section_height * 0.00001), num=max(20, points_per_curve // 3)),
        ])

        for na_depth in na_depths_top:
            points.append(self.calculate_point(na_depth, compression_from_bottom=False))

        # =======================================================================
        # CURVE 2: Compression from BOTTOM (creates negative moments)
        # =======================================================================

        # Use same NA depths but compression from bottom
        for na_depth in na_depths_top:
            points.append(self.calculate_point(na_depth, compression_from_bottom=True))

        # Add pure compression point to close the diagram at the top
        # Pure compression: entire section at ultimate concrete strain
        # For asymmetric reinforcement, this creates a moment due to steel eccentricity
        # Similar to pure tension, calculate N and M from concrete + steel contributions

        section_cx, section_cy = self.section.get_centroid()

        # Concrete contribution (uniform compression at design/mean strength over entire area)
        concrete_area = self.section.outline.area  # mm²
        f_c = self.concrete_model.get_yield_stress()  # Design or mean strength depending on model
        N_concrete = concrete_area * f_c / 1000.0  # kN
        M_concrete = 0.0  # Symmetric concrete stress creates no moment about centroid

        # Steel contribution (all steel at yield in compression)
        N_steel = 0.0
        M_steel = 0.0
        max_steel_strain = 0.0

        for group_idx, group in enumerate(self.section.rebar_groups):
            A_s = group.rebar.area  # Area per bar
            f_yd = self.steel_models[group_idx].get_yield_stress()  # Yield stress for this group

            for pos in group.positions:
                # Compression force (positive)
                bar_force = A_s * f_yd / 1000.0  # kN (positive for compression)
                N_steel += bar_force

                # Moment about section centroid
                y_offset = pos.y - section_cy
                M_steel += bar_force * y_offset / 1000.0  # kN⋅m

            # Track maximum steel yield strain across all groups
            max_steel_strain = max(max_steel_strain, self.steel_models[group_idx].epsilon_y)

        pure_compression_N = N_concrete + N_steel
        pure_compression_M = M_concrete + M_steel

        pure_compression_point = InteractionPoint(
            N=pure_compression_N,
            M=pure_compression_M,
            neutral_axis_depth=self.section_height * 1000,  # NA very deep
            max_concrete_strain=self.concrete_model.get_ultimate_strain(),
            max_steel_strain=max_steel_strain,
        )
        points.append(pure_compression_point)

        # Add pure tension point to close the diagram at the bottom
        # This represents all steel in tension (yielding), concrete cracked throughout
        if include_tension:
            # Calculate pure tension capacity: all steel at yield stress in tension
            # N = -Σ(A_s × f_yd)
            # M = Σ(A_s × f_yd × y_offset) where y_offset is distance from section centroid

            section_cx, section_cy = self.section.get_centroid()
            pure_tension_N = 0.0
            pure_tension_M = 0.0
            max_steel_strain = 0.0

            for group_idx, group in enumerate(self.section.rebar_groups):
                A_s = group.rebar.area  # Area per bar
                f_yd = self.steel_models[group_idx].get_yield_stress()  # Yield stress for this group

                # Each bar contributes to N and M
                for pos in group.positions:
                    # Tension force (negative)
                    bar_force = -A_s * f_yd / 1000.0  # kN (negative for tension)
                    pure_tension_N += bar_force

                    # Moment about section centroid
                    y_offset = pos.y - section_cy
                    pure_tension_M += bar_force * y_offset / 1000.0  # kN⋅m

                # Track maximum steel yield strain across all groups
                max_steel_strain = max(max_steel_strain, self.steel_models[group_idx].epsilon_y)

            # Create pure tension point
            pure_tension_point = InteractionPoint(
                N=pure_tension_N,
                M=pure_tension_M,
                neutral_axis_depth=-self.section_height * 10,  # NA far above section
                max_concrete_strain=0.0,
                max_steel_strain=max_steel_strain,
            )
            points.append(pure_tension_point)

        # Use convex hull to get the true capacity envelope
        # This filters out interior points and handles the curve junction at poles correctly
        # The convex hull ensures a smooth rugby ball shape by removing crossing/interior points

        from scipy.spatial import ConvexHull

        # Convert points to array for convex hull calculation
        points_array = np.array([[p.M, p.N] for p in points])

        # Compute convex hull
        try:
            hull = ConvexHull(points_array)

            # Extract points on the hull in order
            hull_indices = hull.vertices

            # Sort hull vertices to trace around the perimeter
            # Start from point with maximum N (pure compression)
            max_n_idx = np.argmax([points[i].N for i in hull_indices])
            start_idx = hull_indices[max_n_idx]

            # Reorder hull_indices to start from max N and go counterclockwise
            # This traces: compression → +M → tension → -M → back to compression
            start_pos = np.where(hull_indices == start_idx)[0][0]
            hull_indices_ordered = np.roll(hull_indices, -start_pos)

            # Build ordered list of hull points
            hull_points = [points[i] for i in hull_indices_ordered]

            # Ensure we trace the correct direction (should go through positive M first)
            # Check if second point has positive or negative M
            if len(hull_points) > 1 and hull_points[1].M < 0:
                # Reverse the order (we're going the wrong way)
                hull_points = [hull_points[0]] + hull_points[1:][::-1]

            # Add the first point at the end to explicitly close the loop
            # This ensures matplotlib draws a complete closed curve
            if len(hull_points) > 0:
                hull_points.append(hull_points[0])

            return hull_points

        except Exception as e:
            # If convex hull fails (e.g., collinear points), fall back to simple ordering
            print(f"Warning: Convex hull computation failed ({e}), using simple ordering")

            # Split into the two curves
            n_curve_points = len(na_depths_top)
            curve_positive_m = points[:n_curve_points]
            curve_negative_m = points[n_curve_points:n_curve_points * 2]

            # Simple ordering fallback
            points_sorted = curve_positive_m + list(reversed(curve_negative_m))

            # Close the loop
            if len(points_sorted) > 0:
                points_sorted.append(points_sorted[0])

            return points_sorted

    def get_capacity_fixed_n(
        self,
        N_Ed: float,
        n_points: int = 100
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Get moment capacity for given axial force using fixed-N method.

        Finds the maximum moment capacity on the interaction diagram at a
        specific axial force level by interpolating along the M-N boundary curve.
        This is the traditional approach where axial force is known/fixed and
        moment capacity is determined.

        Method: Takes a horizontal slice through the M-N diagram at the given N value.

        Args:
            N_Ed: Applied axial force in kN (positive = compression)
            n_points: Number of points to form the M-N curve from

        Returns:
            Tuple of (M_Rd_pos, M_Rd_neg) - moment capacity in kN·m
            M_Rd_pos: Maximum positive moment capacity at this N (None if N_Ed out of bounds)
            M_Rd_neg: Maximum negative moment capacity at this N (None if N_Ed out of bounds)

            Returns (None, None) if N_Ed is outside the interaction diagram bounds.
        """
        # Generate diagram (returns closed convex hull boundary)
        diagram = self.generate_diagram(n_points=n_points)

        # Extract N and M values
        N_values = np.array([p.N for p in diagram])
        M_values = np.array([p.M for p in diagram])

        # Check if N_Ed is within the overall diagram bounds
        N_min_global = np.min(N_values)
        N_max_global = np.max(N_values)

        if N_Ed < N_min_global or N_Ed > N_max_global:
            # N_Ed is outside the interaction diagram capacity
            return (None, None)

        # Split the curve into positive M and negative M sides
        # The curve is ordered as a closed loop, so we need to separate the two sides

        # Find indices for positive and negative M regions
        pos_M_indices = []
        neg_M_indices = []

        for i, M in enumerate(M_values):
            if M >= 0:
                pos_M_indices.append(i)
            else:
                neg_M_indices.append(i)

        # Helper function to interpolate M for given N on a curve segment
        def interpolate_capacity(indices, N_target):
            if len(indices) == 0:
                return 0.0

            N_seg = N_values[indices]
            M_seg = M_values[indices]

            # Check if N_target is within the range of this segment
            N_min, N_max = np.min(N_seg), np.max(N_seg)

            if N_target < N_min or N_target > N_max:
                # N_target is outside this segment's range
                # Return 0 (this can happen for one side when N_Ed is near pure compression/tension)
                return 0.0

            # Find the two points that bracket N_target
            # Sort by N to enable bracketing search
            sorted_indices = np.argsort(N_seg)
            N_sorted = N_seg[sorted_indices]
            M_sorted = M_seg[sorted_indices]

            # Find bracketing indices
            # searchsorted finds where to insert N_target to maintain order
            insert_idx = np.searchsorted(N_sorted, N_target)

            if insert_idx == 0:
                # N_target is at or below minimum
                return M_sorted[0]
            elif insert_idx >= len(N_sorted):
                # N_target is at or above maximum
                return M_sorted[-1]
            else:
                # Interpolate between points
                N1, N2 = N_sorted[insert_idx - 1], N_sorted[insert_idx]
                M1, M2 = M_sorted[insert_idx - 1], M_sorted[insert_idx]

                if abs(N2 - N1) < 1e-6:
                    # Avoid division by zero
                    return (M1 + M2) / 2.0

                # Linear interpolation
                alpha = (N_target - N1) / (N2 - N1)
                M_interp = M1 + alpha * (M2 - M1)
                return M_interp

        # Interpolate capacity on both sides
        M_Rd_pos = interpolate_capacity(pos_M_indices, N_Ed)
        M_Rd_neg = interpolate_capacity(neg_M_indices, N_Ed)

        # Convert numpy scalars to Python float for type consistency
        return (float(M_Rd_pos), float(M_Rd_neg))

    def get_utilization_vector(
        self,
        N_Ed: float,
        M_Ed: float,
        n_points: int = 100
    ) -> Tuple[bool, float]:
        """
        Check capacity using vector projection method (load ratio approach).

        Projects a vector from the origin through the applied load point (M_Ed, N_Ed)
        and finds where it intersects the M-N boundary curve. The utilization ratio
        is the ratio of the distance to the applied load vs. distance to the boundary.

        This is the geometrically correct method for M-N interaction checking as it
        properly accounts for the interaction between axial force and moment.

        Method: Projects a ray from origin through (M_Ed, N_Ed) to find (M_Rd, N_Rd).

        WARNING: This gives different results than get_capacity_fixed_n() because they
        use fundamentally different approaches. Do not mix methods - i.e., don't take
        M from get_capacity_fixed_n() and expect 50% utilization when checking at 0.5*M.

        Args:
            N_Ed: Applied axial force in kN (positive = compression)
            M_Ed: Applied moment in kN·m
            n_points: Number of points to form the M-N curve from

        Returns:
            Tuple of (is_safe, utilization)
            - is_safe: True if utilization <= 1.0
            - utilization: ||(M_Ed, N_Ed)|| / ||(M_Rd, N_Rd)|| where (M_Rd, N_Rd) is
                          the intersection point on the boundary
        """
        # Generate diagram (returns closed convex hull boundary)
        diagram = self.generate_diagram(n_points=n_points)

        # Extract N and M values
        N_values = np.array([p.N for p in diagram])
        M_values = np.array([p.M for p in diagram])

        # Special case: origin point (no load)
        if abs(M_Ed) < 1e-6 and abs(N_Ed) < 1e-6:
            return (True, 0.0)

        # Find intersection of ray from origin through (M_Ed, N_Ed) with boundary
        # Ray equation: (M, N) = alpha * (M_Ed, N_Ed) for alpha >= 0

        max_alpha = 0.0  # Maximum scaling factor where boundary is intersected

        # Check intersection with each edge of the boundary polygon
        n_points = len(M_values)
        for i in range(n_points):
            # Edge from point i to point (i+1) % n_points
            M1, N1 = M_values[i], N_values[i]
            M2, N2 = M_values[(i + 1) % n_points], N_values[(i + 1) % n_points]

            # Parametric form of boundary edge: (M, N) = (M1, N1) + s * ((M2, N2) - (M1, N1))
            # Parametric form of ray: (M, N) = alpha * (M_Ed, N_Ed)
            #
            # Solve: alpha * M_Ed = M1 + s * (M2 - M1)
            #        alpha * N_Ed = N1 + s * (N2 - N1)

            dM = M2 - M1
            dN = N2 - N1

            # Matrix form: [M_Ed, -dM] [alpha] = [M1]
            #              [N_Ed, -dN] [s    ]   [N1]

            det = M_Ed * (-dN) - N_Ed * (-dM)

            if abs(det) < 1e-10:
                # Lines are parallel or nearly parallel
                continue

            # Solve using Cramer's rule
            alpha = (M1 * (-dN) - N1 * (-dM)) / det
            s = (M_Ed * N1 - N_Ed * M1) / det

            # Check if intersection is valid:
            # - alpha > 0 (intersection is in the direction of the applied load)
            # - 0 <= s <= 1 (intersection is on this edge segment)
            if alpha > 1e-10 and 0 <= s <= 1:
                max_alpha = max(max_alpha, alpha)

        # If no intersection found, the point might be outside or there's a numerical issue
        if max_alpha < 1e-10:
            # Likely outside the boundary - use large utilization
            return (False, float('inf'))

        # Utilization ratio
        utilization = 1.0 / max_alpha
        is_safe = utilization <= 1.0

        return (is_safe, utilization)

    def find_balanced_point(
        self,
        max_concrete_strain: Optional[float] = None,
    ) -> Tuple[InteractionPoint, float]:
        """
        Find the balanced failure point on the M-N diagram.

        The balanced failure point occurs when the extreme compression fiber
        in concrete reaches its ultimate strain (ε_cu) at the same time that
        the tensile steel reaches its yield strain (ε_y).

        This method uses numerical optimization to find the neutral axis depth
        that satisfies the balanced failure condition.

        Args:
            max_concrete_strain: Maximum concrete strain (default: ε_cu2 from material)

        Returns:
            Tuple of (balanced_point, neutral_axis_depth_balanced)
            - balanced_point: InteractionPoint at balanced failure
            - neutral_axis_depth_balanced: Neutral axis depth at balanced failure (mm)

        Raises:
            RuntimeError: If balanced point cannot be found (optimization fails)
        """
        if max_concrete_strain is None:
            max_concrete_strain = self.concrete_model.get_ultimate_strain()

        # Get fiber arrays to find deepest steel location
        x, y, area, material_type, material_index = self.mesh.get_fiber_arrays()
        steel_mask = material_type == 'steel'

        if not np.any(steel_mask):
            raise ValueError("Cannot find balanced point - no steel reinforcement")

        # Find extreme tension steel (bottom of section)
        steel_y = y[steel_mask]
        steel_indices = material_index[steel_mask]
        extreme_tension_steel_y = np.min(steel_y)  # Lowest y (bottom)

        # Find which group the extreme tension steel belongs to
        extreme_steel_idx = np.argmin(steel_y)
        extreme_steel_group_idx = steel_indices[extreme_steel_idx]

        # Get steel yield strain for the extreme tension steel group
        steel_yield_strain = self.steel_models[extreme_steel_group_idx].epsilon_y

        # Distance from top of section to extreme tension steel
        y_steel_from_top = self.section_top - extreme_tension_steel_y

        # For balanced failure:
        # ε_concrete (at top) = ε_cu
        # ε_steel (at bottom) = ε_y
        # Using similar triangles:
        # ε_cu / x_bal = ε_y / (y_steel - x_bal)
        # Solving: x_bal = ε_cu * y_steel / (ε_cu + ε_y)

        # Initial estimate for balanced neutral axis depth
        x_bal_estimate = (
            max_concrete_strain * y_steel_from_top
            / (max_concrete_strain + steel_yield_strain)
        )

        # Define objective function: difference between target and actual max steel strain
        def objective(na_depth: float) -> float:
            """
            Objective function for finding balanced point.

            Returns the difference between the maximum tensile steel strain
            and the yield strain. Zero when balanced.
            """
            # Calculate strains at extreme tension steel
            y_from_top = y_steel_from_top

            if na_depth > 0:
                # Strain at extreme tension steel (negative = tension)
                strain_steel = max_concrete_strain * (na_depth - y_from_top) / na_depth
            else:
                # NA above section
                strain_steel = -max_concrete_strain * y_from_top / abs(na_depth)

            # Balanced when |ε_steel| = ε_yield (for tension, strain is negative)
            return abs(strain_steel) - steel_yield_strain

        # Solve using root finding
        # Search range: from 0.1*height to 2*height
        try:
            solution = root_scalar(
                objective,
                bracket=[self.section_height * 0.1, self.section_height * 2.0],
                method='brentq',
                xtol=0.01,  # 0.01 mm tolerance
            )

            if not solution.converged:
                raise RuntimeError("Balanced point optimization did not converge")

            na_balanced = solution.root

        except ValueError as e:
            # If bracket doesn't contain a root, use the estimate
            # This can happen for sections with very high or very low reinforcement
            na_balanced = x_bal_estimate

        # Calculate the balanced point
        balanced_point = self.calculate_point(
            neutral_axis_depth=na_balanced,
            max_concrete_strain=max_concrete_strain,
        )

        return (balanced_point, na_balanced)

    def get_diagram_arrays(
        self,
        n_points: int = 100,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Get M-N diagram as numpy arrays for plotting.

        Args:
            n_points: Number of points to generate

        Returns:
            Tuple of (N_array, M_array) in kN and kN·m
        """
        points = self.generate_diagram(n_points=n_points)

        N = np.array([p.N for p in points])
        M = np.array([p.M for p in points])

        return (N, M)

    def export_to_json(
        self,
        file_path: str | Path,
        n_points: int = 50,
        include_metadata: bool = True,
        indent: int = 2,
    ) -> None:
        """
        Export M-N diagram to JSON file.

        Args:
            file_path: Output file path
            n_points: Number of points to generate
            include_metadata: Include section and material metadata
            indent: JSON indentation (None for compact)
        """
        points = self.generate_diagram(n_points=n_points)

        data: Dict[str, Any] = {
            "diagram_points": [p.to_dict() for p in points],
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
            }

        file_path = Path(file_path)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent)

    def export_to_csv(
        self,
        file_path: str | Path,
        n_points: int = 50,
        include_strains: bool = True,
    ) -> None:
        """
        Export M-N diagram to CSV file.

        Args:
            file_path: Output file path
            n_points: Number of points to generate
            include_strains: Include strain data columns
        """
        points = self.generate_diagram(n_points=n_points)

        file_path = Path(file_path)
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            if include_strains:
                fieldnames = [
                    'N_kN',
                    'M_kNm',
                    'neutral_axis_depth_mm',
                    'max_concrete_strain',
                    'max_steel_strain',
                ]
            else:
                fieldnames = ['N_kN', 'M_kNm']

            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for point in points:
                row_data = point.to_dict()
                if not include_strains:
                    row_data = {
                        'N_kN': row_data['N_kN'],
                        'M_kNm': row_data['M_kNm'],
                    }
                writer.writerow(row_data)

    def to_dict(
        self,
        n_points: int = 50,
        include_metadata: bool = True,
    ) -> Dict[str, Any]:
        """
        Export M-N diagram to dictionary for programmatic access.

        Args:
            n_points: Number of points to generate
            include_metadata: Include section and material metadata

        Returns:
            Dictionary with diagram data
        """
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
            }

        return data

    def __repr__(self) -> str:
        return (
            f"MNInteractionDiagram("
            f"section={self.section.section_name}, "
            f"concrete={self.concrete.grade}, "
            f"fibers={self.mesh.total_fibers})"
        )


def create_interaction_diagram(
    section: RCSection,
    concrete: ConcreteMaterial,
    **kwargs,
) -> MNInteractionDiagram:
    """
    Factory function to create M-N interaction diagram.

    Args:
        section: RC section with reinforcement
        concrete: Concrete material
        **kwargs: Additional arguments passed to MNInteractionDiagram

    Returns:
        MNInteractionDiagram instance

    Example:
        >>> from materials.reinforced_concrete.geometry import create_rectangular_section
        >>> from materials.reinforced_concrete.materials import ConcreteMaterial
        >>>
        >>> section = create_rectangular_section(300, 500)
        >>> # ... add reinforcement ...
        >>> concrete = ConcreteMaterial(grade="C30/37")
        >>>
        >>> diagram = create_interaction_diagram(section, concrete)
        >>> N, M = diagram.get_diagram_arrays(n_points=100)
        >>>
        >>> # Check capacity
        >>> is_safe, util = diagram.get_utilization_vector(N_Ed=500, M_Ed=150)
    """
    return MNInteractionDiagram(section=section, concrete=concrete, **kwargs)
