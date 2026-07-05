"""
Bending (flexure) check using M-N interaction diagrams.

This is a FIRST PRINCIPLES check based on strain compatibility and force equilibrium.
Uses the fibre-based M-N interaction diagram infrastructure.
"""

from functools import cached_property
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
from pydantic import Field, PrivateAttr
import numpy as np

from materials.reinforced_concrete.code_checks.base_check import (
    BaseCodeCheck,
    CheckResult,
)
from materials.reinforced_concrete.constitutive import ConcreteModelType, SteelModelType
from materials.reinforced_concrete.geometry import RCSection
from materials.reinforced_concrete.materials import ConcreteMaterial, ShearRebar
from materials.reinforced_concrete.analysis import create_interaction_diagram
from materials.reinforced_concrete.analysis.interaction_diagram import MNInteractionDiagram


class BendingCheck(BaseCodeCheck):
    """
    EC2 bending check using M-N interaction diagram (§6.1).

    This check uses FIRST PRINCIPLES:
    1. Strain compatibility (plane sections remain plane)
    2. Force equilibrium (ΣF = N, ΣM = M)
    3. Constitutive models (stress-strain with codified factors γ_c, γ_s)

    The M-N diagram already handles:
    - fibre-based integration
    - Design strengths (f_cd, f_yd)
    - Ultimate limit state strains
    - Stress-strain models per EC2 Figs 3.2-3.8

    Attributes:
        section: RC section geometry with reinforcement
        concrete: Concrete material (with γ_c factor)
        concrete_model_type: EC2 constitutive model to use
        steel_model_type: Steel post-yield behaviour
        n_fibres_width: Mesh resolution (width)
        n_fibres_height: Mesh resolution (height)

    Example:
        >>> from materials.reinforced_concrete.geometry import create_rectangular_section
        >>> from materials.reinforced_concrete.materials import ConcreteMaterial, Rebar
        >>>
        >>> # Create section
        >>> section = create_rectangular_section(width=300, height=500)
        >>> # ... add reinforcement ...
        >>>
        >>> # Create check
        >>> concrete = ConcreteMaterial(grade="C30/37")
        >>> check = BendingCheck(section=section, concrete=concrete)
        >>>
        >>> # Perform check for applied loads (all parameters must be keyword arguments)
        >>> result = check.perform_check(M_Ed=150, N_Ed=500)  # kN·m, kN
        >>> print(result)
        >>> # Bending check (EC2 §6.1): PASS (utilization: 68.5%)
    """

    section: RCSection = Field(
        ...,
        description="RC section with reinforcement",
    )

    concrete: ConcreteMaterial = Field(
        ...,
        description="Concrete material (γ_c applied to get f_cd)",
    )

    concrete_model_type: ConcreteModelType = Field(
        default=ConcreteModelType.PARABOLA_RECTANGLE,
        description="EC2 concrete stress-strain model (Fig 3.3, 3.4, 3.2)",
    )

    steel_model_type: SteelModelType = Field(
        default=SteelModelType.INCLINED,
        description="Steel post-yield behaviour (Fig 3.8)",
    )

    n_fibres_width: int = Field(
        default=20,
        description="Number of concrete fibres across width",
        ge=10,
        le=500,
    )

    n_fibres_height: int = Field(
        default=30,
        description="Number of concrete fibres across height",
        ge=10,
        le=500,
    )


    # ===========================
    # Limit state factors
    # ===========================

    use_accidental: bool = Field(
        default=False,
        description="Use accidental limit state partial factors (gamma_c_accidental, gamma_s_accidental)",
    )


    # ===========================
    # Internal state (private)
    # ===========================

    _diagram: MNInteractionDiagram = PrivateAttr()
    _diagram_no_comp_steel: Optional[MNInteractionDiagram] = PrivateAttr(default=None)

    def model_post_init(self, __context):
        """
        Create M-N interaction diagram on initialization for reuse across multiple checks.

        This significantly improves performance when checking multiple load cases against
        the same section, as the diagram (mesh + material models) only needs to be created once.
        """
        super().model_post_init(__context)

        # Create and cache the diagram for reuse (with compression steel)
        self._diagram = create_interaction_diagram(
            section=self.section,
            concrete=self.concrete,
            concrete_model_type=self.concrete_model_type,
            steel_model_type=self.steel_model_type,
            n_fibres_width=self.n_fibres_width,
            n_fibres_height=self.n_fibres_height,
            use_accidental=self.use_accidental,
            ignore_compression_steel=False,
        )
        # Diagram without compression steel is created lazily on first use
        self._diagram_no_comp_steel = None

        # cached properties to save time later
        self._A_transformed = self.section.get_transformed_area(self.concrete.E_cm)  # mm²
        self._A_gross = self.section.get_area()  # mm²

    def _get_diagram(self, ignore_compression_steel: bool = False) -> MNInteractionDiagram:
        """Get the appropriate cached diagram based on ignore_compression_steel flag."""
        if not ignore_compression_steel:
            return self._diagram

        # Lazily create the diagram without compression steel
        if self._diagram_no_comp_steel is None:
            self._diagram_no_comp_steel = create_interaction_diagram(
                section=self.section,
                concrete=self.concrete,
                concrete_model_type=self.concrete_model_type,
                steel_model_type=self.steel_model_type,
                n_fibres_width=self.n_fibres_width,
                n_fibres_height=self.n_fibres_height,
                use_accidental=self.use_accidental,
                ignore_compression_steel=True,
            )
        return self._diagram_no_comp_steel


    # ===============================================
    # Properties (immutable - don't depend on loads)
    # ===============================================

    @cached_property
    def f_cd_design(self) -> float:
        """Design concrete strength (accidental or persistent) in MPa."""
        return self.concrete.f_cd_accidental if self.use_accidental else self.concrete.f_cd


    def perform_check(
        self,
        *,
        M_Ed: float,
        N_Ed: float = 0.0,
        V_Ed: Optional[float] = None,
        M_cap: Optional[float] = None,
        shear_reinforcement: Optional[ShearRebar] = None,
        warning_threshold: float = 0.95,
        ignore_compression_steel: bool = False,
        iterate_z: bool = False,
        **kwargs,
    ) -> CheckResult:
        """
        Check section capacity against applied bending moment and axial force.

        Uses M–N interaction diagram and ray intersection:
        - Finds boundary point (N_Rd, M_Rd) along the load vector (M_Ed, N_Ed)
        - Utilization = 1 / t_cap where (M_Rd, N_Rd) = t_cap * (M_Ed, N_Ed)

        Tension Shift Rule (EC2 §9.2.1.3):
        When M_cap is provided, automatically applies tension shift rule to account for
        additional tensile force in longitudinal reinforcement due to shear (truss analogy):

        With shear reinforcement:
        - Calculates optimal cot(θ) from V_Ed using V_Rd,max formula
        - a_l = 0.5 · z · cot(θ)  [shift distance, vertical links]
        - M_add = V_Ed · a_l

        Without shear reinforcement:
        - a_l = d  [EC2 §9.2.1.3(2)]
        - M_add = V_Ed · d

        Design moment: M_design = min(M_cap, M_Ed + M_add)

        Args:
            M_Ed: Design bending moment (kN·m)
            N_Ed: Design axial force (kN, positive = compression)
            V_Ed: Design shear force (kN) - required if M_cap is provided
            M_cap: Moment capacity cap (kN·m) from envelope analysis.
                   If provided, enables tension shift rule. Limits M_design = min(M_cap, M_Ed + M_add)
            shear_reinforcement: Optional ShearReinforcement object.
                                If provided, calculates cot(θ) from V_Ed.
                                If not provided, uses a_l = d (no shear reinforcement)
            warning_threshold: Utilization threshold for warnings (default 0.95)
            ignore_compression_steel: If True, steel in compression contributes zero force.
                                     This is a conservative option used by some commercial software.
            iterate_z: If True, iteratively recalculate z based on M_design until convergence
                      (0.5% tolerance, max 5 iterations). If diverges, uses original z.
                      Only relevant when tension shift is applied with shear reinforcement.

        Returns:
            CheckResult with pass/fail status and utilization
        """
        # Validate tension shift inputs
        # If M_cap provided, tension shift is enabled
        apply_tension_shift = M_cap is not None
        if apply_tension_shift and V_Ed is None:
            raise ValueError("V_Ed must be provided when M_cap is provided (tension shift enabled)")

        return self._check_single_case(
            M_Ed=M_Ed,
            N_Ed=N_Ed,
            V_Ed=V_Ed,
            M_cap=M_cap,
            shear_reinforcement=shear_reinforcement,
            warning_threshold=warning_threshold,
            ignore_compression_steel=ignore_compression_steel,
            iterate_z=iterate_z,
        )


    def _check_single_case(
        self,
        *,
        M_Ed: float,
        N_Ed: float,
        V_Ed: Optional[float],
        M_cap: Optional[float],
        shear_reinforcement: Optional[ShearRebar],
        warning_threshold: float,
        ignore_compression_steel: bool = False,
        iterate_z: bool = False,
    ) -> CheckResult:
        M_Ed_original = float(M_Ed)

        # --- Step 1: tension shift (only if M_cap is provided) ---
        if M_cap is not None:
            if V_Ed is None:
                raise ValueError("V_Ed must be provided when M_cap is provided (tension shift enabled)")

            # Use the diagram's apply_tension_shift which handles all the policy decisions
            shift_result = self._diagram.apply_tension_shift(
                M_Ed=M_Ed_original,
                V_Ed=float(V_Ed),
                N_Ed=float(N_Ed),
                M_cap=float(M_cap),
                shear_reinforcement=shear_reinforcement,
                iterate_z=iterate_z,
            )
            M_design = shift_result.M_design
            shift_details = {
                "tension_shift_applied": True,
                "M_add": float(shift_result.M_add),
                "V_Ed": float(V_Ed),
                "M_cap": float(M_cap),
                "cot_theta": float(shift_result.cot_theta) if shift_result.cot_theta is not None else None,
                "shift_distance_a_l_mm": float(shift_result.shift_distance_a_l_mm),
                "z_lever_arm_mm": float(shift_result.z_mm),
                "shear_reinforcement_provided": shear_reinforcement is not None,
            }
        else:
            # No tension shift - use original moment
            M_design = M_Ed_original
            shift_details = {
                "tension_shift_applied": False,
                "M_add": None,
                "V_Ed": None,
                "M_cap": None,
                "cot_theta": None,
                "shift_distance_a_l_mm": None,
                "z_lever_arm_mm": None,
                "shear_reinforcement_provided": False,
            }

        # --- Step 2: capacity check against diagram ---
        diagram = self._get_diagram(ignore_compression_steel)
        capacity = diagram.get_capacity_vector(N_Ed=N_Ed, M_Ed=M_design, return_details=False)
        N_Rd, M_Rd, utilization = capacity.N_Rd, capacity.M_Rd, capacity.utilization

        demand_components = {"N": float(N_Ed), "M": float(M_Ed_original)}
        units_components = {"N": "kN", "M": "kN·m"}

        # Build base details dict
        base_details = {
            "N_Ed": float(N_Ed),
            "M_Ed_original": float(M_Ed_original),
            "M_Ed_design": float(M_design),
            **shift_details,
            "concrete_model": self.concrete_model_type,
            "steel_model": self.steel_model_type,
            "section_name": self.section.section_name or "unnamed",
            "concrete_grade": self.concrete.grade,
            "reinforcement_ratio": self.section.reinforcement_ratio,
            "ignore_compression_steel": ignore_compression_steel,
        }

        # --- Step 3: handle genuinely invalid outcomes ---
        if (
            N_Rd is None
            or M_Rd is None
            or utilization is None
            or utilization == float("inf")
            or utilization != utilization  # NaN
        ):
            message = "Load point outside interaction diagram domain (no capacity found)"
            details = {
                **base_details,
                "N_Rd": N_Rd,
                "M_Rd": M_Rd,
                "utilization": float(utilization) if utilization is not None else None,
            }
            return self._create_result(
                check_name="Bending check (EC2 §6.1)",
                code_reference="EC2 §6.1",
                warning_threshold=warning_threshold,
                utilization=float("inf"),
                demand_components=demand_components,
                capacity_components=None,
                units_components=units_components,
                message=message,
                details=details,
            )

        utilization_f = float(utilization)
        capacity_components = {"N": float(N_Rd), "M": float(M_Rd)}

        if utilization_f <= 1.0:
            message = (
                "High utilization - consider increasing section or reinforcement"
                if utilization_f >= warning_threshold
                else "Section capacity adequate"
            )
        else:
            deficit = (utilization_f - 1.0) * 100.0
            message = f"Section capacity exceeded - increase section or reinforcement by ~{deficit:.0f}%"

        details = {
            **base_details,
            "N_Rd": float(N_Rd),
            "M_Rd": float(M_Rd),
            "utilization": utilization_f,
        }

        return self._create_result(
            check_name="Bending check (EC2 §6.1)",
            code_reference="EC2 §6.1",
            warning_threshold=warning_threshold,
            utilization=utilization_f,
            demand_components=demand_components,
            capacity_components=capacity_components,
            units_components=units_components,
            message=message,
            details=details,
        )

    def get_moment_capacity(self, N_Ed: float = 0.0) -> tuple[Optional[float], Optional[float]]:
        """
        Get moment capacity at specified axial force.

        Args:
            N_Ed: Design axial force in kN

        Returns:
            Tuple of (M_Rd_positive, M_Rd_negative) in kN·m
            Returns (None, None) if N_Ed is outside the interaction diagram bounds.
        """
        N_cap, M_Rd_pos, M_Rd_neg = self._diagram.get_capacity_fixed_n(N_Ed=N_Ed)

        if N_cap is not None:
            if N_cap >= 0:  # N_cap is positive
                if N_Ed > N_cap:  # N_Ed outside upper bound
                    M_Rd_pos = None
                    M_Rd_neg = None
            else:  # N_cap is negative
                if N_Ed < N_cap:  # N_Ed outside lower bound
                    M_Rd_pos = None
                    M_Rd_neg = None
        return (M_Rd_pos, M_Rd_neg)


    def generate_interaction_diagram_arrays(self, n_points: int = 120) -> tuple["np.ndarray", "np.ndarray"]:
        """
        Generate complete M-N interaction diagram for visualization.

        Args:
            n_points: Number of points on the curve

        Returns:
            Tuple of (N_array, M_array) for plotting
        """
        return self._diagram.get_diagram_arrays(n_points=n_points)

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

        Example:
            >>> check = BendingCheck(section=section, concrete=concrete)
            >>> # Plot diagram with load cases
            >>> check.plot_mn(
            ...     load_points=[
            ...         {"N_Ed": 500, "M_Ed": 150, "name": "LC1"},
            ...         {"N_Ed": 800, "M_Ed": 100, "name": "LC2"},
            ...     ],
            ...     show_vectors=True,
            ... )
        """
        return self._diagram.plot_mn(
            load_points=load_points,
            show_vectors=show_vectors,
            show_metadata=show_metadata,
            n_points=n_points,
            save_path=save_path,
            show=show,
            title=title,
        )

    def plot_stress_strain(
        self,
        M_Ed: float,
        N_Ed: float,
        *,
        show: bool = True,
        title: Optional[str] = None,
        width: int = 1200,
        height: int = 600,
        section_render: Literal["points", "filled"] = "points",
    ) -> Any:
        """
        Visualize stress and strain distribution for a given load case.

        Creates an interactive plot showing:
        - Section geometry with reinforcement
        - Strain profile across the section depth
        - Stress distribution in concrete and steel

        Args:
            M_Ed: Design bending moment (kN·m)
            N_Ed: Design axial force (kN, positive = compression)
            show: If True, display plot (fig.show())
            title: Custom plot title (optional)
            width: Plot width in pixels
            height: Plot height in pixels
            section_render: How to render section - "points" for fibre centroids,
                           "filled" for filled polygon

        Returns:
            Plotly Figure object

        Example:
            >>> check = BendingCheck(section=section, concrete=concrete)
            >>> # Visualize stress/strain for a specific load case
            >>> check.plot_stress_strain(M_Ed=150, N_Ed=500)
        """
        return self._diagram.plot_stress_strain(
            M_Ed=M_Ed,
            N_Ed=N_Ed,
            show=show,
            title=title,
            width=width,
            height=height,
            section_render=section_render,
        )
