"""
Adapter that wraps BiaxialMNInteractionSurface to provide the
MNInteractionDiagram interface for the Mz=0 slice.

Used when a section is asymmetric about the minor axis and the neutral axis
must be free to rotate for correct equilibrium.
"""

from __future__ import annotations

import warnings
from typing import Optional, Tuple, TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from materials.utils.helpers import as_float
from materials.core.units import ForceUnit, MomentUnit, to_kn, to_knm
from materials.reinforced_concrete.analysis.interaction_diagram import (
    InteractionPoint,
    CapacityResult,
    _ray_segment_intersection_alpha,
)
from materials.reinforced_concrete.analysis.strain_state import StrainState

if TYPE_CHECKING:
    from materials.reinforced_concrete.analysis.biaxial_interaction import (
        BiaxialMNInteractionSurface,
    )
    from materials.reinforced_concrete.code_checks.ec2_2004.shear_utils import TensionShiftResult
    from materials.reinforced_concrete.materials.rebar import ShearRebar


class FreeNADiagramAdapter:
    """
    Adapter that wraps :class:`BiaxialMNInteractionSurface` and exposes the
    same interface as :class:`MNInteractionDiagram` for the ``Mz = 0`` slice.

    This enables transparent routing: downstream code checks (bending, cracking,
    stress limits) call the same methods they would on the 2D diagram, but
    equilibrium is solved with a free neutral axis angle.

    The adapter uses the biaxial surface's convex hull to extract a 2D M-N
    envelope at Mz=0, and delegates strain solving to the biaxial
    ``calculate_point_pivot`` with an NA angle search.
    """

    def __init__(
        self,
        biaxial_surface: "BiaxialMNInteractionSurface",
        n_angles: int = 72,
        n_axial_levels: int = 30,
    ):
        self._biaxial = biaxial_surface

        # Forward attributes that downstream checks access directly
        self.section = biaxial_surface.section
        self.concrete = biaxial_surface.concrete
        self.concrete_model = biaxial_surface.concrete_model
        self.steel_models = biaxial_surface.steel_models
        self.mesh = biaxial_surface.mesh
        self.tension_stiffening = biaxial_surface.tension_stiffening
        self.confined_concrete = biaxial_surface.confined_concrete
        self.ignore_compression_steel = biaxial_surface.ignore_compression_steel
        self.include_tension = biaxial_surface.include_tension
        self.crack_to_neutral_axis_on_first_tension_failure = (
            biaxial_surface.crack_to_neutral_axis_on_first_tension_failure
        )

        # Geometry refs matching MNInteractionDiagram convention
        _, min_y, _, max_y = self.section.get_bounding_box()
        self.section_top = max_y
        self.section_bottom = min_y
        self.section_height = max_y - min_y
        self._section_cx, self._section_cy = self.section.get_centroid()
        self.section_centroid_x, self.section_centroid_y = self._section_cx, self._section_cy

        # Forward fibre arrays (accessed directly by CrackingCheck)
        self._fibre_x = biaxial_surface._fibre_x
        self._fibre_y = biaxial_surface._fibre_y
        self._fibre_area = biaxial_surface._fibre_area
        self._fibre_mat = biaxial_surface._fibre_mat
        self._fibre_mi = biaxial_surface._fibre_mi
        self._fibre_i = biaxial_surface._fibre_i
        self._fibre_j = biaxial_surface._fibre_j

        # Surface generation params
        self._n_angles = n_angles
        self._n_axial_levels = n_axial_levels

        # Caches
        self._slice_cache: dict[int, list[tuple[float, float]]] = {}
        self._diagram_points_cache: dict[int, tuple[InteractionPoint, ...]] = {}
        self._strain_cache: dict[tuple, tuple] = {}

    # ----------------------------
    # Strain field computation (1D)
    # ----------------------------

    def _strain_field_from_end_strains(
        self,
        eps_top: float,
        eps_bottom: float,
    ) -> npt.NDArray[np.float64]:
        """
        Plane-sections strain field from end strains (horizontal NA projection).

        Identical to MNInteractionDiagram._strain_field_from_end_strains.
        """
        y = self._fibre_y
        y_bot = float(self.section_bottom)
        h = float(self.section_height)
        t = (y - y_bot) / h
        return eps_bottom + (eps_top - eps_bottom) * t

    # ----------------------------
    # Section forces
    # ----------------------------

    def calculate_section_forces(
        self,
        stresses: npt.NDArray[np.float64],
        *,
        use_section_centroid: bool = True,
    ) -> Tuple[float, float]:
        """Calculate (N, M) from fibre stresses — identical to 2D diagram."""
        y = self._fibre_y
        area = self._fibre_area

        N = to_kn(np.sum(stresses * area), ForceUnit.N)

        if use_section_centroid:
            cy = self._section_cy
        else:
            cy = float(np.sum(y * area) / np.sum(area))

        y_offset = y - cy
        M = to_knm(np.sum(stresses * area * y_offset), MomentUnit.NMM)

        return (as_float(N), as_float(M))

    # ----------------------------
    # 2D M-N slice
    # ----------------------------

    def _get_slice(self, n_points: int = 120) -> list[tuple[float, float]]:
        """Get Mz=0 slice as (My, N) polygon, cached."""
        if n_points in self._slice_cache:
            return self._slice_cache[n_points]

        slice_pts = self._biaxial.get_mn_slice(
            mz_target=0.0,
            n_angles=self._n_angles,
            n_axial_levels=self._n_axial_levels,
        )
        self._slice_cache[n_points] = slice_pts
        return slice_pts

    def generate_diagram_points(
        self,
        n_points: int = 120,
        n_dense: int = 800,
    ) -> tuple[InteractionPoint, ...]:
        """Generate interaction points from the Mz=0 slice."""
        if n_points in self._diagram_points_cache:
            return self._diagram_points_cache[n_points]

        slice_pts = self._get_slice(n_points)
        if not slice_pts:
            return ()

        points = []
        for my, n in slice_pts:
            points.append(InteractionPoint(
                N=n,
                M=my,
                neutral_axis_depth=0.0,  # Approximate metadata
                compression_from_bottom=(n > 0),
                max_concrete_strain=0.0,
                max_steel_strain=0.0,
            ))

        result = tuple(points)
        self._diagram_points_cache[n_points] = result
        return result

    def get_diagram_arrays(
        self,
        n_points: int = 120,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Return (N_array, M_array) for the Mz=0 slice."""
        pts = self.generate_diagram_points(n_points=n_points)
        N = np.array([p.N for p in pts], dtype=float)
        M = np.array([p.M for p in pts], dtype=float)
        return (N, M)

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
        Ray intersection on the Mz=0 slice for capacity check.

        Matches MNInteractionDiagram.get_capacity_vector interface.
        """
        diagram_points = self.generate_diagram_points(n_points=n_points)
        pts = [(p.M, p.N) for p in diagram_points]
        if len(pts) < 3:
            return CapacityResult(N_Rd=None, M_Rd=None, is_safe=False, utilization=float("inf"))

        if abs(M_Ed) < 1e-18 and abs(N_Ed) < 1e-18:
            return CapacityResult(N_Rd=0.0, M_Rd=0.0, is_safe=True, utilization=0.0)

        ray_dir = (float(M_Ed), float(N_Ed))

        if pts[0] != pts[-1]:
            pts = pts + [pts[0]]

        intersections = []
        for p1, p2 in zip(pts[:-1], pts[1:]):
            if p1 == p2:
                continue
            t = _ray_segment_intersection_alpha(ray_dir, p1, p2, tol=1e-12)
            if t is not None:
                intersections.append(t)

        ts = [t for t in intersections if t > 1e-12]
        if not ts:
            return CapacityResult(N_Rd=None, M_Rd=None, is_safe=False, utilization=float("inf"))

        t_cap = min(ts)
        M_Rd = t_cap * float(M_Ed)
        N_Rd = t_cap * float(N_Ed)
        utilization = 1.0 / t_cap
        is_safe = utilization <= 1.0

        if not return_details:
            return CapacityResult(
                N_Rd=float(N_Rd), M_Rd=float(M_Rd),
                is_safe=bool(is_safe), utilization=float(utilization),
            )

        # Compute details via strain solve
        try:
            eps_top, eps_bottom = self.find_strains_for_MN(M_Rd, N_Rd)
            strains = self._strain_field_from_end_strains(eps_top, eps_bottom)
            conc_mask = self._fibre_mat == "concrete"
            steel_mask = self._fibre_mat == "steel"
            max_conc = float(np.max(np.abs(strains[conc_mask]))) if np.any(conc_mask) else 0.0
            max_steel = float(np.max(np.abs(strains[steel_mask]))) if np.any(steel_mask) else 0.0

            details = {
                'eps_top': float(eps_top),
                'eps_bottom': float(eps_bottom),
                'neutral_axis_depth': None,
                'compression_from_bottom': None,
                'max_concrete_strain': max_conc,
                'max_steel_strain': max_steel,
            }
            return CapacityResult(
                N_Rd=float(N_Rd), M_Rd=float(M_Rd),
                is_safe=bool(is_safe), utilization=float(utilization),
                details=details,
            )
        except Exception:
            return CapacityResult(
                N_Rd=float(N_Rd), M_Rd=float(M_Rd),
                is_safe=bool(is_safe), utilization=float(utilization),
            )

    def get_utilization_vector(
        self,
        N_Ed: float,
        M_Ed: float,
        n_points: int = 120,
    ) -> Tuple[bool, float]:
        """Convenience wrapper returning (is_safe, utilization)."""
        cap = self.get_capacity_vector(N_Ed=N_Ed, M_Ed=M_Ed, n_points=n_points)
        return (bool(cap.is_safe), float(cap.utilization))

    def get_capacity_fixed_n(
        self,
        N_Ed: float,
        *,
        n_points: int = 160,
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Horizontal-line capacity at fixed axial force on Mz=0 slice."""
        diagram_points = self.generate_diagram_points(n_points=n_points)
        if len(diagram_points) < 4:
            return (None, None, None)

        pts = [(float(p.M), float(p.N)) for p in diagram_points]
        if pts[0] != pts[-1]:
            pts = pts + [pts[0]]

        N_vals = [n for _, n in pts]
        N_min, N_max = min(N_vals), max(N_vals)
        N_cap = float(min(max(N_Ed, N_min), N_max))

        # Find horizontal intersections
        from materials.reinforced_concrete.analysis.interaction_diagram import MNInteractionDiagram
        Ms = MNInteractionDiagram._intersections_with_horizontal(pts, N0=N_cap)

        if not Ms:
            return (None, None, None)

        return (N_cap, float(max(Ms)), float(min(Ms)))

    def get_capacity_biaxial(
        self,
        N_Ed: float,
        My_Ed: float,
        Mz_Ed: float,
        n_angles: int = 72,
        n_axial_levels: int = 30,
    ) -> CapacityResult:
        """
        3D ray intersection on the biaxial M-M-N surface for capacity check.

        Computes utilisation as the ratio of the demand magnitude to the
        capacity boundary along the (My, Mz, N) ray from the origin.

        Args:
            N_Ed: Design axial force (kN).
            My_Ed: Design major axis moment (kN·m).
            Mz_Ed: Design minor axis moment (kN·m).

        Returns:
            CapacityResult with utilisation on the biaxial surface.
        """
        if abs(My_Ed) < 1e-18 and abs(Mz_Ed) < 1e-18 and abs(N_Ed) < 1e-18:
            return CapacityResult(N_Rd=0.0, M_Rd=0.0, is_safe=True, utilization=0.0)

        try:
            from scipy.spatial import ConvexHull
        except ImportError:
            # Fallback: use Mz=0 slice capacity (conservative)
            return self.get_capacity_vector(N_Ed=N_Ed, M_Ed=My_Ed)

        # Generate surface points
        surface_data = self._biaxial.generate_surface_pivot(
            n_angles=n_angles, n_axial_levels=n_axial_levels,
        )
        if not surface_data:
            return CapacityResult(N_Rd=None, M_Rd=None, is_safe=False, utilization=float("inf"))

        my_arr = np.array([p[0] for p in surface_data], dtype=float)
        mz_arr = np.array([p[1] for p in surface_data], dtype=float)
        n_arr = np.array([p[2] for p in surface_data], dtype=float)

        pts_3d = np.column_stack([my_arr, mz_arr, n_arr])
        if len(pts_3d) < 4:
            return CapacityResult(N_Rd=None, M_Rd=None, is_safe=False, utilization=float("inf"))

        try:
            hull = ConvexHull(pts_3d)
        except Exception:
            return CapacityResult(N_Rd=None, M_Rd=None, is_safe=False, utilization=float("inf"))

        # Ray from origin in direction (My_Ed, Mz_Ed, N_Ed)
        ray = np.array([float(My_Ed), float(Mz_Ed), float(N_Ed)], dtype=float)
        ray_mag = float(np.linalg.norm(ray))
        if ray_mag < 1e-18:
            return CapacityResult(N_Rd=0.0, M_Rd=0.0, is_safe=True, utilization=0.0)

        ray_dir = ray / ray_mag

        # Intersect ray with all hull facets
        t_min = float("inf")
        for eq in hull.equations:
            normal = eq[:3]
            offset = eq[3]
            denom = float(np.dot(normal, ray_dir))
            if abs(denom) < 1e-18:
                continue
            t = -offset / denom
            if t > 1e-12 and t < t_min:
                t_min = t

        if t_min == float("inf"):
            return CapacityResult(N_Rd=None, M_Rd=None, is_safe=False, utilization=float("inf"))

        utilization = ray_mag / t_min
        is_safe = utilization <= 1.0
        cap_point = ray_dir * t_min
        M_Rd = float(cap_point[0])
        N_Rd = float(cap_point[2])

        return CapacityResult(
            N_Rd=N_Rd, M_Rd=M_Rd,
            is_safe=bool(is_safe), utilization=float(utilization),
        )

    # ----------------------------
    # Analytical elastic solve (uncracked SLS fast path)
    # ----------------------------

    def _solve_uncracked_elastic(
        self,
        My: float,
        N: float,
        Mz: float,
    ) -> Optional[Tuple[float, float, float]]:
        """
        Direct closed-form solve for an uncracked linear-elastic section.

        For an elastic section the equilibrium equations in (ε₀, κ_y, κ_z) are
        a 3×3 linear system — no iteration required and no spurious local minima.
        This eliminates the biaxial solver convergence issue at low SLS loads
        where the Levenberg-Marquardt sweep can find a physically incorrect
        nearly-vertical NA.

        Uses the transformed section (accounting for steel modular ratio via
        ``section.get_transformed_second_moment_area``).  Moments are referenced
        about the GEOMETRIC centroid to match the convention used by
        ``calculate_point_pivot`` (``My = Σ σ·A·y_rel``).

        Returns:
            ``(eps_0, kappa_y, kappa_z)`` where ``eps_0`` is the strain at the
            geometric centroid, ``kappa_y`` is the curvature in the y-direction
            and ``kappa_z`` the curvature in the x-direction — ready for
            ``StrainState.from_plane(plane_a=kappa_z, plane_b=kappa_y, plane_c=eps_0)``.

            Returns ``None`` if:
            - the concrete model is not linear-elastic with tension enabled, or
            - any concrete fibre exceeds the cracking strain (section is cracked
              → caller should fall through to the iterative solver).
        """
        from materials.reinforced_concrete.constitutive.concrete_stress_strain import (
            ConcreteStressStrainLinearElastic,
        )

        if not (
            self._biaxial.include_tension
            and isinstance(self._biaxial.concrete_model, ConcreteStressStrainLinearElastic)
        ):
            return None

        E_c = self._biaxial.elastic_modulus
        if E_c is None or E_c <= 0.0:
            return None

        section = self._biaxial.section
        try:
            A_tr, cx_tr, cy_tr = section.get_transformed_centroid(E_c)
            I_xx_tr, I_yy_tr, I_xy_tr = section.get_transformed_second_moment_area(E_c)
        except Exception:
            return None

        cx_g = self.section_centroid_x
        cy_g = self.section_centroid_y
        dcx = cx_tr - cx_g
        dcy = cy_tr - cy_g

        # Parallel-axis shift: I values from transformed centroid → geometric centroid
        I_xx_g = I_xx_tr + A_tr * dcy ** 2
        I_yy_g = I_yy_tr + A_tr * dcx ** 2
        I_xy_g = I_xy_tr + A_tr * dcx * dcy

        # 3×3 stiffness matrix: [N, My, Mz] = E_c · K · [ε₀, κ_y, κ_z]
        K = np.array([
            [A_tr,           A_tr * dcy,  A_tr * dcx],
            [A_tr * dcy,     I_xx_g,      I_xy_g    ],
            [A_tr * dcx,     I_xy_g,      I_yy_g    ],
        ])
        # np.linalg.solve raises only for an EXACTLY singular K; a near-singular K
        # (degenerate fibre layout) yields garbage silently. Reject it on the
        # condition number and fall through to the iterative solver.
        cond = np.linalg.cond(K)
        if not np.isfinite(cond) or cond > 1e12:
            return None
        try:
            # Unit conversion: N [kN→N] *1e3, My/Mz [kN·m→N·mm] *1e6
            rhs = np.array([N * 1e3, My * 1e6, Mz * 1e6]) / E_c
            x = np.linalg.solve(K, rhs)
        except np.linalg.LinAlgError:
            return None
        if not np.all(np.isfinite(x)):
            return None

        eps_0, kappa_y, kappa_z = float(x[0]), float(x[1]), float(x[2])

        # Check cracking: any concrete fibre strain below cracking strain?
        conc_mask = self._fibre_mat == "concrete"
        if np.any(conc_mask):
            conc_eps = (
                eps_0
                + kappa_y * (self._fibre_y[conc_mask] - cy_g)
                + kappa_z * (self._fibre_x[conc_mask] - cx_g)
            )
            cracking_strain = float(self._biaxial.concrete_model.cracking_strain)
            if np.any(conc_eps < cracking_strain):
                return None  # section is cracked; fall through to iterative solver

        return eps_0, kappa_y, kappa_z

    # ----------------------------
    # Inverse solver
    # ----------------------------

    def _pivot_slope_and_y_na(
        self, na_depth: float, cos_a: float, sin_a: float
    ) -> Tuple[float, float]:
        """EC2 pivot-method strain slope and neutral-axis position for a solved
        (na_depth, angle).

        Single source of truth shared by ``find_strains_for_MN`` and
        ``find_strain_state_for_MN`` (the two used to carry verbatim copies of this
        block, which is exactly how their strain reconstructions could drift apart).
        Returns ``(slope, y_na)`` in the axis perpendicular to the neutral axis
        (centroidal coordinates). Zones: tension pivot (eps_ud) below the balanced
        depth, concrete pivot (eps_cu2) up to the full depth, then the over-
        compression branch (eps_c2 about z_p).
        """
        full_x_rel = self._fibre_x - self.section_centroid_x
        full_y_rel = self._fibre_y - self.section_centroid_y
        full_dist = full_y_rel * cos_a + full_x_rel * sin_a
        y_max_perp = float(np.max(full_dist))

        steel_mask = self._fibre_mat == 'steel'
        rebar_y_min = (
            float(np.min(full_dist[steel_mask])) if np.any(steel_mask) else float(np.min(full_dist))
        )
        d_eff = y_max_perp - rebar_y_min
        y_na = y_max_perp - na_depth

        eps_cu2 = self._biaxial.concrete.epsilon_cu2
        eps_c2 = self._biaxial.concrete.epsilon_c2
        eps_ud = self._biaxial._eps_tension_limit()
        x_bal = (eps_cu2 / (eps_cu2 + eps_ud)) * d_eff
        h_perp = y_max_perp - float(np.min(full_dist))
        z_p = (1.0 - eps_c2 / eps_cu2) * h_perp

        if na_depth <= x_bal:
            slope = -eps_ud / (rebar_y_min - y_na)
        elif na_depth <= h_perp:
            slope = eps_cu2 / na_depth
        else:
            slope = eps_c2 / (na_depth - z_p)
        return float(slope), float(y_na)

    def find_strains_for_MN(
        self,
        My_target: float,
        N_target: float,
        initial_guess: Optional[Tuple[float, float]] = None,
        tol: float = 1e-6,
        strict: bool = False,
        Mz_target: float = 0.0,
    ) -> Tuple[float, float]:
        """
        Find end strains that produce target (My, N, Mz) with free NA.

        Uses the biaxial solver to find an NA depth and angle that gives
        (N_target, My=My_target, Mz=Mz_target), then projects the resulting
        2D strain plane onto the vertical centroidal axis.

        Args:
            My_target: Target major axis moment (kN·m).
            N_target: Target axial force (kN).
            Mz_target: Target minor axis moment (kN·m), default 0.

        Returns:
            (eps_top, eps_bottom) projected strains.
        """
        cache_key = (My_target, Mz_target, N_target, strict)
        cached = self._strain_cache.get(cache_key)
        if cached is not None:
            return cached

        # --- Analytical fast path for uncracked linear-elastic SLS ---
        # For an uncracked elastic section the equilibrium is a 3×3 linear system;
        # this avoids the local-minima problem of the L-M sweep entirely.
        elastic_plane = self._solve_uncracked_elastic(My_target, N_target, Mz_target)
        if elastic_plane is not None:
            eps_0, kappa_y, _kappa_z = elastic_plane
            y_top_rel = float(self.section_top) - self.section_centroid_y
            y_bot_rel = float(self.section_bottom) - self.section_centroid_y
            eps_top = float(eps_0 + kappa_y * y_top_rel)
            eps_bottom = float(eps_0 + kappa_y * y_bot_rel)
            result_tuple = (eps_top, eps_bottom)
            self._strain_cache[cache_key] = result_tuple
            return result_tuple

        from scipy.optimize import least_squares
        import math as _math

        max_dim = max(self._biaxial.section_width, self._biaxial.section_height)

        def residual(params: npt.NDArray) -> npt.NDArray:
            """Residual: [N_err, My_err, Mz_err]."""
            phi, angle_deg = params[0], params[1]
            na_depth = max_dim * np.tan(phi)
            pt = self._biaxial.calculate_point_pivot(na_depth, angle_deg)
            return np.array([
                pt.N - N_target,
                pt.My - My_target,
                pt.Mz - Mz_target,
            ])

        # Try multiple initial guesses
        best_result = None
        best_cost = float('inf')

        # Include angle hint based on moment ratio when Mz != 0
        angle_inits = [0.0, 5.0, -5.0, 10.0, -10.0]
        if abs(Mz_target) > 1e-9:
            hint = _math.degrees(_math.atan2(Mz_target, My_target)) if abs(My_target) > 1e-9 else 90.0
            angle_inits = [hint, hint + 10, hint - 10] + angle_inits

        # phi_inits: include shallow SLS-region values (phi≈0.02–0.15 → na_depth≈10–80mm
        # for a 500mm section) in addition to the ULS-range deep values.
        for phi_init in [0.02, 0.05, 0.1, 0.15, 0.5, 0.8, 1.0, -0.3]:
            for angle_init in angle_inits:
                try:
                    result = least_squares(
                        residual,
                        x0=np.array([phi_init, angle_init]),
                        method='lm',
                        ftol=tol,
                        xtol=tol,
                        max_nfev=200,
                    )
                    cost = float(np.sum(result.fun ** 2))
                    if cost < best_cost:
                        best_cost = cost
                        best_result = result
                    if cost < tol ** 2:
                        break
                except Exception:
                    continue
            if best_cost < tol ** 2:
                break

        if best_result is None:
            raise ValueError(
                f"FreeNADiagramAdapter: Cannot find strain state for "
                f"My={My_target:.3f} kN.m, Mz={Mz_target:.3f} kN.m, N={N_target:.3f} kN"
            )

        phi_sol, angle_sol = best_result.x
        na_depth_sol = max_dim * np.tan(phi_sol)
        angle_rad = np.radians(angle_sol)

        # Reconstruct 2D strain plane from solved NA parameters
        # The strain plane coefficients come from the pivot method geometry
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)

        # Compute strains at top and bottom of the vertical centroidal axis
        # Using the same strain computation as calculate_point_pivot
        x_rel = np.array([0.0, 0.0])  # centroidal axis
        y_top_rel = float(self.section_top) - self.section_centroid_y
        y_bot_rel = float(self.section_bottom) - self.section_centroid_y
        y_rel = np.array([y_top_rel, y_bot_rel])

        dist_perp = y_rel * cos_a + x_rel * sin_a

        slope, y_na = self._pivot_slope_and_y_na(na_depth_sol, cos_a, sin_a)
        strains_at_centroidal = slope * (dist_perp - y_na)
        eps_top = float(strains_at_centroidal[0])
        eps_bottom = float(strains_at_centroidal[1])

        result_tuple = (eps_top, eps_bottom)
        self._strain_cache[cache_key] = result_tuple
        return result_tuple

    def find_strain_state_for_MN(
        self,
        My_target: float,
        N_target: float,
        initial_guess: Optional[Tuple[float, float]] = None,
        tol: float = 1e-6,
        strict: bool = False,
        Mz_target: float = 0.0,
    ) -> StrainState:
        """
        Full strain state solver returning biaxial plane coefficients.

        Args:
            My_target: Target major axis moment (kN·m).
            N_target: Target axial force (kN).
            Mz_target: Target minor axis moment (kN·m), default 0.

        Returns:
            StrainState with full plane coefficients (is_biaxial=True when section
            requires NA rotation for equilibrium, e.g. when I_xy ≠ 0).
        """
        import math as _math

        # --- Analytical fast path for uncracked linear-elastic SLS ---
        elastic_plane = self._solve_uncracked_elastic(My_target, N_target, Mz_target)
        if elastic_plane is not None:
            eps_0, kappa_y, kappa_z = elastic_plane
            y_top_rel = float(self.section_top) - self.section_centroid_y
            y_bot_rel = float(self.section_bottom) - self.section_centroid_y
            # NA angle: gradient direction (kappa_z, kappa_y); angle from horizontal
            if abs(kappa_y) > 1e-15 or abs(kappa_z) > 1e-15:
                na_angle = _math.degrees(_math.atan2(kappa_z, kappa_y))
            else:
                na_angle = 0.0
            return StrainState.from_plane(
                plane_a=kappa_z,
                plane_b=kappa_y,
                plane_c=eps_0,
                y_top=y_top_rel,
                y_bottom=y_bot_rel,
                na_angle_deg=na_angle,
            )

        from scipy.optimize import least_squares

        max_dim = max(self._biaxial.section_width, self._biaxial.section_height)

        def residual(params: npt.NDArray) -> npt.NDArray:
            phi, angle_deg = params[0], params[1]
            na_depth = max_dim * np.tan(phi)
            pt = self._biaxial.calculate_point_pivot(na_depth, angle_deg)
            return np.array([pt.N - N_target, pt.My - My_target, pt.Mz - Mz_target])

        best_result = None
        best_cost = float('inf')

        # Include angle hint based on moment ratio when Mz != 0
        angle_inits = [0.0, 5.0, -5.0]
        if abs(Mz_target) > 1e-9:
            hint = _math.degrees(_math.atan2(Mz_target, My_target)) if abs(My_target) > 1e-9 else 90.0
            angle_inits = [hint, hint + 10, hint - 10] + angle_inits

        # phi_inits: include shallow SLS-region values
        for phi_init in [0.02, 0.05, 0.1, 0.15, 0.5, 0.8, 1.0, -0.3]:
            for angle_init in angle_inits:
                try:
                    result = least_squares(
                        residual,
                        x0=np.array([phi_init, angle_init]),
                        method='lm',
                        ftol=tol,
                        xtol=tol,
                        max_nfev=200,
                    )
                    cost = float(np.sum(result.fun ** 2))
                    if cost < best_cost:
                        best_cost = cost
                        best_result = result
                    if cost < tol ** 2:
                        break
                except Exception:
                    continue
            if best_cost < tol ** 2:
                break

        if best_result is None:
            raise ValueError(
                f"FreeNADiagramAdapter: Cannot find strain state for "
                f"My={My_target:.3f} kN.m, Mz={Mz_target:.3f} kN.m, N={N_target:.3f} kN"
            )

        phi_sol, angle_sol = best_result.x
        na_depth_sol = max_dim * np.tan(phi_sol)
        angle_rad = np.radians(angle_sol)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)

        # Reconstruct strain plane coefficients (shared pivot logic)
        slope, y_na = self._pivot_slope_and_y_na(na_depth_sol, cos_a, sin_a)

        # strain(x, y) = slope * ((y_rel * cos_a + x_rel * sin_a) - y_na)
        #              = slope * cos_a * y_rel + slope * sin_a * x_rel - slope * y_na
        # So: plane_a = slope * sin_a, plane_b = slope * cos_a, plane_c = -slope * y_na
        plane_a = slope * sin_a
        plane_b = slope * cos_a
        plane_c = -slope * y_na

        y_top_rel = float(self.section_top) - self.section_centroid_y
        y_bot_rel = float(self.section_bottom) - self.section_centroid_y

        return StrainState.from_plane(
            plane_a=plane_a,
            plane_b=plane_b,
            plane_c=plane_c,
            y_top=y_top_rel,
            y_bottom=y_bot_rel,
            na_angle_deg=float(angle_sol),
        )

    # ----------------------------
    # Fibre forces (for CrackingCheck / StressLimitsCheck)
    # ----------------------------

    def get_fibre_forces_from_end_strains(
        self,
        eps_top: float,
        eps_bottom: float,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Compute fibre-level forces from projected end strains.

        Uses the same logic as MNInteractionDiagram.get_fibre_forces_from_end_strains.
        """
        y = self._fibre_y
        area = self._fibre_area
        material_type = self._fibre_mat
        material_index = self._fibre_mi

        strains = self._strain_field_from_end_strains(eps_top=eps_top, eps_bottom=eps_bottom)
        stresses = np.zeros_like(strains)

        conc_mask = material_type == "concrete"
        if np.any(conc_mask):
            stresses[conc_mask] = self._biaxial._concrete_stress_with_options(strains[conc_mask])

        steel_mask = material_type == "steel"
        if np.any(steel_mask):
            steel_strains = strains[steel_mask]
            steel_indices = material_index[steel_mask]
            steel_stresses = np.zeros_like(steel_strains)
            for gi, sm in enumerate(self.steel_models):
                m = steel_indices == gi
                if np.any(m):
                    steel_stresses[m] = sm.get_stress_array(steel_strains[m])
            if self.ignore_compression_steel:
                steel_stresses[steel_strains > 0] = 0.0
            stresses[steel_mask] = steel_stresses

        forces = stresses * area
        return (forces, y, area)


    def get_fibre_forces_from_strain_state(
        self,
        strain_state: "StrainState",
    ) -> Tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ]:
        """
        Compute fibre-level forces from a :class:`StrainState`.

        When 1D, delegates to :meth:`get_fibre_forces_from_end_strains`.
        When biaxial, evaluates the full 2D strain plane across all fibres.

        Returns:
            Tuple of (forces, x_coords, y_coords, areas).
        """
        if not strain_state.is_biaxial:
            forces, y, area = self.get_fibre_forces_from_end_strains(
                strain_state.eps_top, strain_state.eps_bottom,
            )
            return (forces, self._fibre_x, y, area)

        x = self._fibre_x
        y = self._fibre_y
        area = self._fibre_area
        material_type = self._fibre_mat
        material_index = self._fibre_mi

        cx, cy = self.section_centroid_x, self.section_centroid_y
        strains = strain_state.strain_field(x - cx, y - cy)
        stresses = np.zeros_like(strains)

        conc_mask = material_type == "concrete"
        if np.any(conc_mask):
            stresses[conc_mask] = self._biaxial._concrete_stress_with_options(strains[conc_mask])

        steel_mask = material_type == "steel"
        if np.any(steel_mask):
            steel_strains = strains[steel_mask]
            steel_indices = material_index[steel_mask]
            steel_stresses = np.zeros_like(steel_strains)
            for gi, sm in enumerate(self.steel_models):
                m = steel_indices == gi
                if np.any(m):
                    steel_stresses[m] = sm.get_stress_array(steel_strains[m])
            if self.ignore_compression_steel:
                steel_stresses[steel_strains > 0] = 0.0
            stresses[steel_mask] = steel_stresses

        forces = stresses * area
        return (forces, x, y, area)

    # ----------------------------
    # Tension shift (delegate)
    # ----------------------------

    def _compute_z_d_for_moment(
        self,
        M_Ed: float,
        N_Ed: float = 0.0,
        use_mechanical_lever_arm: bool = False,
        z_d_upper: float = 0.95,
        z_d_lower: float = 0.65,
        z_d_approx: float = 0.9,
        warn_on_fallback: bool = False,
    ) -> Tuple[float, float]:
        """Compute lever arm z and effective depth d."""
        eps_top, eps_bottom = None, None
        if abs(M_Ed) > 1e-6:
            eps_top, eps_bottom = self.find_strains_for_MN(M_Ed, N_Ed)

        d = self.get_effective_depth(M_Ed=M_Ed, N_Ed=N_Ed, eps_top=eps_top, eps_bottom=eps_bottom)
        z, _ = self.get_lever_arm(
            M_Ed=M_Ed, N_Ed=N_Ed, d=d,
            eps_top=eps_top, eps_bottom=eps_bottom,
            use_mechanical_lever_arm=use_mechanical_lever_arm,
            z_d_upper=z_d_upper,
            z_d_lower=z_d_lower,
            z_d_approx=z_d_approx,
            warn_on_fallback=warn_on_fallback,
        )
        return z, d

    def get_effective_depth(
        self,
        M_Ed: float = 0.0,
        N_Ed: float = 0.0,
        eps_top: Optional[float] = None,
        eps_bottom: Optional[float] = None,
    ) -> float:
        """
        Effective depth from compression face to tension steel centroid.

        Simplified: uses section height minus cover to bottom steel.
        """
        y = self._fibre_y
        steel_mask = self._fibre_mat == "steel"
        if not np.any(steel_mask):
            return float(self.section_height)

        # Determine compression face
        if eps_top is not None and eps_bottom is not None:
            comp_from_bottom = (eps_bottom > eps_top)
        else:
            comp_from_bottom = False

        steel_y = y[steel_mask]
        if comp_from_bottom:
            # Compression at bottom, tension at top
            y_tension = float(np.max(steel_y))
            return y_tension - float(self.section_bottom)
        else:
            # Compression at top, tension at bottom
            y_tension = float(np.min(steel_y))
            return float(self.section_top) - y_tension

    def get_lever_arm(
        self,
        M_Ed: float = 0.0,
        N_Ed: float = 0.0,
        d: Optional[float] = None,
        eps_top: Optional[float] = None,
        eps_bottom: Optional[float] = None,
        *,
        strain_state: Optional["StrainState"] = None,
        use_mechanical_lever_arm: bool = False,
        z_d_upper: float = 0.95,
        z_d_lower: float = 0.65,
        z_d_approx: float = 0.9,
        warn_on_fallback: bool = False,
        force_virtual: bool = False,
    ) -> Tuple[float, Optional[float]]:
        """Lever arm z and rigorous lever arm (if computed)."""
        if d is None:
            d = self.get_effective_depth(M_Ed=M_Ed, N_Ed=N_Ed, eps_top=eps_top, eps_bottom=eps_bottom)

        z_approx = z_d_approx * d
        z_upper = z_d_upper * d
        z_lower = z_d_lower * d

        if not use_mechanical_lever_arm:
            return (z_approx, None)

        # Biaxial: project force centroids along compression direction
        if strain_state is not None and strain_state.is_biaxial:
            try:
                z_rigorous = self._compute_lever_arm_from_strain_state(strain_state)
                if z_rigorous is not None:
                    z = max(z_lower, min(z_rigorous, z_upper))
                    return (z, z_rigorous)
            except Exception:
                if warn_on_fallback:
                    warnings.warn(
                        f"FreeNADiagramAdapter: Biaxial lever arm failed, falling back to {z_d_approx:.2f}d",
                        stacklevel=3,
                    )
            return (z_approx, None)

        # 1D rigorous: compute from fibre forces
        if eps_top is not None and eps_bottom is not None:
            try:
                forces, y_coords, _ = self.get_fibre_forces_from_end_strains(eps_top, eps_bottom)
                tension_mask = forces < 0
                compression_mask = forces > 0

                if np.any(tension_mask) and np.any(compression_mask):
                    y_T = float(np.sum(-forces[tension_mask] * y_coords[tension_mask]) / np.sum(-forces[tension_mask]))
                    y_C = float(np.sum(forces[compression_mask] * y_coords[compression_mask]) / np.sum(forces[compression_mask]))
                    z_rigorous = abs(y_C - y_T)
                    z = max(z_lower, min(z_rigorous, z_upper))
                    return (z, z_rigorous)
            except Exception:
                if warn_on_fallback:
                    warnings.warn(
                        f"FreeNADiagramAdapter: Rigorous lever arm failed, falling back to {z_d_approx:.2f}d",
                        stacklevel=3,
                    )

        return (z_approx, None)

    def _compute_lever_arm_from_strain_state(
        self,
        strain_state: "StrainState",
    ) -> Optional[float]:
        """
        Mechanical lever arm projected along compression direction for biaxial.
        """
        forces, x_coords, y_coords, _ = self.get_fibre_forces_from_strain_state(
            strain_state,
        )

        tension_mask = forces < 0
        compression_mask = forces > 0

        if (not np.any(tension_mask)) or (not np.any(compression_mask)):
            return None

        T_total = float(np.sum(-forces[tension_mask]))
        C_total = float(np.sum(forces[compression_mask]))

        if T_total <= 0 or C_total <= 0:
            return None

        cx, cy = self.section_centroid_x, self.section_centroid_y
        dx, dy = strain_state.compression_direction
        if abs(dx) < 1e-18 and abs(dy) < 1e-18:
            return None

        proj = dx * (x_coords - cx) + dy * (y_coords - cy)

        proj_T = float(
            np.sum((-forces[tension_mask]) * proj[tension_mask]) / T_total
        )
        proj_C = float(
            np.sum((forces[compression_mask]) * proj[compression_mask]) / C_total
        )

        z_mech = abs(proj_C - proj_T)
        if not np.isfinite(z_mech):
            return None

        return float(z_mech)

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
        use_mechanical_lever_arm: bool = False,
        z_d_upper: float = 0.95,
        z_d_lower: float = 0.65,
        z_d_approx: float = 0.9,
        warn_on_fallback: bool = False,
    ) -> "TensionShiftResult":
        """Apply EC2 tension shift — delegates to shear_utils with adapter's z, d."""
        from materials.reinforced_concrete.code_checks.ec2_2004.shear_utils import (
            calculate_tension_shift,
            calculate_section_breadth,
        )

        z, d = self._compute_z_d_for_moment(
            M_Ed=M_Ed, N_Ed=N_Ed,
            use_mechanical_lever_arm=use_mechanical_lever_arm,
            z_d_upper=z_d_upper,
            z_d_lower=z_d_lower,
            z_d_approx=z_d_approx,
            warn_on_fallback=warn_on_fallback,
        )

        b_w = None
        f_cd = None
        f_ck = None
        sigma_cp = 0.0

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

        return calculate_tension_shift(
            M_Ed=M_Ed,
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

    # ----------------------------
    # Tension limit (delegated)
    # ----------------------------

    def _eps_tension_limit(self) -> float:
        return self._biaxial._eps_tension_limit()

    # ----------------------------
    # Concrete stress (delegated)
    # ----------------------------

    def _concrete_stress_with_options(
        self,
        concrete_strains: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        return self._biaxial._concrete_stress_with_options(concrete_strains)

    # ----------------------------
    # calculate_point_from_end_strains (for compatibility)
    # ----------------------------

    def calculate_point_from_end_strains(
        self,
        eps_top: float,
        eps_bottom: float,
    ) -> InteractionPoint:
        """Compute (N, M) from projected end strains."""
        strains = self._strain_field_from_end_strains(eps_top, eps_bottom)
        stresses = np.zeros_like(strains)

        conc_mask = self._fibre_mat == "concrete"
        steel_mask = self._fibre_mat == "steel"

        if np.any(conc_mask):
            stresses[conc_mask] = self._biaxial._concrete_stress_with_options(strains[conc_mask])

        if np.any(steel_mask):
            steel_strains = strains[steel_mask]
            steel_indices = self._fibre_mi[steel_mask]
            steel_stresses = np.zeros_like(steel_strains)
            for gi, sm in enumerate(self.steel_models):
                m = steel_indices == gi
                if np.any(m):
                    steel_stresses[m] = sm.get_stress_array(steel_strains[m])
            if self.ignore_compression_steel:
                steel_stresses[steel_strains > 0] = 0.0
            stresses[steel_mask] = steel_stresses

        N, M = self.calculate_section_forces(stresses)

        max_conc = float(np.max(np.abs(strains[conc_mask]))) if np.any(conc_mask) else 0.0
        max_steel = float(np.max(np.abs(strains[steel_mask]))) if np.any(steel_mask) else 0.0

        return InteractionPoint(
            N=as_float(N),
            M=as_float(M),
            neutral_axis_depth=0.0,
            compression_from_bottom=(eps_bottom > eps_top),
            max_concrete_strain=max_conc,
            max_steel_strain=max_steel,
        )

    def plot_mn(
        self,
        *,
        load_points=None,
        show_vectors: bool = False,
        show_metadata: bool = True,
        n_points: int = 120,
        Mz_slice: float = 0.0,
        save_path=None,
        show: bool = True,
        title=None,
        width: int = 900,
        height: int = 700,
    ):
        from materials.reinforced_concrete.analysis.mn_diagram_viewer import MNDiagramViewer

        viewer = MNDiagramViewer(self)
        return viewer.plot(
            load_points=load_points,
            show_vectors=show_vectors,
            show_metadata=show_metadata,
            n_points=n_points,
            Mz_slice=Mz_slice,
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
        title=None,
        width: int = 1200,
        height: int = 1000,
        section_render: str = "points",
    ):
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

    def __repr__(self) -> str:
        return (
            f"FreeNADiagramAdapter("
            f"section={self.section.section_name}, "
            f"biaxial={self._biaxial!r})"
        )
