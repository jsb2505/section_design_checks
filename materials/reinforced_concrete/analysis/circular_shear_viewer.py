"""
Shear visualization helpers for circular section EC2 shear checks (Orr 2012).

Provides plotting routines for comparative studies of:
- cot(theta) sweeps with lambda_1*lambda_2 efficiency factors
- cot(theta) vs tension-shift / utilization
- axial force vs cot(theta) heatmaps

Link angle is fixed at 90 deg for circular sections, so link-angle sweep
plots are not included.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, TYPE_CHECKING, Union

import numpy as np

from materials.core.units import ForceUnit, to_kn
from materials.reinforced_concrete.code_checks.ec2_2004.shear_check import ShearLoadCase
from materials.reinforced_concrete.code_checks.ec2_2004.shear_utils import (
    calculate_tension_shift,
    find_alpha_cw,
    find_nu_1_factor,
    find_V_Rd_c_cracked,
    sigma_cp_from_N_and_area,
    cap_sigma_cp_upper,
)
from materials.reinforced_concrete.analysis.shear_viewer import (
    _axis_centers_from_edges,
    _build_axial_moment_plot_domain,
    _build_axis_edges,
    _build_horizontal_clip_masks,
    _build_outside_clip_masks,
    _as_load_case,
    _build_slider_animation_controls,
    _build_nice_force_slider_values,
    _build_slider_values,
    _format_slider_numeric_label,
    _get_closed_mn_polyline,
    _get_force_band_fixed_m,
    _normalize_mn_loadcases,
    _show_or_save,
    _subdivide_axis,
    _utilization_colorscale,
)

if TYPE_CHECKING:
    from materials.reinforced_concrete.analysis.interaction_diagram import MNInteractionDiagram
    from materials.reinforced_concrete.code_checks.ec2_2004.circular_section_check import (
        CircularSectionCheck,
    )


@dataclass(frozen=True)
class _CircularStudyContext:
    """Load-case dependent values reused across sweeps (circular variant)."""

    V_Ed: float
    M_Ed: float
    N_Ed: float
    d: float
    z: float
    sigma_cp: float
    rho_l: float
    V_Rd_c: float
    V_Rd_c_cracked: float
    V_Rd_c_uncracked: float
    cot_min: float
    cot_max: float
    # Circular-specific
    lambda_1: float
    lambda_2: float
    b_w: float
    b_wc: float
    b_wt: float
    z_0: float


@dataclass(frozen=True)
class _CircularCotThetaStudySeries:
    """Computed series used by cot(theta)-based plotting methods."""

    context: _CircularStudyContext
    cot_vals: np.ndarray
    V_Rd_s_vals: list[float]
    V_Rd_max_theta_vals: list[float]
    util_vals: list[float]
    M_add_vals: list[float]
    V_Rd_s_design: float
    V_Rd_max_design: float
    cot_intersection: Optional[float]


class CircularShearViewer:
    """Plotting utilities for ``CircularSectionCheck`` shear studies."""

    def __init__(self, check: "CircularSectionCheck") -> None:
        self.check = check

    def _require_shear_reinforcement(self) -> None:
        if self.check.shear_reinforcement is None:
            raise ValueError("Shear reinforcement is required for shear study plots.")

    def _resolve_plot_diagram(
        self,
    ) -> "MNInteractionDiagram":
        """Return the default cached interaction diagram for plotting."""
        shear_check = self.check._shear_check
        assert shear_check is not None
        return shear_check._get_diagram(ignore_compression_steel=False)

    def _build_context(
        self,
        *,
        load_case: ShearLoadCase,
        use_uncracked_V_Rd_c: bool = False,
        ignore_compression_steel: bool = False,
        diagram: Optional["MNInteractionDiagram"] = None,
    ) -> _CircularStudyContext:
        """Compute shared parameters for a load case once."""
        V_Ed = abs(float(load_case.V_Ed))
        M_Ed = float(load_case.M_Ed)
        N_Ed = float(load_case.N_Ed)

        shear_check = self.check._shear_check
        assert shear_check is not None
        assert self.check._concrete_uls is not None

        # 1. Solve strains once
        eps_top: Optional[float]
        eps_bottom: Optional[float]
        if abs(M_Ed) > 1e-6:
            interaction_diagram = diagram
            if interaction_diagram is None:
                interaction_diagram = shear_check._get_diagram(ignore_compression_steel)
            eps_top, eps_bottom = interaction_diagram.find_strains_for_MN(M_Ed, N_Ed)
        else:
            eps_top, eps_bottom = None, None

        d = shear_check.find_effective_depth(
            M_Ed,
            N_Ed,
            eps_top=eps_top,
            eps_bottom=eps_bottom,
            ignore_compression_steel=ignore_compression_steel,
        )
        z_ec2, z_mech = shear_check.find_lever_arm(
            M_Ed,
            N_Ed,
            d,
            eps_top=eps_top,
            eps_bottom=eps_bottom,
            ignore_compression_steel=ignore_compression_steel,
        )
        z = z_mech if z_mech is not None else z_ec2

        # 2. sigma_cp
        A_c = self.check.section.get_area()
        sigma_cp = sigma_cp_from_N_and_area(N_Ed=N_Ed, area=A_c)
        sigma_cp_capped = cap_sigma_cp_upper(sigma_cp=sigma_cp, f_cd=self.check._f_cd_design)

        # 3. Circular-specific
        z_0 = d - self.check.diameter / 2
        lambda_1 = self.check.calculate_lambda_1(z_0, z)
        lambda_2 = self.check.calculate_lambda_2()
        b_w, b_wc, b_wt = self.check.calculate_equivalent_web_width(d, z)

        rho_l = self.check._find_rho_l(
            M_Ed=M_Ed,
            N_Ed=N_Ed,
            b_w=b_w,
            d=d,
            eps_top=eps_top,
            eps_bottom=eps_bottom,
            ignore_compression_steel=ignore_compression_steel,
        )
        V_Rd_c_cracked = find_V_Rd_c_cracked(
            b_w=b_w,
            d=d,
            rho_l=rho_l,
            sigma_cp=sigma_cp_capped,
            f_ck=self.check._concrete_uls.f_ck,
            gamma_c=self.check._concrete_uls.gamma_c,
        )
        V_Rd_c_uncracked = self.check.calculate_V_Rd_c_uncracked(sigma_cp_capped)
        V_Rd_c = V_Rd_c_uncracked if use_uncracked_V_Rd_c else V_Rd_c_cracked

        cot_min, cot_max = self.check._get_cot_theta_limits(sigma_cp_capped)

        return _CircularStudyContext(
            V_Ed=V_Ed,
            M_Ed=M_Ed,
            N_Ed=N_Ed,
            d=d,
            z=z,
            sigma_cp=sigma_cp_capped,
            rho_l=rho_l,
            V_Rd_c=V_Rd_c,
            V_Rd_c_cracked=V_Rd_c_cracked,
            V_Rd_c_uncracked=V_Rd_c_uncracked,
            cot_min=cot_min,
            cot_max=cot_max,
            lambda_1=lambda_1,
            lambda_2=lambda_2,
            b_w=b_w,
            b_wc=b_wc,
            b_wt=b_wt,
            z_0=z_0,
        )

    def _find_V_Rd_s(self, cot_theta: float, context: _CircularStudyContext) -> float:
        """V_Rd,s with lambda_1*lambda_2 efficiency factors (kN)."""
        reinforcement = self.check.shear_reinforcement
        assert reinforcement is not None
        f_ywd = self.check._f_ywd_design
        V_Rd_s_N = (
            context.lambda_1
            * context.lambda_2
            * reinforcement.area_per_unit_length
            * context.z
            * f_ywd
            * cot_theta
        )
        return to_kn(V_Rd_s_N, ForceUnit.N)

    def _find_V_Rd_max(self, cot_theta: float, context: _CircularStudyContext) -> float:
        """V_Rd,max with circular equivalent web width (kN)."""
        assert self.check._concrete_uls is not None
        f_cd = self.check._f_cd_design
        f_ck = self.check._concrete_uls.f_ck
        alpha_cw = find_alpha_cw(
            f_cd,
            context.sigma_cp,
            use_sigma_cp_for_alpha_cw=self.check.use_sigma_cp_for_alpha_cw,
        )
        nu_1 = find_nu_1_factor(f_ck, link_angle_degrees=90.0)
        tan_theta = 1.0 / cot_theta
        V_Rd_max_N = (
            alpha_cw * context.b_w * context.z * nu_1 * f_cd
            / (cot_theta + tan_theta)
        )
        return to_kn(V_Rd_max_N, ForceUnit.N)

    @staticmethod
    def _find_curve_intersection_x(
        x_vals: np.ndarray,
        y_a_vals: Sequence[float],
        y_b_vals: Sequence[float],
    ) -> Optional[float]:
        """Return first x-position where two sampled curves intersect."""
        diff = np.asarray(y_a_vals, dtype=float) - np.asarray(y_b_vals, dtype=float)
        if diff.size < 2:
            return None

        for i in range(diff.size - 1):
            d0 = float(diff[i])
            d1 = float(diff[i + 1])
            x0 = float(x_vals[i])
            x1 = float(x_vals[i + 1])

            if np.isclose(d0, 0.0):
                return x0
            if np.isclose(d1, 0.0):
                return x1
            if d0 * d1 < 0.0:
                return x0 - d0 * (x1 - x0) / (d1 - d0)

        return None

    def _compute_cot_theta_study_series(
        self,
        *,
        load_case: Union[ShearLoadCase, Dict[str, Any]],
        n_points: int,
        cot_theta_min: Optional[float],
        cot_theta_max: Optional[float],
        use_uncracked_V_Rd_c: bool,
    ) -> _CircularCotThetaStudySeries:
        """Compute reusable cot(theta) sweep values for plotting."""
        assert self.check._concrete_uls is not None
        case = _as_load_case(load_case)
        context = self._build_context(load_case=case, use_uncracked_V_Rd_c=use_uncracked_V_Rd_c)

        cot_min = context.cot_min if cot_theta_min is None else float(cot_theta_min)
        cot_max = context.cot_max if cot_theta_max is None else float(cot_theta_max)
        if cot_min > cot_max:
            cot_min, cot_max = cot_max, cot_min

        cot_vals = np.linspace(cot_min, cot_max, max(2, int(n_points)))

        V_Rd_s_vals: list[float] = []
        V_Rd_max_theta_vals: list[float] = []
        util_vals: list[float] = []
        M_add_vals: list[float] = []

        # Design reference values at code limits
        V_Rd_max_design = self._find_V_Rd_max(context.cot_min, context)
        V_Rd_s_design = self._find_V_Rd_s(context.cot_max, context)

        for cot_theta in cot_vals:
            cot_theta_f = float(cot_theta)
            V_Rd_s = self._find_V_Rd_s(cot_theta_f, context)
            V_Rd_max_theta = self._find_V_Rd_max(cot_theta_f, context)

            V_Rd = min(V_Rd_s, V_Rd_max_design)
            util = context.V_Ed / V_Rd if V_Rd > 0.0 else float("inf")

            shift = calculate_tension_shift(
                M_Ed=context.M_Ed,
                V_Ed=context.V_Ed,
                z=context.z,
                d=context.d,
                b_w=context.b_w,
                f_cd=self.check._f_cd_design,
                f_ck=self.check._concrete_uls.f_ck,
                sigma_cp=context.sigma_cp,
                use_sigma_cp_for_alpha_cw=self.check.use_sigma_cp_for_alpha_cw,
                shear_reinforcement=self.check.shear_reinforcement,
                cot_theta_override=cot_theta_f,
            )

            V_Rd_s_vals.append(V_Rd_s)
            V_Rd_max_theta_vals.append(V_Rd_max_theta)
            util_vals.append(util)
            M_add_vals.append(shift.M_add)

        cot_intersection = self._find_curve_intersection_x(cot_vals, V_Rd_s_vals, V_Rd_max_theta_vals)
        return _CircularCotThetaStudySeries(
            context=context,
            cot_vals=cot_vals,
            V_Rd_s_vals=V_Rd_s_vals,
            V_Rd_max_theta_vals=V_Rd_max_theta_vals,
            util_vals=util_vals,
            M_add_vals=M_add_vals,
            V_Rd_s_design=V_Rd_s_design,
            V_Rd_max_design=V_Rd_max_design,
            cot_intersection=cot_intersection,
        )

    # ------------------------------------------------------------------
    # Plot methods
    # ------------------------------------------------------------------

    def plot_cot_theta_study(
        self,
        *,
        load_case: Union[ShearLoadCase, Dict[str, Any]],
        n_points: int = 60,
        cot_theta_min: Optional[float] = None,
        cot_theta_max: Optional[float] = None,
        use_uncracked_V_Rd_c: bool = False,
        save_path: Optional[Union[str, Path]] = None,
        show: bool = True,
        title: Optional[str] = None,
        width: int = 1000,
        height: int = 560,
    ) -> Any:
        """
        Plot shear-capacity components over a cot(theta) sweep.

        Includes V_Ed, V_Rd,c, V_Rd,s(cot) with lambda_1*lambda_2 factors,
        V_Rd,max(cot) with circular b_w, and design reference lines.

        Args:
            load_case: Shear demand definition as either ``ShearLoadCase`` or a
                ``dict`` with keys ``V_Ed`` and optional ``M_Ed``/``N_Ed`` (kN, kN*m).
            n_points: Number of cot(theta) samples in the sweep.
            cot_theta_min: Optional lower bound for cot(theta).
            cot_theta_max: Optional upper bound for cot(theta).
            use_uncracked_V_Rd_c: If ``True``, use uncracked V_Rd,c (Orr Eq.17).
            save_path: Optional file path for ``fig.write_html(...)`` output.
            show: If ``True``, call ``fig.show()`` before returning.
            title: Optional custom plot title.
            width: Figure width in pixels.
            height: Figure height in pixels.

        Returns:
            plotly.graph_objects.Figure
        """
        self._require_shear_reinforcement()

        try:
            import plotly.graph_objects as go
        except ImportError as e:
            raise ImportError("Plotly is required for plotting. Install with: pip install plotly") from e

        series = self._compute_cot_theta_study_series(
            load_case=load_case,
            n_points=n_points,
            cot_theta_min=cot_theta_min,
            cot_theta_max=cot_theta_max,
            use_uncracked_V_Rd_c=use_uncracked_V_Rd_c,
        )
        context = series.context
        cot_vals = series.cot_vals

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=cot_vals,
                y=[context.V_Ed] * len(cot_vals),
                mode="lines",
                name="V_Ed",
                line=dict(color="black", dash="dot"),
                hovertemplate="cot(theta): %{x:.3f}<br>V_Ed: %{y:.1f} kN<extra></extra>",
            ),
        )
        fig.add_trace(
            go.Scatter(
                x=cot_vals,
                y=[context.V_Rd_c] * len(cot_vals),
                mode="lines",
                name="V_Rd,c",
                line=dict(color="#8c564b", dash="dash"),
                hovertemplate="cot(theta): %{x:.3f}<br>V_Rd,c: %{y:.1f} kN<extra></extra>",
            ),
        )
        fig.add_trace(
            go.Scatter(
                x=cot_vals,
                y=series.V_Rd_s_vals,
                mode="lines",
                name="V_Rd,s(cot)",
                line=dict(color="#1f77b4"),
                hovertemplate="cot(theta): %{x:.3f}<br>V_Rd,s: %{y:.1f} kN<extra></extra>",
            ),
        )
        fig.add_trace(
            go.Scatter(
                x=cot_vals,
                y=series.V_Rd_max_theta_vals,
                mode="lines",
                name="V_Rd,max(cot)",
                line=dict(color="#ff7f0e"),
                hovertemplate="cot(theta): %{x:.3f}<br>V_Rd,max(cot): %{y:.1f} kN<extra></extra>",
            ),
        )
        fig.add_trace(
            go.Scatter(
                x=cot_vals,
                y=[series.V_Rd_max_design] * len(cot_vals),
                mode="lines",
                name="V_Rd,max design",
                line=dict(color="#ff7f0e", dash="dash"),
                hovertemplate="cot(theta): %{x:.3f}<br>V_Rd,max design: %{y:.1f} kN<extra></extra>",
            ),
        )
        fig.add_trace(
            go.Scatter(
                x=cot_vals,
                y=[series.V_Rd_s_design] * len(cot_vals),
                mode="lines",
                name="V_Rd,s design",
                line=dict(color="#1f77b4", dash="dash"),
                hovertemplate="cot(theta): %{x:.3f}<br>V_Rd,s design: %{y:.1f} kN<extra></extra>",
            ),
        )

        if series.cot_intersection is not None:
            y_intersection = float(
                np.interp(series.cot_intersection, cot_vals.tolist(), series.V_Rd_s_vals),
            )
            fig.add_trace(
                go.Scatter(
                    x=[series.cot_intersection, series.cot_intersection, float(cot_vals[0])],
                    y=[0.0, y_intersection, y_intersection],
                    mode="lines",
                    name="V_Ed,max",
                    line=dict(color="#ff0000", dash="dashdot"),
                    customdata=[
                        [series.cot_intersection, y_intersection],
                        [series.cot_intersection, y_intersection],
                        [series.cot_intersection, y_intersection],
                    ],
                    hovertemplate=(
                        "cot(theta): %{customdata[0]:.3f}<br>"
                        "V_Ed,max: %{customdata[1]:.1f} kN<extra></extra>"
                    ),
                ),
            )

        cot_theta_min = self._find_curve_intersection_x(
            cot_vals,
            series.V_Rd_s_vals,
            [context.V_Ed] * len(cot_vals),
        )
        if cot_theta_min is not None and float(cot_vals[0]) <= cot_theta_min <= float(cot_vals[-1]):
            y_vertical = np.linspace(0.0, context.V_Ed, 25)
            fig.add_trace(
                go.Scatter(
                    x=[cot_theta_min] * len(y_vertical),
                    y=y_vertical,
                    mode="lines",
                    name="Cot(theta),min",
                    line=dict(color="#ff0000", dash="dot"),
                    customdata=[[cot_theta_min, context.V_Ed]] * len(y_vertical),
                    hovertemplate=(
                        "cot(theta): %{customdata[0]:.3f}<br>"
                        "V_Ed: %{customdata[1]:.1f} kN<extra></extra>"
                    ),
                ),
            )

        cot_theta_max = self._find_curve_intersection_x(
            cot_vals,
            series.V_Rd_max_theta_vals,
            [context.V_Ed] * len(cot_vals),
        )
        if cot_theta_max is not None and float(cot_vals[0]) <= cot_theta_max <= float(cot_vals[-1]):
            y_vertical = np.linspace(0.0, context.V_Ed, 25)
            fig.add_trace(
                go.Scatter(
                    x=[cot_theta_max] * len(y_vertical),
                    y=y_vertical,
                    mode="lines",
                    name="Cot(theta),max",
                    line=dict(color="#ff0000", dash="dot"),
                    customdata=[[cot_theta_max, context.V_Ed]] * len(y_vertical),
                    hovertemplate=(
                        "cot(theta): %{customdata[0]:.3f}<br>"
                        "V_Ed: %{customdata[1]:.1f} kN<extra></extra>"
                    ),
                ),
            )

        subtitle = (
            f"lambda_1={context.lambda_1:.3f}, lambda_2={context.lambda_2:.2f}, "
            f"b_w={context.b_w:.1f} mm"
        )
        fig.update_xaxes(title_text="cot(theta)")
        fig.update_yaxes(title_text="Capacity (kN)")
        fig.update_layout(
            title=title or f"Circular Shear Capacity Study vs cot(theta)<br><sub>{subtitle}</sub>",
            template="plotly_white",
            width=width,
            height=height,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1.0,
                xanchor="left",
                x=1.02,
            ),
            margin=dict(r=240),
        )

        _show_or_save(fig, save_path=save_path, show=show)
        return fig

    def plot_cot_theta_moment_shift_study(
        self,
        *,
        load_case: Union[ShearLoadCase, Dict[str, Any]],
        n_points: int = 60,
        cot_theta_min: Optional[float] = None,
        cot_theta_max: Optional[float] = None,
        use_uncracked_V_Rd_c: bool = False,
        save_path: Optional[Union[str, Path]] = None,
        show: bool = True,
        title: Optional[str] = None,
        width: int = 1000,
        height: int = 560,
    ) -> Any:
        """
        Plot utilization and tension-shift add-on versus cot(theta).

        Dual-axis plot showing:
        - utilization ratio ``V_Ed / V_Rd``
        - additional moment from tension shift ``M_add``

        Args:
            load_case: Shear demand definition as either ``ShearLoadCase`` or a
                ``dict`` with keys ``V_Ed`` and optional ``M_Ed``/``N_Ed`` (kN, kN*m).
            n_points: Number of cot(theta) samples in the sweep.
            cot_theta_min: Optional lower bound for cot(theta).
            cot_theta_max: Optional upper bound for cot(theta).
            use_uncracked_V_Rd_c: If ``True``, use uncracked V_Rd,c (Orr Eq.17).
            save_path: Optional file path for ``fig.write_html(...)`` output.
            show: If ``True``, call ``fig.show()`` before returning.
            title: Optional custom plot title.
            width: Figure width in pixels.
            height: Figure height in pixels.

        Returns:
            plotly.graph_objects.Figure
        """
        self._require_shear_reinforcement()

        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError as e:
            raise ImportError("Plotly is required for plotting. Install with: pip install plotly") from e

        series = self._compute_cot_theta_study_series(
            load_case=load_case,
            n_points=n_points,
            cot_theta_min=cot_theta_min,
            cot_theta_max=cot_theta_max,
            use_uncracked_V_Rd_c=use_uncracked_V_Rd_c,
        )

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Scatter(
                x=series.cot_vals,
                y=series.util_vals,
                mode="lines",
                name="Utilization",
                line=dict(color="#d62728"),
                hovertemplate="cot(theta): %{x:.3f}<br>Utilization: %{y:.3f}<extra></extra>",
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=series.cot_vals,
                y=series.M_add_vals,
                mode="lines",
                name="M_add (tension shift)",
                line=dict(color="#9467bd"),
                hovertemplate="cot(theta): %{x:.3f}<br>M_add: %{y:.2f} kN*m<extra></extra>",
            ),
            secondary_y=True,
        )
        fig.add_trace(
            go.Scatter(
                x=series.cot_vals,
                y=[1.0] * len(series.cot_vals),
                mode="lines",
                name="Utilization = 1.0",
                line=dict(color="#ff0000", dash="dot"),
                hovertemplate="Utilization limit<extra></extra>",
            ),
            secondary_y=False,
        )

        util_intersection = self._find_curve_intersection_x(
            series.cot_vals,
            series.util_vals,
            [1.0] * len(series.cot_vals),
        )
        if util_intersection is not None:
            m_add_intersection = float(
                np.interp(
                    util_intersection,
                    series.cot_vals.tolist(),
                    series.M_add_vals,
                ),
            )
            y_vertical = np.linspace(0.0, 1.0, 25)
            fig.add_trace(
                go.Scatter(
                    x=[util_intersection] * len(y_vertical),
                    y=y_vertical,
                    mode="lines",
                    name="Utilization = 1.0 intercept",
                    line=dict(color="#2f2f2f", dash="dash"),
                    customdata=[[util_intersection, m_add_intersection]] * len(y_vertical),
                    hovertemplate=(
                        "cot(theta): %{customdata[0]:.3f}<br>"
                        "M_add at util=1.0: %{customdata[1]:.2f} kN*m<extra></extra>"
                    ),
                ),
                secondary_y=False,
            )

        fig.update_xaxes(title_text="cot(theta)")
        fig.update_yaxes(title_text="Utilization", secondary_y=False)
        fig.update_yaxes(title_text="M_add (kN*m)", secondary_y=True)
        fig.update_layout(
            title=title or "Circular Tension-Shift Study vs cot(theta)",
            template="plotly_white",
            width=width,
            height=height,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1.0,
                xanchor="left",
                x=1.02,
            ),
            margin=dict(r=240),
        )

        _show_or_save(fig, save_path=save_path, show=show)
        return fig

    def plot_force_cot_theta_contour(
        self,
        *,
        load_case: Union[ShearLoadCase, Dict[str, Any]],
        n_axial: int = 31,
        n_moment: int = 31,
        moment_on_y_axis: bool = False,
        cot_theta_min: Optional[float] = None,
        cot_theta_max: Optional[float] = None,
        n_cot: int = 40,
        metric: str = "utilization",
        use_uncracked_V_Rd_c: bool = False,
        save_path: Optional[Union[str, Path]] = None,
        show: bool = True,
        title: Optional[str] = None,
        width: int = 980,
        height: int = 760,
    ) -> Any:
        """
        Plot a cot(theta)-vs-force heatmap with a slider for the other force.

        Depending on ``moment_on_y_axis``, the y-axis is either ``M_Ed`` or
        ``N_Ed``. The slider controls the other quantity. Only in-envelope
        M-N states are computed and plotted.

        Args:
            load_case: Base shear load case (``V_Ed`` is kept fixed). Can be
                ``ShearLoadCase`` or a ``dict`` with keys ``V_Ed`` and optional
                ``M_Ed``/``N_Ed``.
            n_axial: Number of axial-force samples used on the heatmap force axis.
            n_moment: Number of moment samples used on the heatmap force axis.
            moment_on_y_axis: If ``True``, the y-axis is moment and the slider
                controls axial force. If ``False``, the y-axis is axial force and
                the slider controls moment. Ranges are derived from the current
                M-N interaction diagram, and slider values are auto-generated
                with nice round increments targeting about 50 steps.
            cot_theta_min: Optional lower bound for cot(theta).
            cot_theta_max: Optional upper bound for cot(theta).
            n_cot: Number of cot(theta) samples.
            metric: Response quantity on the color axis. Supported values are:
                ``"utilization"``, ``"capacity"``, ``"v_rd_s"``, and ``"v_rd_max"``.
            use_uncracked_V_Rd_c: If ``True``, use uncracked V_Rd,c (Orr Eq.17).
            save_path: Optional file path for ``fig.write_html(...)`` output.
            show: If ``True``, call ``fig.show()`` before returning.
            title: Optional custom plot title.
            width: Figure width in pixels.
            height: Figure height in pixels.

        Returns:
            plotly.graph_objects.Figure
        """
        self._require_shear_reinforcement()

        try:
            import plotly.graph_objects as go
        except ImportError as e:
            raise ImportError("Plotly is required for plotting. Install with: pip install plotly") from e

        case = _as_load_case(load_case)
        plot_diagram = self._resolve_plot_diagram()
        diagram_n_points = 160
        diagram_polyline = _get_closed_mn_polyline(plot_diagram, n_points=diagram_n_points)
        m_curve = np.asarray([point[0] for point in diagram_polyline], dtype=float)
        n_curve = np.asarray([point[1] for point in diagram_polyline], dtype=float)
        m_global_min = float(np.min(m_curve))
        m_global_max = float(np.max(m_curve))
        n_global_min = float(np.min(n_curve))
        n_global_max = float(np.max(n_curve))

        metric_key = metric.strip().lower()
        valid_metrics = {"utilization", "capacity", "v_rd_s", "v_rd_max"}
        if metric_key not in valid_metrics:
            raise ValueError(f"metric must be one of {sorted(valid_metrics)}.")

        if moment_on_y_axis:
            y_vals = np.linspace(m_global_min, m_global_max, max(2, int(n_moment)))
            slider_vals = _build_nice_force_slider_values(
                value_min=n_global_min,
                value_max=n_global_max,
            )
            y_label = "Moment My_Ed (kN*m)"
            y_hover_name = "My_Ed"
            y_hover_format = ".2f"
            y_hover_unit = "kN*m"
            slider_prefix = "N_Ed (kN): "
        else:
            y_vals = np.linspace(n_global_min, n_global_max, max(2, int(n_axial)))
            slider_vals = _build_nice_force_slider_values(
                value_min=m_global_min,
                value_max=m_global_max,
            )
            y_label = "Axial force N_Ed (kN)"
            y_hover_name = "N_Ed"
            y_hover_format = ".1f"
            y_hover_unit = "kN"
            slider_prefix = "My_Ed (kN*m): "

        y_edges = _build_axis_edges(y_vals)
        display_oversample = 3
        display_y_edges = _subdivide_axis(y_edges, display_oversample)
        display_y_vals = _axis_centers_from_edges(display_y_edges)

        context_grid: list[list[Optional[_CircularStudyContext]]] = [
            [None for _ in range(len(y_vals))]
            for _ in range(len(slider_vals))
        ]
        band_lower_vals = np.full(len(slider_vals), np.nan, dtype=float)
        band_upper_vals = np.full(len(slider_vals), np.nan, dtype=float)
        cot_min_candidates: list[float] = []
        cot_max_candidates: list[float] = []

        for i_slider, slider_val in enumerate(slider_vals):
            if moment_on_y_axis:
                n_cap, m_pos, m_neg = plot_diagram.get_capacity_fixed_n(
                    N_Ed=float(slider_val),
                    n_points=diagram_n_points,
                )
                if n_cap is None or m_pos is None or m_neg is None:
                    continue
                fixed_force = float(n_cap)
                y_lower = min(float(m_neg), float(m_pos))
                y_upper = max(float(m_neg), float(m_pos))
            else:
                m_cap, n_upper, n_lower = _get_force_band_fixed_m(diagram_polyline, float(slider_val))
                if m_cap is None or n_upper is None or n_lower is None:
                    continue
                fixed_force = float(m_cap)
                y_lower = float(n_lower)
                y_upper = float(n_upper)

            band_lower_vals[i_slider] = y_lower
            band_upper_vals[i_slider] = y_upper
            tol = 1e-9 * max(1.0, abs(y_lower), abs(y_upper))

            for i_y, y_val in enumerate(y_vals):
                cell_lower = float(y_edges[i_y])
                cell_upper = float(y_edges[i_y + 1])
                if cell_upper < y_lower - tol or cell_lower > y_upper + tol:
                    continue

                y_eval = float(np.clip(float(y_val), y_lower, y_upper))
                if moment_on_y_axis:
                    sweep_case = ShearLoadCase(V_Ed=case.V_Ed, M_Ed=y_eval, N_Ed=fixed_force)
                else:
                    sweep_case = ShearLoadCase(V_Ed=case.V_Ed, M_Ed=fixed_force, N_Ed=y_eval)

                context = self._build_context(
                    load_case=sweep_case,
                    use_uncracked_V_Rd_c=use_uncracked_V_Rd_c,
                    diagram=plot_diagram,
                )
                context_grid[i_slider][i_y] = context
                cot_min_candidates.append(float(context.cot_min))
                cot_max_candidates.append(float(context.cot_max))

        if cot_min_candidates and cot_max_candidates:
            cot_min = min(cot_min_candidates) if cot_theta_min is None else float(cot_theta_min)
            cot_max = max(cot_max_candidates) if cot_theta_max is None else float(cot_theta_max)
        else:
            fallback_context = self._build_context(
                load_case=case,
                use_uncracked_V_Rd_c=use_uncracked_V_Rd_c,
                diagram=plot_diagram,
            )
            cot_min = float(fallback_context.cot_min) if cot_theta_min is None else float(cot_theta_min)
            cot_max = float(fallback_context.cot_max) if cot_theta_max is None else float(cot_theta_max)
        if cot_min > cot_max:
            cot_min, cot_max = cot_max, cot_min

        cot_vals = _build_slider_values(
            value_min=cot_min,
            value_max=cot_max,
            n_points=n_cot,
            step=None,
        )
        z_volume = np.full((len(slider_vals), len(y_vals), len(cot_vals)), np.nan, dtype=float)

        for i_slider, row in enumerate(context_grid):
            for i_y, context in enumerate(row):
                if context is None:
                    continue
                for i_cot, cot_theta in enumerate(cot_vals):
                    cot_f = float(cot_theta)
                    V_Rd_s = self._find_V_Rd_s(cot_f, context)
                    V_Rd_max = self._find_V_Rd_max(cot_f, context)

                    V_Rd = min(V_Rd_s, V_Rd_max)

                    if metric_key == "utilization":
                        value = context.V_Ed / V_Rd if V_Rd > 0.0 else float("inf")
                    elif metric_key == "capacity":
                        value = V_Rd
                    elif metric_key == "v_rd_s":
                        value = V_Rd_s
                    else:
                        value = V_Rd_max
                    z_volume[i_slider, i_y, i_cot] = value

        if metric_key == "utilization":
            colorbar_title = "Utilization"
            zmin = 0.0
            finite_vals = z_volume[np.isfinite(z_volume)]
            zmax = max(1.5, float(np.nanmax(finite_vals))) if finite_vals.size else 1.5
            z_plot_volume = np.array(z_volume, copy=True)
            z_plot_volume[np.isinf(z_plot_volume)] = zmax
            colorscale = _utilization_colorscale(zmin=zmin, zmax=zmax)
        else:
            colorbar_title = "kN"
            colorscale = "Viridis"
            zmin = None
            zmax = None
            z_plot_volume = z_volume

        display_z_volume = np.repeat(z_plot_volume, display_oversample, axis=1)
        x_min = float(cot_vals[0])
        x_max = float(cot_vals[-1])
        y_min = float(y_vals[0])
        y_max = float(y_vals[-1])

        def _frame_band(index: int) -> tuple[float, float]:
            lower = float(band_lower_vals[index])
            upper = float(band_upper_vals[index])
            if not np.isfinite(lower) or not np.isfinite(upper):
                return y_min, y_min
            return (min(lower, upper), max(lower, upper))

        first_lower, first_upper = _frame_band(0)
        first_top_mask, first_bottom_mask = _build_horizontal_clip_masks(
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            y_lower=first_lower,
            y_upper=first_upper,
        )
        first_slice = display_z_volume[0, :, :]
        hovertemplate = (
            "cot(theta): %{x:.3f}<br>"
            f"{y_hover_name}: %{{y:{y_hover_format}}} {y_hover_unit}<br>"
            f"{metric_key}: "
            "%{z:.3f}<extra></extra>"
        )

        fig = go.Figure()
        fig.add_trace(
            go.Heatmap(
                x=cot_vals,
                y=display_y_vals,
                z=first_slice,
                colorscale=colorscale,
                zmin=zmin,
                zmax=zmax,
                colorbar=dict(title=colorbar_title),
                hovertemplate=hovertemplate,
                name=metric_key,
            ),
        )
        if metric_key == "utilization":
            fig.add_trace(
                go.Contour(
                    x=cot_vals,
                    y=display_y_vals,
                    z=first_slice,
                    contours=dict(start=1.0, end=1.0, size=1.0, coloring="none"),
                    line=dict(color="black", width=2),
                    showscale=False,
                    name="Utilization = 1.0",
                    hoverinfo="skip",
                ),
            )
        fig.add_trace(
            go.Scatter(
                x=cot_vals,
                y=[first_upper] * len(cot_vals),
                mode="lines",
                name="Upper M-N limit",
                line=dict(color="black", width=2),
                hoverinfo="skip",
            ),
        )
        fig.add_trace(
            go.Scatter(
                x=cot_vals,
                y=[first_lower] * len(cot_vals),
                mode="lines",
                name="Lower M-N limit",
                line=dict(color="black", width=2),
                hoverinfo="skip",
            ),
        )
        fig.add_trace(
            go.Scatter(
                x=first_top_mask[0],
                y=first_top_mask[1],
                mode="lines",
                fill="toself",
                fillcolor="white",
                line=dict(color="white", width=0),
                showlegend=False,
                hoverinfo="skip",
                name="_top_mask",
            ),
        )
        fig.add_trace(
            go.Scatter(
                x=first_bottom_mask[0],
                y=first_bottom_mask[1],
                mode="lines",
                fill="toself",
                fillcolor="white",
                line=dict(color="white", width=0),
                showlegend=False,
                hoverinfo="skip",
                name="_bottom_mask",
            ),
        )

        frames = []
        for i_slider, slider_val in enumerate(slider_vals):
            lower_bound, upper_bound = _frame_band(i_slider)
            top_mask, bottom_mask = _build_horizontal_clip_masks(
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
                y_lower=lower_bound,
                y_upper=upper_bound,
            )
            z_slice = display_z_volume[i_slider, :, :]
            slider_label = _format_slider_numeric_label(float(slider_val))
            frame_data: list[Any] = [
                go.Heatmap(
                    x=cot_vals,
                    y=display_y_vals,
                    z=z_slice,
                    colorscale=colorscale,
                    zmin=zmin,
                    zmax=zmax,
                    showscale=False,
                    hovertemplate=hovertemplate,
                    name=metric_key,
                ),
            ]
            if metric_key == "utilization":
                frame_data.append(
                    go.Contour(
                        x=cot_vals,
                        y=display_y_vals,
                        z=z_slice,
                        contours=dict(start=1.0, end=1.0, size=1.0, coloring="none"),
                        line=dict(color="black", width=2),
                        showscale=False,
                        name="Utilization = 1.0",
                        hoverinfo="skip",
                    ),
                )
            frame_data.extend(
                [
                    go.Scatter(
                        x=cot_vals,
                        y=[upper_bound] * len(cot_vals),
                        mode="lines",
                        name="Upper M-N limit",
                        line=dict(color="black", width=2),
                        hoverinfo="skip",
                    ),
                    go.Scatter(
                        x=cot_vals,
                        y=[lower_bound] * len(cot_vals),
                        mode="lines",
                        name="Lower M-N limit",
                        line=dict(color="black", width=2),
                        hoverinfo="skip",
                    ),
                    go.Scatter(
                        x=top_mask[0],
                        y=top_mask[1],
                        mode="lines",
                        fill="toself",
                        fillcolor="white",
                        line=dict(color="white", width=0),
                        showlegend=False,
                        hoverinfo="skip",
                        name="_top_mask",
                    ),
                    go.Scatter(
                        x=bottom_mask[0],
                        y=bottom_mask[1],
                        mode="lines",
                        fill="toself",
                        fillcolor="white",
                        line=dict(color="white", width=0),
                        showlegend=False,
                        hoverinfo="skip",
                        name="_bottom_mask",
                    ),
                ],
            )
            frames.append({"name": slider_label, "data": frame_data})
        fig.frames = frames

        mode_title = "Moment on y-axis" if moment_on_y_axis else "Axial force on y-axis"
        layout_kwargs: dict[str, Any] = dict(
            title=title or f"Circular Force vs cot(theta): {metric_key} ({mode_title})",
            xaxis_title="cot(theta)",
            yaxis_title=y_label,
            yaxis=dict(range=[y_min, y_max], autorange=False),
            template="plotly_white",
            width=width,
            height=height,
        )
        if len(slider_vals) > 1:
            steps = [
                {
                    "label": _format_slider_numeric_label(float(slider_val)),
                    "method": "animate",
                    "args": [
                        [_format_slider_numeric_label(float(slider_val))],
                        {
                            "frame": {"duration": 0, "redraw": True},
                            "mode": "immediate",
                            "transition": {"duration": 0},
                        },
                    ],
                }
                for slider_val in slider_vals
            ]
            layout_kwargs.update(
                _build_slider_animation_controls(
                    steps=steps,
                    currentvalue_prefix=slider_prefix,
                )
            )
        fig.update_layout(**layout_kwargs)

        _show_or_save(fig, save_path=save_path, show=show)
        return fig

    def plot_axial_moment_contour(
        self,
        *,
        V_Ed: float,
        loadcases: Optional[Sequence[Union[Dict[str, Any], Sequence[float]]]] = None,
        n_diagram_points: int = 120,
        n_moment: int = 41,
        n_axial: int = 31,
        cot_theta_min: Optional[float] = None,
        cot_theta_max: Optional[float] = None,
        n_cot: int = 20,
        cot_theta_step: Optional[float] = 0.05,
        use_uncracked_V_Rd_c: bool = False,
        save_path: Optional[Union[str, Path]] = None,
        show: bool = True,
        title: Optional[str] = None,
        width: int = 1000,
        height: int = 760,
    ) -> Any:
        """
        Plot an M-N utilization heatmap clipped to the interaction diagram (circular).

        The x-axis is ``M_Ed``, y-axis is ``N_Ed`` and heatmap color is shear
        utilization. The plotted heatmap is masked outside the M-N envelope,
        and the capacity boundary is overlaid as a line trace. A contour line
        for ``utilization = 1.0`` is included for each cot(theta) frame.
        Slider values use 0.05 cot(theta) increments by default; set
        ``cot_theta_step`` to ``None`` or non-positive to use ``n_cot``
        linear spacing.
        """
        self._require_shear_reinforcement()

        try:
            import plotly.graph_objects as go
        except ImportError as e:
            raise ImportError("Plotly is required for plotting. Install with: pip install plotly") from e

        plot_diagram = self._resolve_plot_diagram()
        plot_domain = _build_axial_moment_plot_domain(
            plot_diagram,
            n_diagram_points=n_diagram_points,
            n_moment=n_moment,
            n_axial=n_axial,
        )
        display_oversample = 3
        display_m_edges = _subdivide_axis(plot_domain.m_edges, display_oversample)
        display_n_edges = _subdivide_axis(plot_domain.n_edges, display_oversample)
        display_m_vals = _axis_centers_from_edges(display_m_edges)
        display_n_vals = _axis_centers_from_edges(display_n_edges)
        left_mask, right_mask = _build_outside_clip_masks(
            plot_diagram,
            y_edges=display_n_edges,
            x_min=float(display_m_edges[0]),
            x_max=float(display_m_edges[-1]),
            n_diagram_points=n_diagram_points,
        )
        plotted_loadcases = _normalize_mn_loadcases(loadcases)

        context_grid: list[list[Optional[_CircularStudyContext]]] = [
            [None for _ in range(len(plot_domain.m_vals))]
            for _ in range(len(plot_domain.n_vals))
        ]
        cot_min_candidates: list[float] = []
        cot_max_candidates: list[float] = []

        for i_n, n_ed in enumerate(plot_domain.n_vals):
            for i_m, m_ed in enumerate(plot_domain.m_vals):
                if not plot_domain.valid_mask[i_n, i_m]:
                    continue
                left_bound = float(plot_domain.center_left[i_n])
                right_bound = float(plot_domain.center_right[i_n])
                m_eval = float(np.clip(float(m_ed), left_bound, right_bound))
                case = ShearLoadCase(V_Ed=V_Ed, M_Ed=m_eval, N_Ed=float(n_ed))
                context = self._build_context(
                    load_case=case,
                    use_uncracked_V_Rd_c=use_uncracked_V_Rd_c,
                    diagram=plot_diagram,
                )
                context_grid[i_n][i_m] = context
                cot_min_candidates.append(float(context.cot_min))
                cot_max_candidates.append(float(context.cot_max))

        if cot_min_candidates and cot_max_candidates:
            cot_default_min = min(cot_min_candidates)
            cot_default_max = max(cot_max_candidates)
        else:
            fallback_context = self._build_context(
                load_case=ShearLoadCase(V_Ed=V_Ed, M_Ed=0.0, N_Ed=0.0),
                use_uncracked_V_Rd_c=use_uncracked_V_Rd_c,
                diagram=plot_diagram,
            )
            cot_default_min = float(fallback_context.cot_min)
            cot_default_max = float(fallback_context.cot_max)

        cot_min = cot_default_min if cot_theta_min is None else float(cot_theta_min)
        cot_max = cot_default_max if cot_theta_max is None else float(cot_theta_max)
        if cot_min > cot_max:
            cot_min, cot_max = cot_max, cot_min

        cot_vals = _build_slider_values(
            value_min=cot_min,
            value_max=cot_max,
            n_points=n_cot,
            step=cot_theta_step,
        )
        cot_labels = [f"{float(cot_theta):.2f}" for cot_theta in cot_vals]
        util_volume = np.full(
            (len(cot_vals), len(plot_domain.n_vals), len(plot_domain.m_vals)),
            np.nan,
            dtype=float,
        )

        for i_n, row in enumerate(context_grid):
            for i_m, context in enumerate(row):
                if context is None:
                    continue
                for i_cot, cot_theta in enumerate(cot_vals):
                    cot_f = float(cot_theta)
                    V_Rd_s = self._find_V_Rd_s(cot_f, context)
                    V_Rd_max = self._find_V_Rd_max(cot_f, context)
                    V_Rd = min(V_Rd_s, V_Rd_max)
                    util_volume[i_cot, i_n, i_m] = context.V_Ed / V_Rd if V_Rd > 0.0 else float("inf")

        display_volume = np.repeat(
            np.repeat(util_volume, display_oversample, axis=1),
            display_oversample,
            axis=2,
        )

        finite_vals = util_volume[np.isfinite(util_volume)]
        zmax = max(1.5, float(np.nanmax(finite_vals))) if finite_vals.size else 1.5
        colorscale = _utilization_colorscale(zmin=0.0, zmax=zmax)

        first_slice = display_volume[0, :, :]
        fig = go.Figure()
        fig.add_trace(
            go.Heatmap(
                x=display_m_vals,
                y=display_n_vals,
                z=first_slice,
                colorscale=colorscale,
                zmin=0.0,
                zmax=zmax,
                colorbar=dict(title="Utilization"),
                hovertemplate=(
                    "My_Ed: %{x:.2f} kN*m<br>"
                    "N_Ed: %{y:.1f} kN<br>"
                    "Utilization: %{z:.3f}<extra></extra>"
                ),
                name="utilization",
            ),
        )
        fig.add_trace(
            go.Contour(
                x=display_m_vals,
                y=display_n_vals,
                z=first_slice,
                contours=dict(start=1.0, end=1.0, size=1.0, coloring="none"),
                line=dict(color="black", width=2),
                showscale=False,
                name="Utilization = 1.0",
                hoverinfo="skip",
            ),
        )
        fig.add_trace(
            go.Scatter(
                x=left_mask[0],
                y=left_mask[1],
                mode="none",
                fill="toself",
                fillcolor="white",
                showlegend=False,
                hoverinfo="skip",
                name="_clip_left",
            ),
        )
        fig.add_trace(
            go.Scatter(
                x=right_mask[0],
                y=right_mask[1],
                mode="none",
                fill="toself",
                fillcolor="white",
                showlegend=False,
                hoverinfo="skip",
                name="_clip_right",
            ),
        )
        fig.add_trace(
            go.Scatter(
                x=plot_domain.m_curve,
                y=plot_domain.n_curve,
                mode="lines",
                name="M-N Capacity",
                line=dict(color="black", width=2),
                hovertemplate=(
                    "M_Rd: %{x:.2f} kN*m<br>"
                    "N_Rd: %{y:.1f} kN<extra></extra>"
                ),
            ),
        )

        for plotted_case in plotted_loadcases:
            capacity = plot_diagram.get_capacity_vector(
                N_Ed=plotted_case.N_Ed,
                M_Ed=plotted_case.M_Ed,
                n_points=n_diagram_points,
                return_details=False,
            )
            is_inside = bool(capacity.is_safe)
            status = "Inside envelope" if is_inside else "Outside envelope"
            fig.add_trace(
                go.Scatter(
                    x=[plotted_case.M_Ed],
                    y=[plotted_case.N_Ed],
                    mode="markers",
                    name=plotted_case.name,
                    marker=dict(
                        color="green" if is_inside else "red",
                        size=8,
                        symbol="circle",
                        line=dict(color="black", width=1),
                    ),
                    hovertemplate=(
                        f"<b>{plotted_case.name}</b><br>"
                        f"My_Ed: {plotted_case.M_Ed:.2f} kN*m<br>"
                        f"N_Ed: {plotted_case.N_Ed:.1f} kN<br>"
                        f"Status: {status}<extra></extra>"
                    ),
                ),
            )

        frames = []
        for i, cot_label in enumerate(cot_labels):
            util_slice = display_volume[i, :, :]
            frames.append(
                {
                    "name": cot_label,
                    "data": [
                        go.Heatmap(
                            x=display_m_vals,
                            y=display_n_vals,
                            z=util_slice,
                            colorscale=colorscale,
                            zmin=0.0,
                            zmax=zmax,
                            showscale=False,
                            hovertemplate=(
                                "My_Ed: %{x:.2f} kN*m<br>"
                                "N_Ed: %{y:.1f} kN<br>"
                                "Utilization: %{z:.3f}<extra></extra>"
                            ),
                            name="utilization",
                        ),
                        go.Contour(
                            x=display_m_vals,
                            y=display_n_vals,
                            z=util_slice,
                            contours=dict(start=1.0, end=1.0, size=1.0, coloring="none"),
                            line=dict(color="black", width=2),
                            showscale=False,
                            name="Utilization = 1.0",
                            hoverinfo="skip",
                        ),
                    ],
                },
            )
        fig.frames = frames

        layout_kwargs: dict[str, Any] = dict(
            title=title or "Circular Axial-Moment Utilization (cot(theta) slider)",
            xaxis_title="Moment My_Ed (kN*m)",
            yaxis_title="Axial force N_Ed (kN)",
            template="plotly_white",
            width=width,
            height=height,
        )
        if len(cot_vals) > 1:
            steps = [
                {
                    "label": cot_label,
                    "method": "animate",
                    "args": [
                        [cot_label],
                        {
                            "frame": {"duration": 0, "redraw": True},
                            "mode": "immediate",
                            "transition": {"duration": 0},
                        },
                    ],
                }
                for cot_label in cot_labels
            ]
            layout_kwargs.update(
                _build_slider_animation_controls(
                    steps=steps,
                    currentvalue_prefix="cot(theta): ",
                )
            )
        fig.update_layout(**layout_kwargs)

        _show_or_save(fig, save_path=save_path, show=show)
        return fig
