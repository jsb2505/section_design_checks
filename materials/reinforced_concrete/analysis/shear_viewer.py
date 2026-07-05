"""
Shear visualization helpers for EC2 shear checks.

Provides plotting routines for comparative studies of:
- cot(theta) sweeps
- shear link angle sweeps
- cot(theta) vs link angle heatmaps
- axial force vs cot(theta) heatmaps
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from math import radians, sin
from typing import Any, Dict, Optional, Sequence, Tuple, TYPE_CHECKING, Union

import numpy as np

from materials.core.units import ForceUnit, to_kn
from materials.reinforced_concrete.code_checks.ec2_2004.shear_check import ShearLoadCase
from materials.reinforced_concrete.code_checks.ec2_2004.shear_utils import (
    calculate_tension_shift,
    find_alpha_cw,
    find_nu_1_factor,
    find_nu_1_factor_note_2,
)
from materials.utils.helpers import cot

if TYPE_CHECKING:
    from materials.reinforced_concrete.code_checks.ec2_2004.shear_check import ShearCheck


@dataclass(frozen=True)
class _StudyContext:
    """Load-case dependent values reused across sweeps."""

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


@dataclass(frozen=True)
class _CotThetaStudySeries:
    """Computed series used by cot(theta)-based plotting methods."""

    context: _StudyContext
    cot_vals: np.ndarray
    V_Rd_s_vals: list[float]
    V_Rd_max_theta_vals: list[float]
    util_vals: list[float]
    M_add_vals: list[float]
    V_Rd_s_design: float
    V_Rd_max_design: float
    cot_intersection: Optional[float]


@dataclass(frozen=True)
class _LinkAngleStudySeries:
    """Computed series used by link-angle plotting methods."""

    context: _StudyContext
    cot_theta: float
    angle_vals: np.ndarray
    V_Rd_s_vals: list[float]
    V_Rd_max_vals: list[float]
    util_vals: list[float]
    M_add_vals: list[float]


def _as_load_case(load_case: Union[ShearLoadCase, Dict[str, Any]]) -> ShearLoadCase:
    """Normalize supported load-case inputs to ShearLoadCase."""
    if isinstance(load_case, ShearLoadCase):
        return load_case
    if isinstance(load_case, dict):
        return ShearLoadCase(
            V_Ed=float(load_case["V_Ed"]),
            M_Ed=float(load_case.get("M_Ed", 0.0)),
            N_Ed=float(load_case.get("N_Ed", 0.0)),
        )
    raise TypeError("load_case must be a ShearLoadCase or dict with V_Ed/M_Ed/N_Ed keys.")


def _show_or_save(fig: Any, *, save_path: Optional[Union[str, Path]], show: bool) -> None:
    """Apply standard save/show behaviour used by viewer methods."""
    if save_path:
        fig.write_html(str(save_path))
    if show:
        fig.show()


def _utilization_colorscale(*, zmin: float, zmax: float) -> list[list[Union[float, str]]]:
    """Return a colorscale with utilization 1.0 fixed at white."""
    if np.isclose(zmax, zmin):
        white_anchor = 0.5
    else:
        white_anchor = float(np.clip((1.0 - zmin) / (zmax - zmin), 0.0, 1.0))
    return [
        [0.0, "#1a9850"],
        [white_anchor, "#ffffff"],
        [1.0, "#d73027"],
    ]


class ShearViewer:
    """Plotting utilities for ``ShearCheck`` comparative studies."""

    def __init__(self, check: "ShearCheck") -> None:
        self.check = check

    def _require_shear_reinforcement(self) -> None:
        if self.check.shear_reinforcement is None:
            raise ValueError("Shear reinforcement is required for shear study plots.")

    def _build_context(
        self,
        *,
        load_case: ShearLoadCase,
        use_uncracked_V_Rd_c: bool = False,
        ignore_compression_steel: bool = False,
    ) -> _StudyContext:
        """Compute shared parameters for a load case once."""
        V_Ed = abs(float(load_case.V_Ed))
        M_Ed = float(load_case.M_Ed)
        N_Ed = float(load_case.N_Ed)

        if abs(M_Ed) > 1e-6:
            diagram = self.check._get_diagram(ignore_compression_steel)
            eps_top, eps_bottom = diagram.find_strains_for_MN(M_Ed, N_Ed)
            # Detect if the load case is outside the interaction envelope.
            # When strict=True fails, the strains from strict=False are
            # "nearest feasible" projections — not true equilibrium — so
            # z_mech from those strains can be misleading.  Force the
            # virtual lever arm in that case.
            try:
                diagram.find_strains_for_MN(M_Ed, N_Ed, strict=True)
                force_virtual = False
            except ValueError:
                force_virtual = True
        else:
            eps_top, eps_bottom = None, None
            force_virtual = False

        d = self.check.find_effective_depth(
            M_Ed,
            N_Ed,
            eps_top=eps_top,
            eps_bottom=eps_bottom,
            ignore_compression_steel=ignore_compression_steel,
        )
        sigma_cp = self.check._find_sigma_cp(N_Ed)
        rho_l = self.check._find_rho_l(
            M_Ed,
            N_Ed,
            d,
            eps_top=eps_top,
            eps_bottom=eps_bottom,
            ignore_compression_steel=ignore_compression_steel,
        )
        V_Rd_c_cracked = self.check.find_V_Rd_c(d, rho_l, sigma_cp)
        V_Rd_c_uncracked = self.check.find_V_Rd_c_uncracked(sigma_cp=sigma_cp)
        V_Rd_c = V_Rd_c_uncracked if use_uncracked_V_Rd_c else V_Rd_c_cracked

        z, _ = self.check.find_lever_arm(
            M_Ed=M_Ed,
            N_Ed=N_Ed,
            d=d,
            eps_top=eps_top,
            eps_bottom=eps_bottom,
            ignore_compression_steel=ignore_compression_steel,
            force_virtual=force_virtual,
        )
        cot_min, cot_max = self.check._find_cot_theta_limits(sigma_cp=sigma_cp, z=z, V_Ed=V_Ed)

        return _StudyContext(
            V_Ed=V_Ed,
            M_Ed=M_Ed,
            N_Ed=N_Ed,
            d=d,
            z=z,
            sigma_cp=sigma_cp,
            rho_l=rho_l,
            V_Rd_c=V_Rd_c,
            V_Rd_c_cracked=V_Rd_c_cracked,
            V_Rd_c_uncracked=V_Rd_c_uncracked,
            cot_min=cot_min,
            cot_max=cot_max,
        )

    def _find_angle_sweep_capacity(
        self,
        *,
        angle_deg: float,
        cot_theta: float,
        context: _StudyContext,
        use_note_2: bool,
    ) -> Tuple[float, float]:
        """
        Return (V_Rd_s, V_Rd_max) for a custom link angle at fixed cot(theta).

        Uses the same formulas as ShearCheck but with an overridden link angle.
        """
        reinforcement = self.check.shear_reinforcement
        assert reinforcement is not None

        alpha_rad = radians(angle_deg)
        cot_alpha = cot(alpha_rad)
        sin_alpha = sin(alpha_rad)

        f_ywd = 0.8 * reinforcement.f_yk if use_note_2 else self.check.f_ywd_design
        V_Rd_s_N = (
            reinforcement.area_per_unit_length
            * context.z
            * f_ywd
            * (cot_theta + cot_alpha)
            * sin_alpha
        )
        V_Rd_s = to_kn(V_Rd_s_N, ForceUnit.N)

        alpha_cw = find_alpha_cw(
            self.check.f_cd_design,
            context.sigma_cp,
            use_sigma_cp_for_alpha_cw=self.check.use_sigma_cp_for_alpha_cw,
        )
        if use_note_2:
            nu_1 = find_nu_1_factor_note_2(self.check.concrete.f_ck, angle_deg)
        else:
            nu_1 = find_nu_1_factor(self.check.concrete.f_ck, angle_deg)

        V_Rd_max_N = (
            alpha_cw
            * self.check.breadth
            * context.z
            * nu_1
            * self.check.f_cd_design
            * (cot_theta + cot_alpha)
            / (1.0 + cot_theta**2)
        )
        V_Rd_max = to_kn(V_Rd_max_N, ForceUnit.N)
        return V_Rd_s, V_Rd_max

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
        use_note_2: bool,
    ) -> _CotThetaStudySeries:
        """Compute reusable cot(theta) sweep values for plotting."""
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

        V_Rd_max_design = self.check.find_V_Rd_max(
            context.cot_min,
            context.z,
            context.sigma_cp,
            use_note_2=use_note_2,
        )
        V_Rd_s_design = self.check.find_V_Rd_s(
            context.cot_max,
            context.z,
            use_note_2=use_note_2,
        )

        for cot_theta in cot_vals:
            cot_theta_f = float(cot_theta)
            V_Rd_s = self.check.find_V_Rd_s(cot_theta_f, context.z, use_note_2=use_note_2)
            V_Rd_max_theta = self.check.find_V_Rd_max(
                cot_theta_f,
                context.z,
                context.sigma_cp,
                use_note_2=use_note_2,
            )

            V_Rd = min(V_Rd_s, V_Rd_max_design)
            util = context.V_Ed / V_Rd if V_Rd > 0.0 else float("inf")

            shift = calculate_tension_shift(
                M_Ed=context.M_Ed,
                V_Ed=context.V_Ed,
                z=context.z,
                d=context.d,
                b_w=self.check.breadth,
                f_cd=self.check.f_cd_design,
                f_ck=self.check.concrete.f_ck,
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
        return _CotThetaStudySeries(
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

    def plot_cot_theta_study(
        self,
        *,
        load_case: Union[ShearLoadCase, Dict[str, Any]],
        n_points: int = 60,
        cot_theta_min: Optional[float] = None,
        cot_theta_max: Optional[float] = None,
        use_uncracked_V_Rd_c: bool = False,
        use_note_2: bool = False,
        save_path: Optional[Union[str, Path]] = None,
        show: bool = True,
        title: Optional[str] = None,
        width: int = 1000,
        height: int = 560,
    ) -> Any:
        """
        Plot shear-capacity components over a cot(theta) sweep.

        The figure contains demand and capacity references (`V_Ed`, `V_Rd,c`),
        variable capacities (`V_Rd,s`, `V_Rd,max`) and fixed design reference lines
        at the governing code limits for cot(theta).

        Args:
            load_case: Shear demand definition as either ``ShearLoadCase`` or a
                ``dict`` with keys ``V_Ed`` and optional ``M_Ed``/``N_Ed`` (kN, kN·m).
            n_points: Number of cot(theta) samples in the sweep.
            cot_theta_min: Optional lower bound for cot(theta). If ``None``,
                the EC2-based minimum from the current check context is used.
            cot_theta_max: Optional upper bound for cot(theta). If ``None``,
                the EC2-based maximum from the current check context is used.
            use_uncracked_V_Rd_c: If ``True``, use uncracked concrete shear capacity
                ``V_Rd,c,uncracked`` as the concrete reference.
            use_note_2: If ``True``, apply EC2 6.2.3(3) Note 2 variants for
                ``nu_1`` and reinforcement yield stress assumptions.
            save_path: Optional file path for ``fig.write_html(...)`` output.
            show: If ``True``, call ``fig.show()`` before returning.
            title: Optional custom plot title.
            width: Figure width in pixels.
            height: Figure height in pixels.

        Returns:
            plotly.graph_objects.Figure: Plotly figure instance for further
            customization or export.

        Raises:
            ValueError: If the ``ShearCheck`` has no shear reinforcement.
            TypeError: If ``load_case`` is not a ``ShearLoadCase`` or compatible dict.
            ImportError: If Plotly is not installed.
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
            use_note_2=use_note_2,
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
                np.interp(series.cot_intersection, cot_vals, np.asarray(series.V_Rd_s_vals, dtype=float)),
            )
            fig.add_trace(
                go.Scatter(
                    x=[series.cot_intersection, series.cot_intersection, float(cot_vals[0])],
                    y=[0.0, y_intersection, y_intersection],
                    mode="lines",
                    name="V_Rd,s = V_Rd,max",
                    line=dict(color="#ff0000", dash="dash"),
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

        fig.update_xaxes(title_text="cot(theta)")
        fig.update_yaxes(title_text="Capacity (kN)")
        fig.update_layout(
            title=title or "Shear Capacity Study vs cot(theta)",
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
        use_note_2: bool = False,
        save_path: Optional[Union[str, Path]] = None,
        show: bool = True,
        title: Optional[str] = None,
        width: int = 1000,
        height: int = 560,
    ) -> Any:
        """
        Plot utilization and tension-shift add-on versus cot(theta).

        This plot isolates serviceability/design effect indicators rather than
        capacity components:
        - utilization ratio ``V_Ed / V_Rd``
        - additional moment from tension shift ``M_add``

        Args:
            load_case: Shear demand definition as either ``ShearLoadCase`` or a
                ``dict`` with keys ``V_Ed`` and optional ``M_Ed``/``N_Ed`` (kN, kN·m).
            n_points: Number of cot(theta) samples in the sweep.
            cot_theta_min: Optional lower bound for cot(theta). If ``None``,
                the EC2-based minimum from the current check context is used.
            cot_theta_max: Optional upper bound for cot(theta). If ``None``,
                the EC2-based maximum from the current check context is used.
            use_uncracked_V_Rd_c: If ``True``, use uncracked concrete shear capacity
                ``V_Rd,c,uncracked`` when forming utilization.
            use_note_2: If ``True``, apply EC2 6.2.3(3) Note 2 variants for
                ``nu_1`` and reinforcement yield stress assumptions.
            save_path: Optional file path for ``fig.write_html(...)`` output.
            show: If ``True``, call ``fig.show()`` before returning.
            title: Optional custom plot title.
            width: Figure width in pixels.
            height: Figure height in pixels.

        Returns:
            plotly.graph_objects.Figure: Plotly figure instance for further
            customization or export.

        Raises:
            ValueError: If the ``ShearCheck`` has no shear reinforcement.
            TypeError: If ``load_case`` is not a ``ShearLoadCase`` or compatible dict.
            ImportError: If Plotly is not installed.
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
            use_note_2=use_note_2,
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
                hovertemplate="cot(theta): %{x:.3f}<br>M_add: %{y:.2f} kN·m<extra></extra>",
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
                    series.cot_vals,
                    np.asarray(series.M_add_vals, dtype=float),
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
                        "M_add: %{customdata[1]:.2f} kN·m<extra></extra>"
                    ),
                ),
                secondary_y=False,
            )

        fig.update_xaxes(title_text="cot(theta)")
        fig.update_yaxes(title_text="Utilization (-)", secondary_y=False)
        fig.update_yaxes(title_text="M_add (kN·m)", secondary_y=True)
        fig.update_layout(
            title=title or "Tension-Shift Study vs cot(theta)",
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

    def _compute_link_angle_study_series(
        self,
        *,
        load_case: Union[ShearLoadCase, Dict[str, Any]],
        cot_theta: Optional[float],
        angle_min: float,
        angle_max: float,
        n_points: int,
        use_uncracked_V_Rd_c: bool,
        use_note_2: bool,
    ) -> _LinkAngleStudySeries:
        """Compute reusable link-angle sweep values for plotting."""
        case = _as_load_case(load_case)
        context = self._build_context(load_case=case, use_uncracked_V_Rd_c=use_uncracked_V_Rd_c)

        if cot_theta is None:
            cot_theta = self.check._find_cot_theta_for_V_Ed(
                V_Ed=context.V_Ed,
                z=context.z,
                sigma_cp=context.sigma_cp,
                cot_min=context.cot_min,
                cot_max=context.cot_max,
                use_note_2=use_note_2,
                use_v_rd_s_for_cot_theta=False,
            )
        cot_theta_val = float(cot_theta)

        a_min = min(float(angle_min), float(angle_max))
        a_max = max(float(angle_min), float(angle_max))
        angle_vals = np.linspace(a_min, a_max, max(2, int(n_points)))

        V_Rd_s_vals: list[float] = []
        V_Rd_max_vals: list[float] = []
        util_vals: list[float] = []
        M_add_vals: list[float] = []

        reinforcement = self.check.shear_reinforcement
        assert reinforcement is not None

        for angle in angle_vals:
            angle_f = float(angle)
            V_Rd_s, V_Rd_max = self._find_angle_sweep_capacity(
                angle_deg=angle_f,
                cot_theta=cot_theta_val,
                context=context,
                use_note_2=use_note_2,
            )

            V_Rd = min(V_Rd_s, V_Rd_max)
            util = context.V_Ed / V_Rd if V_Rd > 0.0 else float("inf")

            angle_rebar = reinforcement.model_copy(update={"angle": angle_f})
            shift = calculate_tension_shift(
                M_Ed=context.M_Ed,
                V_Ed=context.V_Ed,
                z=context.z,
                d=context.d,
                shear_reinforcement=angle_rebar,
                cot_theta_override=cot_theta_val,
            )

            V_Rd_s_vals.append(V_Rd_s)
            V_Rd_max_vals.append(V_Rd_max)
            util_vals.append(util)
            M_add_vals.append(shift.M_add)

        return _LinkAngleStudySeries(
            context=context,
            cot_theta=cot_theta_val,
            angle_vals=angle_vals,
            V_Rd_s_vals=V_Rd_s_vals,
            V_Rd_max_vals=V_Rd_max_vals,
            util_vals=util_vals,
            M_add_vals=M_add_vals,
        )

    def plot_link_angle_study(
        self,
        *,
        load_case: Union[ShearLoadCase, Dict[str, Any]],
        cot_theta: Optional[float] = None,
        angle_min: float = 45.0,
        angle_max: float = 90.0,
        n_points: int = 46,
        use_uncracked_V_Rd_c: bool = False,
        use_note_2: bool = False,
        save_path: Optional[Union[str, Path]] = None,
        show: bool = True,
        title: Optional[str] = None,
        width: int = 1000,
        height: int = 560,
    ) -> Any:
        """
        Plot shear-capacity components over a link-angle sweep.

        The sweep is run at a fixed cot(theta). If ``cot_theta`` is not provided,
        it is back-calculated from the load case using the same internal logic as
        the shear check.

        Args:
            load_case: Shear demand definition as either ``ShearLoadCase`` or a
                ``dict`` with keys ``V_Ed`` and optional ``M_Ed``/``N_Ed`` (kN, kN·m).
            cot_theta: Fixed cot(theta) used for the angle sweep. If ``None``,
                it is derived from ``V_Ed`` and section context.
            angle_min: Minimum link angle (degrees).
            angle_max: Maximum link angle (degrees).
            n_points: Number of sampled link angles between ``angle_min`` and
                ``angle_max``.
            use_uncracked_V_Rd_c: If ``True``, use uncracked concrete shear capacity
                ``V_Rd,c,uncracked`` as the concrete reference.
            use_note_2: If ``True``, apply EC2 6.2.3(3) Note 2 variants for
                ``nu_1`` and reinforcement yield stress assumptions.
            save_path: Optional file path for ``fig.write_html(...)`` output.
            show: If ``True``, call ``fig.show()`` before returning.
            title: Optional custom plot title.
            width: Figure width in pixels.
            height: Figure height in pixels.

        Returns:
            plotly.graph_objects.Figure: Plotly figure instance for further
            customization or export.

        Raises:
            ValueError: If the ``ShearCheck`` has no shear reinforcement.
            TypeError: If ``load_case`` is not a ``ShearLoadCase`` or compatible dict.
            ImportError: If Plotly is not installed.
        """
        self._require_shear_reinforcement()

        try:
            import plotly.graph_objects as go
        except ImportError as e:
            raise ImportError("Plotly is required for plotting. Install with: pip install plotly") from e

        series = self._compute_link_angle_study_series(
            load_case=load_case,
            cot_theta=cot_theta,
            angle_min=angle_min,
            angle_max=angle_max,
            n_points=n_points,
            use_uncracked_V_Rd_c=use_uncracked_V_Rd_c,
            use_note_2=use_note_2,
        )
        context = series.context

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=series.angle_vals,
                y=[context.V_Ed] * len(series.angle_vals),
                mode="lines",
                name="V_Ed",
                line=dict(color="black", dash="dash"),
                hovertemplate="alpha: %{x:.1f}°<br>V_Ed: %{y:.1f} kN<extra></extra>",
            ),
        )
        fig.add_trace(
            go.Scatter(
                x=series.angle_vals,
                y=[context.V_Rd_c] * len(series.angle_vals),
                mode="lines",
                name="V_Rd,c",
                line=dict(color="#8c564b", dash="dot"),
                hovertemplate="alpha: %{x:.1f}°<br>V_Rd,c: %{y:.1f} kN<extra></extra>",
            ),
        )
        fig.add_trace(
            go.Scatter(
                x=series.angle_vals,
                y=series.V_Rd_s_vals,
                mode="lines",
                name="V_Rd,s(alpha)",
                line=dict(color="#1f77b4"),
                hovertemplate="alpha: %{x:.1f}°<br>V_Rd,s: %{y:.1f} kN<extra></extra>",
            ),
        )
        fig.add_trace(
            go.Scatter(
                x=series.angle_vals,
                y=series.V_Rd_max_vals,
                mode="lines",
                name="V_Rd,max(alpha)",
                line=dict(color="#ff7f0e"),
                hovertemplate="alpha: %{x:.1f}°<br>V_Rd,max: %{y:.1f} kN<extra></extra>",
            ),
        )

        fig.update_xaxes(title_text="Link angle alpha (degrees)")
        fig.update_yaxes(title_text="Capacity (kN)")
        fig.update_layout(
            title=title or f"Shear Capacity Study vs Link Angle (cot(theta)={series.cot_theta:.2f})",
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

    def plot_link_angle_moment_shift_study(
        self,
        *,
        load_case: Union[ShearLoadCase, Dict[str, Any]],
        cot_theta: Optional[float] = None,
        angle_min: float = 45.0,
        angle_max: float = 90.0,
        n_points: int = 46,
        use_uncracked_V_Rd_c: bool = False,
        use_note_2: bool = False,
        save_path: Optional[Union[str, Path]] = None,
        show: bool = True,
        title: Optional[str] = None,
        width: int = 1000,
        height: int = 560,
    ) -> Any:
        """
        Plot utilization and tension-shift add-on versus link angle.

        The sweep is run at a fixed cot(theta). If ``cot_theta`` is not provided,
        it is back-calculated from the load case using the same internal logic as
        the shear check.

        Args:
            load_case: Shear demand definition as either ``ShearLoadCase`` or a
                ``dict`` with keys ``V_Ed`` and optional ``M_Ed``/``N_Ed`` (kN, kN·m).
            cot_theta: Fixed cot(theta) used for the angle sweep. If ``None``,
                it is derived from ``V_Ed`` and section context.
            angle_min: Minimum link angle (degrees).
            angle_max: Maximum link angle (degrees).
            n_points: Number of sampled link angles between ``angle_min`` and
                ``angle_max``.
            use_uncracked_V_Rd_c: If ``True``, use uncracked concrete shear capacity
                ``V_Rd,c,uncracked`` when forming utilization.
            use_note_2: If ``True``, apply EC2 6.2.3(3) Note 2 variants for
                ``nu_1`` and reinforcement yield stress assumptions.
            save_path: Optional file path for ``fig.write_html(...)`` output.
            show: If ``True``, call ``fig.show()`` before returning.
            title: Optional custom plot title.
            width: Figure width in pixels.
            height: Figure height in pixels.

        Returns:
            plotly.graph_objects.Figure: Plotly figure instance for further
            customization or export.

        Raises:
            ValueError: If the ``ShearCheck`` has no shear reinforcement.
            TypeError: If ``load_case`` is not a ``ShearLoadCase`` or compatible dict.
            ImportError: If Plotly is not installed.
        """
        self._require_shear_reinforcement()

        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError as e:
            raise ImportError("Plotly is required for plotting. Install with: pip install plotly") from e

        series = self._compute_link_angle_study_series(
            load_case=load_case,
            cot_theta=cot_theta,
            angle_min=angle_min,
            angle_max=angle_max,
            n_points=n_points,
            use_uncracked_V_Rd_c=use_uncracked_V_Rd_c,
            use_note_2=use_note_2,
        )

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Scatter(
                x=series.angle_vals,
                y=series.util_vals,
                mode="lines",
                name="Utilization",
                line=dict(color="#d62728"),
                hovertemplate="alpha: %{x:.1f}°<br>Utilization: %{y:.3f}<extra></extra>",
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=series.angle_vals,
                y=series.M_add_vals,
                mode="lines",
                name="M_add (tension shift)",
                line=dict(color="#9467bd"),
                hovertemplate="alpha: %{x:.1f}°<br>M_add: %{y:.2f} kN·m<extra></extra>",
            ),
            secondary_y=True,
        )
        fig.add_trace(
            go.Scatter(
                x=series.angle_vals,
                y=[1.0] * len(series.angle_vals),
                mode="lines",
                name="Utilization = 1.0",
                line=dict(color="black", dash="dot"),
                hovertemplate="Utilization limit<extra></extra>",
            ),
            secondary_y=False,
        )

        fig.update_xaxes(title_text="Link angle alpha (degrees)")
        fig.update_yaxes(title_text="Utilization (-)", secondary_y=False)
        fig.update_yaxes(title_text="M_add (kN·m)", secondary_y=True)
        fig.update_layout(
            title=title or f"Tension-Shift Study vs Link Angle (cot(theta)={series.cot_theta:.2f})",
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

    def plot_cot_theta_link_angle_heatmap(
        self,
        *,
        load_case: Union[ShearLoadCase, Dict[str, Any]],
        cot_theta_min: Optional[float] = None,
        cot_theta_max: Optional[float] = None,
        angle_min: float = 45.0,
        angle_max: float = 90.0,
        n_cot: int = 40,
        n_angles: int = 40,
        metric: str = "utilization",
        use_uncracked_V_Rd_c: bool = False,
        use_note_2: bool = False,
        save_path: Optional[Union[str, Path]] = None,
        show: bool = True,
        title: Optional[str] = None,
        width: int = 980,
        height: int = 760,
    ) -> Any:
        """
        Plot a cot(theta)-vs-link-angle heatmap for shear response metrics.

        Args:
            load_case: Shear demand definition as either ``ShearLoadCase`` or a
                ``dict`` with keys ``V_Ed`` and optional ``M_Ed``/``N_Ed`` (kN, kN·m).
            cot_theta_min: Optional lower bound for cot(theta). If ``None``,
                the EC2-based minimum from the current check context is used.
            cot_theta_max: Optional upper bound for cot(theta). If ``None``,
                the EC2-based maximum from the current check context is used.
            angle_min: Minimum link angle (degrees).
            angle_max: Maximum link angle (degrees).
            n_cot: Number of cot(theta) samples.
            n_angles: Number of link-angle samples.
            metric: Response quantity on the color axis. Supported values are:
                ``"utilization"``, ``"capacity"``, ``"v_rd_s"``, and ``"v_rd_max"``.
            use_uncracked_V_Rd_c: If ``True``, use uncracked concrete shear capacity
                ``V_Rd,c,uncracked`` when forming governing capacity/utilization.
            use_note_2: If ``True``, apply EC2 6.2.3(3) Note 2 variants for
                ``nu_1`` and reinforcement yield stress assumptions.
            save_path: Optional file path for ``fig.write_html(...)`` output.
            show: If ``True``, call ``fig.show()`` before returning.
            title: Optional custom plot title.
            width: Figure width in pixels.
            height: Figure height in pixels.

        Returns:
            plotly.graph_objects.Figure: Plotly heatmap figure.

        Raises:
            ValueError: If no shear reinforcement is defined, or if ``metric`` is invalid.
            TypeError: If ``load_case`` is not a ``ShearLoadCase`` or compatible dict.
            ImportError: If Plotly is not installed.
        """
        self._require_shear_reinforcement()

        try:
            import plotly.graph_objects as go
        except ImportError as e:
            raise ImportError("Plotly is required for plotting. Install with: pip install plotly") from e

        case = _as_load_case(load_case)
        context = self._build_context(load_case=case, use_uncracked_V_Rd_c=use_uncracked_V_Rd_c)

        cot_min = context.cot_min if cot_theta_min is None else float(cot_theta_min)
        cot_max = context.cot_max if cot_theta_max is None else float(cot_theta_max)
        if cot_min > cot_max:
            cot_min, cot_max = cot_max, cot_min

        cot_vals = np.linspace(cot_min, cot_max, max(2, int(n_cot)))
        angle_vals = np.linspace(min(angle_min, angle_max), max(angle_min, angle_max), max(2, int(n_angles)))

        Z = np.zeros((len(angle_vals), len(cot_vals)))

        metric_key = metric.strip().lower()
        valid_metrics = {"utilization", "capacity", "v_rd_s", "v_rd_max"}
        if metric_key not in valid_metrics:
            raise ValueError(f"metric must be one of {sorted(valid_metrics)}.")

        for i, angle in enumerate(angle_vals):
            angle_f = float(angle)
            for j, cot_theta in enumerate(cot_vals):
                cot_f = float(cot_theta)
                V_Rd_s, V_Rd_max = self._find_angle_sweep_capacity(
                    angle_deg=angle_f,
                    cot_theta=cot_f,
                    context=context,
                    use_note_2=use_note_2,
                )

                V_Rd = min(V_Rd_s, V_Rd_max)

                if metric_key == "utilization":
                    value = context.V_Ed / V_Rd if V_Rd > 0.0 else float("inf")
                elif metric_key == "capacity":
                    value = V_Rd
                elif metric_key == "v_rd_s":
                    value = V_Rd_s
                else:
                    value = V_Rd_max
                Z[i, j] = value

        if metric_key == "utilization":
            colorbar_title = "Utilization (-)"
            zmin = 0.0
            zmax = max(1.5, float(np.nanmax(Z)))
            colorscale = _utilization_colorscale(zmin=zmin, zmax=zmax)
            contour_trace = go.Contour(
                x=cot_vals,
                y=angle_vals,
                z=Z,
                contours=dict(start=1.0, end=1.0, size=1.0, coloring="none"),
                line=dict(color="black", width=2),
                showscale=False,
                name="Utilization = 1.0",
                hoverinfo="skip",
            )
        else:
            colorbar_title = "kN" if metric_key != "utilization" else "Utilization (-)"
            colorscale = "Viridis"
            zmin = None
            zmax = None
            contour_trace = None

        fig = go.Figure()
        fig.add_trace(
            go.Heatmap(
                x=cot_vals,
                y=angle_vals,
                z=Z,
                colorscale=colorscale,
                zmin=zmin,
                zmax=zmax,
                colorbar=dict(title=colorbar_title),
                hovertemplate=(
                    "cot(theta): %{x:.3f}<br>"
                    "alpha: %{y:.1f}°<br>"
                    f"{metric_key}: "
                    "%{z:.3f}<extra></extra>"
                ),
                name=metric_key,
            )
        )
        if contour_trace is not None:
            fig.add_trace(contour_trace)

        fig.update_layout(
            title=title or f"Shear Study Heatmap: {metric_key}",
            xaxis_title="cot(theta)",
            yaxis_title="Link angle alpha (degrees)",
            template="plotly_white",
            width=width,
            height=height,
        )

        _show_or_save(fig, save_path=save_path, show=show)
        return fig

    def plot_axial_cot_theta_contour(
        self,
        *,
        load_case: Union[ShearLoadCase, Dict[str, Any]],
        N_min: float,
        N_max: float,
        n_axial: int = 31,
        cot_theta_min: Optional[float] = None,
        cot_theta_max: Optional[float] = None,
        n_cot: int = 40,
        metric: str = "utilization",
        use_uncracked_V_Rd_c: bool = False,
        use_note_2: bool = False,
        save_path: Optional[Union[str, Path]] = None,
        show: bool = True,
        title: Optional[str] = None,
        width: int = 980,
        height: int = 760,
    ) -> Any:
        """
        Plot an axial-force-vs-cot(theta) heatmap for shear response metrics.

        For each axial force level, this method recomputes the section context
        and evaluates the selected metric over the cot(theta) sweep.

        Args:
            load_case: Base shear load case (``V_Ed`` and ``M_Ed`` are kept fixed).
                Can be ``ShearLoadCase`` or a ``dict`` with keys ``V_Ed`` and
                optional ``M_Ed``/``N_Ed``.
            N_min: Minimum axial force in kN.
            N_max: Maximum axial force in kN.
            n_axial: Number of axial-force samples.
            cot_theta_min: Optional lower bound for cot(theta). If ``None``,
                the EC2-based minimum from the current check context is used.
            cot_theta_max: Optional upper bound for cot(theta). If ``None``,
                the EC2-based maximum from the current check context is used.
            n_cot: Number of cot(theta) samples.
            metric: Response quantity on the color axis. Supported values are:
                ``"utilization"``, ``"capacity"``, ``"v_rd_s"``, and ``"v_rd_max"``.
            use_uncracked_V_Rd_c: If ``True``, use uncracked concrete shear capacity
                ``V_Rd,c,uncracked`` when forming governing capacity/utilization.
            use_note_2: If ``True``, apply EC2 6.2.3(3) Note 2 variants for
                ``nu_1`` and reinforcement yield stress assumptions.
            save_path: Optional file path for ``fig.write_html(...)`` output.
            show: If ``True``, call ``fig.show()`` before returning.
            title: Optional custom plot title.
            width: Figure width in pixels.
            height: Figure height in pixels.

        Returns:
            plotly.graph_objects.Figure: Plotly heatmap figure.

        Raises:
            ValueError: If no shear reinforcement is defined, or if ``metric`` is invalid.
            TypeError: If ``load_case`` is not a ``ShearLoadCase`` or compatible dict.
            ImportError: If Plotly is not installed.
        """
        self._require_shear_reinforcement()

        try:
            import plotly.graph_objects as go
        except ImportError as e:
            raise ImportError("Plotly is required for plotting. Install with: pip install plotly") from e

        case = _as_load_case(load_case)
        base_context = self._build_context(load_case=case, use_uncracked_V_Rd_c=use_uncracked_V_Rd_c)

        cot_min = base_context.cot_min if cot_theta_min is None else float(cot_theta_min)
        cot_max = base_context.cot_max if cot_theta_max is None else float(cot_theta_max)
        if cot_min > cot_max:
            cot_min, cot_max = cot_max, cot_min

        cot_vals = np.linspace(cot_min, cot_max, max(2, int(n_cot)))
        n_vals = np.linspace(float(N_min), float(N_max), max(2, int(n_axial)))

        Z = np.zeros((len(n_vals), len(cot_vals)))

        metric_key = metric.strip().lower()
        valid_metrics = {"utilization", "capacity", "v_rd_s", "v_rd_max"}
        if metric_key not in valid_metrics:
            raise ValueError(f"metric must be one of {sorted(valid_metrics)}.")

        for i, n_ed in enumerate(n_vals):
            axial_case = ShearLoadCase(V_Ed=case.V_Ed, M_Ed=case.M_Ed, N_Ed=float(n_ed))
            context = self._build_context(load_case=axial_case, use_uncracked_V_Rd_c=use_uncracked_V_Rd_c)

            for j, cot_theta in enumerate(cot_vals):
                cot_f = float(cot_theta)
                V_Rd_s = self.check.find_V_Rd_s(cot_f, context.z, use_note_2=use_note_2)
                V_Rd_max = self.check.find_V_Rd_max(cot_f, context.z, context.sigma_cp, use_note_2=use_note_2)

                V_Rd = min(V_Rd_s, V_Rd_max)

                if metric_key == "utilization":
                    value = context.V_Ed / V_Rd if V_Rd > 0.0 else float("inf")
                elif metric_key == "capacity":
                    value = V_Rd
                elif metric_key == "v_rd_s":
                    value = V_Rd_s
                else:
                    value = V_Rd_max
                Z[i, j] = value

        if metric_key == "utilization":
            colorbar_title = "Utilization (-)"
            zmin = 0.0
            zmax = max(1.5, float(np.nanmax(Z)))
            colorscale = _utilization_colorscale(zmin=zmin, zmax=zmax)
            contour_trace = go.Contour(
                x=cot_vals,
                y=n_vals,
                z=Z,
                contours=dict(start=1.0, end=1.0, size=1.0, coloring="none"),
                line=dict(color="black", width=2),
                showscale=False,
                name="Utilization = 1.0",
                hoverinfo="skip",
            )
        else:
            colorbar_title = "kN"
            colorscale = "Viridis"
            zmin = None
            zmax = None
            contour_trace = None

        fig = go.Figure()
        fig.add_trace(
            go.Heatmap(
                x=cot_vals,
                y=n_vals,
                z=Z,
                colorscale=colorscale,
                zmin=zmin,
                zmax=zmax,
                colorbar=dict(title=colorbar_title),
                hovertemplate=(
                    "cot(theta): %{x:.3f}<br>"
                    "N_Ed: %{y:.1f} kN<br>"
                    f"{metric_key}: "
                    "%{z:.3f}<extra></extra>"
                ),
                name=metric_key,
            )
        )
        if contour_trace is not None:
            fig.add_trace(contour_trace)

        fig.update_layout(
            title=title or f"Axial Force vs cot(theta): {metric_key}",
            xaxis_title="cot(theta)",
            yaxis_title="Axial force N_Ed (kN)",
            template="plotly_white",
            width=width,
            height=height,
        )

        _show_or_save(fig, save_path=save_path, show=show)
        return fig
