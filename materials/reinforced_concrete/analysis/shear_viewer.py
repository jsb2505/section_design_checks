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
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING, Union

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
            eps_top, eps_bottom = self.check._get_diagram(ignore_compression_steel).find_strains_for_MN(M_Ed, N_Ed)
        else:
            eps_top, eps_bottom = None, None

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
        height: int = 820,
    ) -> Any:
        """
        Plot cot(theta) sweep with capacities, utilization, and tension-shift effect.
        """
        self._require_shear_reinforcement()

        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError as e:
            raise ImportError("Plotly is required for plotting. Install with: pip install plotly") from e

        case = _as_load_case(load_case)
        context = self._build_context(load_case=case, use_uncracked_V_Rd_c=use_uncracked_V_Rd_c)

        cot_min = context.cot_min if cot_theta_min is None else float(cot_theta_min)
        cot_max = context.cot_max if cot_theta_max is None else float(cot_theta_max)
        if cot_min > cot_max:
            cot_min, cot_max = cot_max, cot_min

        cot_vals = np.linspace(cot_min, cot_max, max(2, int(n_points)))

        V_Rd_s_vals: list[float] = []
        V_Rd_max_theta_vals: list[float] = []
        V_Rd_vals: list[float] = []
        util_vals: list[float] = []
        M_add_vals: list[float] = []

        V_Rd_max_design = self.check.find_V_Rd_max(
            context.cot_min,
            context.z,
            context.sigma_cp,
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

            if context.V_Ed > context.V_Rd_c:
                V_Rd = min(V_Rd_s, V_Rd_max_design)
            else:
                V_Rd = min(context.V_Rd_c, V_Rd_max_design)
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
            V_Rd_vals.append(V_Rd)
            util_vals.append(util)
            M_add_vals.append(shift.M_add)

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=(
                "Shear Capacities vs cot(theta)",
                "Utilization and Tension-Shift Moment Add-on",
            ),
            specs=[[{}], [{"secondary_y": True}]],
        )

        fig.add_trace(
            go.Scatter(
                x=cot_vals,
                y=[context.V_Ed] * len(cot_vals),
                mode="lines",
                name="V_Ed",
                line=dict(color="black", dash="dash"),
                hovertemplate="cot(theta): %{x:.3f}<br>V_Ed: %{y:.1f} kN<extra></extra>",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=cot_vals,
                y=[context.V_Rd_c] * len(cot_vals),
                mode="lines",
                name="V_Rd,c",
                line=dict(color="#8c564b", dash="dot"),
                hovertemplate="cot(theta): %{x:.3f}<br>V_Rd,c: %{y:.1f} kN<extra></extra>",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=cot_vals,
                y=V_Rd_s_vals,
                mode="lines",
                name="V_Rd,s(cot)",
                line=dict(color="#1f77b4"),
                hovertemplate="cot(theta): %{x:.3f}<br>V_Rd,s: %{y:.1f} kN<extra></extra>",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=cot_vals,
                y=V_Rd_max_theta_vals,
                mode="lines",
                name="V_Rd,max(cot)",
                line=dict(color="#ff7f0e"),
                hovertemplate="cot(theta): %{x:.3f}<br>V_Rd,max(cot): %{y:.1f} kN<extra></extra>",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=cot_vals,
                y=[V_Rd_max_design] * len(cot_vals),
                mode="lines",
                name="V_Rd,max design",
                line=dict(color="#ff7f0e", dash="dash"),
                hovertemplate="cot(theta): %{x:.3f}<br>V_Rd,max design: %{y:.1f} kN<extra></extra>",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=cot_vals,
                y=V_Rd_vals,
                mode="lines",
                name="Governing V_Rd",
                line=dict(color="#2ca02c", width=3),
                hovertemplate="cot(theta): %{x:.3f}<br>V_Rd: %{y:.1f} kN<extra></extra>",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=cot_vals,
                y=util_vals,
                mode="lines",
                name="Utilization",
                line=dict(color="#d62728"),
                hovertemplate="cot(theta): %{x:.3f}<br>Utilization: %{y:.3f}<extra></extra>",
            ),
            row=2,
            col=1,
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=cot_vals,
                y=M_add_vals,
                mode="lines",
                name="M_add (tension shift)",
                line=dict(color="#9467bd"),
                hovertemplate="cot(theta): %{x:.3f}<br>M_add: %{y:.2f} kN·m<extra></extra>",
            ),
            row=2,
            col=1,
            secondary_y=True,
        )
        fig.add_trace(
            go.Scatter(
                x=cot_vals,
                y=[1.0] * len(cot_vals),
                mode="lines",
                name="Utilization = 1.0",
                line=dict(color="black", dash="dot"),
                hovertemplate="Utilization limit<extra></extra>",
            ),
            row=2,
            col=1,
            secondary_y=False,
        )

        fig.update_xaxes(title_text="cot(theta)", row=2, col=1)
        fig.update_yaxes(title_text="Capacity (kN)", row=1, col=1)
        fig.update_yaxes(title_text="Utilization (-)", row=2, col=1, secondary_y=False)
        fig.update_yaxes(title_text="M_add (kN·m)", row=2, col=1, secondary_y=True)
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
        height: int = 820,
    ) -> Any:
        """
        Plot link-angle sweep with capacities, utilization, and tension-shift effect.
        """
        self._require_shear_reinforcement()

        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError as e:
            raise ImportError("Plotly is required for plotting. Install with: pip install plotly") from e

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
        cot_theta = float(cot_theta)

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
                cot_theta=cot_theta,
                context=context,
                use_note_2=use_note_2,
            )
            if context.V_Ed > context.V_Rd_c:
                V_Rd = min(V_Rd_s, V_Rd_max)
            else:
                V_Rd = min(context.V_Rd_c, V_Rd_max)
            util = context.V_Ed / V_Rd if V_Rd > 0.0 else float("inf")

            angle_rebar = reinforcement.model_copy(update={"angle": angle_f})
            shift = calculate_tension_shift(
                M_Ed=context.M_Ed,
                V_Ed=context.V_Ed,
                z=context.z,
                d=context.d,
                shear_reinforcement=angle_rebar,
                cot_theta_override=cot_theta,
            )

            V_Rd_s_vals.append(V_Rd_s)
            V_Rd_max_vals.append(V_Rd_max)
            util_vals.append(util)
            M_add_vals.append(shift.M_add)

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=(
                "Shear Capacities vs Link Angle",
                "Utilization and Tension-Shift Moment Add-on",
            ),
            specs=[[{}], [{"secondary_y": True}]],
        )

        fig.add_trace(
            go.Scatter(
                x=angle_vals,
                y=[context.V_Ed] * len(angle_vals),
                mode="lines",
                name="V_Ed",
                line=dict(color="black", dash="dash"),
                hovertemplate="alpha: %{x:.1f}°<br>V_Ed: %{y:.1f} kN<extra></extra>",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=angle_vals,
                y=[context.V_Rd_c] * len(angle_vals),
                mode="lines",
                name="V_Rd,c",
                line=dict(color="#8c564b", dash="dot"),
                hovertemplate="alpha: %{x:.1f}°<br>V_Rd,c: %{y:.1f} kN<extra></extra>",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=angle_vals,
                y=V_Rd_s_vals,
                mode="lines",
                name="V_Rd,s(alpha)",
                line=dict(color="#1f77b4"),
                hovertemplate="alpha: %{x:.1f}°<br>V_Rd,s: %{y:.1f} kN<extra></extra>",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=angle_vals,
                y=V_Rd_max_vals,
                mode="lines",
                name="V_Rd,max(alpha)",
                line=dict(color="#ff7f0e"),
                hovertemplate="alpha: %{x:.1f}°<br>V_Rd,max: %{y:.1f} kN<extra></extra>",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=angle_vals,
                y=util_vals,
                mode="lines",
                name="Utilization",
                line=dict(color="#d62728"),
                hovertemplate="alpha: %{x:.1f}°<br>Utilization: %{y:.3f}<extra></extra>",
            ),
            row=2,
            col=1,
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=angle_vals,
                y=M_add_vals,
                mode="lines",
                name="M_add (tension shift)",
                line=dict(color="#9467bd"),
                hovertemplate="alpha: %{x:.1f}°<br>M_add: %{y:.2f} kN·m<extra></extra>",
            ),
            row=2,
            col=1,
            secondary_y=True,
        )
        fig.add_trace(
            go.Scatter(
                x=angle_vals,
                y=[1.0] * len(angle_vals),
                mode="lines",
                name="Utilization = 1.0",
                line=dict(color="black", dash="dot"),
                hovertemplate="Utilization limit<extra></extra>",
            ),
            row=2,
            col=1,
            secondary_y=False,
        )

        fig.update_xaxes(title_text="Link angle alpha (degrees)", row=2, col=1)
        fig.update_yaxes(title_text="Capacity (kN)", row=1, col=1)
        fig.update_yaxes(title_text="Utilization (-)", row=2, col=1, secondary_y=False)
        fig.update_yaxes(title_text="M_add (kN·m)", row=2, col=1, secondary_y=True)
        fig.update_layout(
            title=title or f"Shear Capacity Study vs Link Angle (cot(theta)={cot_theta:.2f})",
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
        Heatmap study for cot(theta) and link angle interactions.

        Args:
            metric: ``"utilization"`` (default), ``"capacity"``, ``"V_Rd_s"``,
                or ``"V_Rd_max"``.
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
                if context.V_Ed > context.V_Rd_c:
                    V_Rd = min(V_Rd_s, V_Rd_max)
                else:
                    V_Rd = min(context.V_Rd_c, V_Rd_max)

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
            colorscale = "RdYlGn_r"
            zmin = 0.0
            zmax = max(1.5, float(np.nanmax(Z)))
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
        Heatmap of axial force vs cot(theta) for the current shear reinforcement setup.
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

                if context.V_Ed > context.V_Rd_c:
                    V_Rd = min(V_Rd_s, V_Rd_max)
                else:
                    V_Rd = min(context.V_Rd_c, V_Rd_max)

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
            colorscale = "RdYlGn_r"
            zmin = 0.0
            zmax = max(1.5, float(np.nanmax(Z)))
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
