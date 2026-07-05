"""
Crack width visualization for reinforced concrete sections.

Provides two plot types:
- ``plot_load_cases``: 3D stem plot of crack widths at discrete M-N load cases
- ``plot_contours``: 2D contour map of crack width field with w_k,lim boundary
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from materials.reinforced_concrete.analysis.interaction_diagram import (
    create_interaction_diagram,
)
from materials.reinforced_concrete.code_checks.ec2_2004.cracking_check import (
    CrackingCheck,
    CrackingResult,
)
from materials.reinforced_concrete.constitutive import ConcreteModelType


@dataclass(frozen=True)
class _LoadCaseResult:
    """Intermediate result for a single load case."""
    name: str
    M_Ed: float
    N_Ed: float
    w_k: float
    w_k_limit: float
    is_cracked: bool
    passes: bool


def _compute_load_case_result(
    check: CrackingCheck,
    M_Ed: float,
    N_Ed: float,
    name: str,
) -> _LoadCaseResult:
    """Run the cracking calculation and wrap the result."""
    result: CrackingResult = check.calculate_detailed(M_Ed=M_Ed, N_Ed=N_Ed)
    return _LoadCaseResult(
        name=name,
        M_Ed=M_Ed,
        N_Ed=N_Ed,
        w_k=result.w_k,
        w_k_limit=result.w_k_limit,
        is_cracked=result.is_cracked,
        passes=result.w_k <= result.w_k_limit,
    )


def _get_domain_bounds(
    check: CrackingCheck,
    concrete_model_type: ConcreteModelType,
    n_points: int,
) -> Tuple[float, float, float, float]:
    """Return (M_min, M_max, N_min, N_max) for the evaluation grid.

    All bounds come from the ULS diagram (design strengths). SLS loads
    that exceed ULS capacity are meaningless in a valid design.
    """
    uls_diagram = create_interaction_diagram(
        section=check.section,
        concrete=check.concrete,
        concrete_model_type=concrete_model_type,
        use_characteristic=False,
    )
    uls_pts = uls_diagram.generate_diagram_points(n_points=n_points)
    N_max = float(max(p.N for p in uls_pts))
    N_min = float(min(p.N for p in uls_pts))
    M_min = float(min(p.M for p in uls_pts))
    M_max = float(max(p.M for p in uls_pts))

    return M_min, M_max, N_min, N_max


def _eval_w_k(
    check: CrackingCheck,
    M: float,
    N: float,
    force_cracked: bool,
) -> float:
    """Evaluate crack width, returning NaN on failure."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = check.calculate_detailed(
                M_Ed=M, N_Ed=N, force_cracked=force_cracked,
            )
        return result.w_k
    except (ValueError, ZeroDivisionError):
        return float("nan")


def _find_crack_width_boundary(
    check: CrackingCheck,
    N_values: np.ndarray,
    M_min: float,
    M_max: float,
    w_k_limit: float,
    force_cracked: bool,
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """Find (M, N) boundary points where w_k = w_k_limit.

    For each N_i, use brentq to find M* where w_k(M*, N_i) = w_lim.
    Searches both positive M (sagging) and negative M (hogging) directions.

    Returns:
        (positive_boundary, negative_boundary) — each a list of (M, N) tuples.
    """
    from scipy.optimize import brentq

    LARGE_VALUE = 10.0  # Sentinel for "outside capacity"
    n_bracket = 30  # Coarse bracket resolution

    def g(M: float, N: float) -> float:
        w_k = _eval_w_k(check, M, N, force_cracked)
        if np.isnan(w_k):
            return LARGE_VALUE  # Acts as upper bracket
        return w_k - w_k_limit

    pos_boundary: List[Tuple[float, float]] = []
    neg_boundary: List[Tuple[float, float]] = []

    for N_i in N_values:
        N_i_float = float(N_i)

        # --- Positive M direction (sagging) ---
        M_probe = np.linspace(0.0, M_max, n_bracket)
        g_vals = [g(float(m), N_i_float) for m in M_probe]

        for k in range(len(g_vals) - 1):
            if g_vals[k] < 0 and g_vals[k + 1] >= 0:
                try:
                    M_star = brentq(
                        lambda m: g(m, N_i_float),
                        float(M_probe[k]),
                        float(M_probe[k + 1]),
                        xtol=0.1,
                        maxiter=30,
                    )
                    pos_boundary.append((M_star, N_i_float))
                except ValueError:
                    pass
                break  # Only first crossing per N slice

        # --- Negative M direction (hogging) ---
        if M_min < 0:
            M_probe_neg = np.linspace(0.0, M_min, n_bracket)
            g_vals_neg = [g(float(m), N_i_float) for m in M_probe_neg]

            for k in range(len(g_vals_neg) - 1):
                if g_vals_neg[k] < 0 and g_vals_neg[k + 1] >= 0:
                    try:
                        M_star = brentq(
                            lambda m: g(m, N_i_float),
                            float(M_probe_neg[k]),
                            float(M_probe_neg[k + 1]),
                            xtol=0.1,
                            maxiter=30,
                        )
                        neg_boundary.append((M_star, N_i_float))
                    except ValueError:
                        pass
                    break

    return pos_boundary, neg_boundary


class CrackWidthViewer:
    """
    Visualise crack widths for a ``CrackingCheck``.

    Args:
        check: A configured ``CrackingCheck`` instance.
    """

    def __init__(self, check: CrackingCheck) -> None:
        self.check = check

    # ------------------------------------------------------------------
    # Plot 1: 3D stem plot
    # ------------------------------------------------------------------
    def plot_load_cases(
        self,
        load_cases: Sequence[Dict[str, Any]],
        *,
        show: bool = True,
        title: Optional[str] = None,
        width: int = 900,
        height: int = 700,
    ) -> Any:
        """
        3D stem plot of crack widths at discrete M-N load cases.

        Each load case is a vertical line from the M-N plane (z = 0) up to
        z = w_k. A translucent plane at z = w_k,limit shows the crack width
        limit. Stems below the plane are green (pass), those that pierce it
        are red (fail).

        Args:
            load_cases: Sequence of dicts, each with keys:
                ``M_Ed`` (float, kN·m), ``N_Ed`` (float, kN),
                and optionally ``name`` (str).
            show: If True, call ``fig.show()``.
            title: Custom plot title.
            width: Figure width in pixels.
            height: Figure height in pixels.

        Returns:
            Plotly ``Figure`` object.
        """
        try:
            import plotly.graph_objects as go
        except ImportError as e:
            raise ImportError(
                "Plotly is required for plotting. Install with: pip install plotly"
            ) from e

        results: List[_LoadCaseResult] = []
        for idx, lc in enumerate(load_cases):
            M_Ed = float(lc["M_Ed"])
            N_Ed = float(lc.get("N_Ed", 0.0))
            name = str(lc.get("name", f"LC {idx + 1}"))
            results.append(_compute_load_case_result(self.check, M_Ed, N_Ed, name))

        w_k_limit = self.check.w_k_limit

        fig = go.Figure()

        # --- Limit plane ---
        all_M = [r.M_Ed for r in results]
        all_N = [r.N_Ed for r in results]
        M_min, M_max = min(all_M), max(all_M)
        N_min, N_max = min(all_N), max(all_N)
        M_pad = max(0.2 * (M_max - M_min), 20.0)
        N_pad = max(0.2 * (N_max - N_min), 20.0)

        plane_M = [M_min - M_pad, M_max + M_pad, M_max + M_pad, M_min - M_pad]
        plane_N = [N_min - N_pad, N_min - N_pad, N_max + N_pad, N_max + N_pad]
        plane_Z = [w_k_limit] * 4

        fig.add_trace(go.Mesh3d(
            x=plane_M,
            y=plane_N,
            z=plane_Z,
            i=[0, 0],
            j=[1, 2],
            k=[2, 3],
            color="rgba(255, 165, 0, 0.25)",
            name=f"w_k limit ({w_k_limit:.2f} mm)",
            hovertemplate=f"w_k limit = {w_k_limit:.2f} mm<extra></extra>",
            showlegend=True,
        ))

        # --- Stems ---
        for r in results:
            color = "green" if r.passes else "red"
            fig.add_trace(go.Scatter3d(
                x=[r.M_Ed, r.M_Ed],
                y=[r.N_Ed, r.N_Ed],
                z=[0.0, r.w_k],
                mode="lines",
                line=dict(color=color, width=4),
                name=r.name,
                showlegend=False,
                hoverinfo="skip",
            ))
            status = "PASS" if r.passes else "FAIL"
            hover = (
                f"<b>{r.name}</b><br>"
                f"M_Ed: {r.M_Ed:.1f} kN·m<br>"
                f"N_Ed: {r.N_Ed:.1f} kN<br>"
                f"w_k: {r.w_k:.3f} mm<br>"
                f"Limit: {r.w_k_limit:.2f} mm<br>"
                f"Status: {status}"
            )
            fig.add_trace(go.Scatter3d(
                x=[r.M_Ed],
                y=[r.N_Ed],
                z=[r.w_k],
                mode="markers+text",
                marker=dict(size=6, color=color),
                text=[r.name],
                textposition="top center",
                name=r.name,
                hovertemplate=hover + "<extra></extra>",
            ))

        fig.update_layout(
            title=dict(text=title or "Crack Width — Load Cases"),
            scene=dict(
                xaxis_title="Moment M (kN·m)",
                yaxis_title="Axial Force N (kN)",
                zaxis_title="Crack Width w_k (mm)",
            ),
            width=width,
            height=height,
        )

        if show:
            fig.show()
        return fig

    # ------------------------------------------------------------------
    # Plot 2: 2D contour map
    # ------------------------------------------------------------------
    def plot_contours(
        self,
        *,
        load_cases: Optional[Sequence[Dict[str, Any]]] = None,
        concrete_model_type: ConcreteModelType = ConcreteModelType.PARABOLA_RECTANGLE,
        n_grid: int = 20,
        n_boundary_points: int = 40,
        force_cracked: bool = True,
        show: bool = True,
        title: Optional[str] = None,
        width: int = 900,
        height: int = 700,
        n_envelope_points: int = 120,
    ) -> Any:
        """
        2D contour map of crack width across the M-N domain.

        A regular grid is laid over the ULS M-N region. The ``w_k = w_k,lim``
        boundary curve is computed via 1D root-finding on fixed-N slices
        (``scipy.optimize.brentq``), giving a crisp pass/fail boundary
        independent of grid resolution.

        Args:
            load_cases: Optional discrete load cases to overlay as markers.
            concrete_model_type: Concrete model for the ULS diagram used to
                determine domain bounds. Default ``PARABOLA_RECTANGLE``.
            n_grid: Number of grid divisions in each direction.
            n_boundary_points: Number of N slices for boundary root-finding.
            force_cracked: If True (default), compute crack widths even when
                M_Ed < M_cr. Recommended because a section cracked under one
                load case remains cracked under subsequent cases.
            show: If True, call ``fig.show()``.
            title: Custom plot title.
            width: Figure width in pixels.
            height: Figure height in pixels.
            n_envelope_points: Resolution for diagram point generation.

        Returns:
            Plotly ``Figure`` object.
        """
        try:
            import plotly.graph_objects as go
        except ImportError as e:
            raise ImportError(
                "Plotly is required for plotting. Install with: pip install plotly"
            ) from e

        # --- Domain bounds (all from ULS diagram) ---
        M_min, M_max, N_min, N_max = _get_domain_bounds(
            self.check, concrete_model_type, n_envelope_points,
        )

        w_k_limit = self.check.w_k_limit

        # --- Build evaluation grid ---
        M_vals = np.linspace(M_min, M_max, n_grid)
        N_vals = np.linspace(N_min, N_max, n_grid)
        W = np.full((n_grid, n_grid), np.nan)

        for i, n_val in enumerate(N_vals):
            for j, m_val in enumerate(M_vals):
                W[i, j] = _eval_w_k(
                    self.check, float(m_val), float(n_val), force_cracked,
                )

        # Cap display range at 3x w_k_limit to prevent outliers distorting colourscale
        w_max_display = w_k_limit * 3
        w_max_data = float(np.nanmax(W)) if np.any(~np.isnan(W)) else w_k_limit * 2
        w_max = min(w_max_data, w_max_display)

        # --- Boundary curve via root-finding ---
        N_boundary = np.linspace(N_min, N_max, n_boundary_points)
        pos_boundary, neg_boundary = _find_crack_width_boundary(
            self.check, N_boundary, M_min, M_max, w_k_limit, force_cracked,
        )

        fig = go.Figure()

        # --- Contour fill of w_k ---
        fig.add_trace(go.Contour(
            x=M_vals,
            y=N_vals,
            z=W,
            colorscale="YlOrRd",
            zmin=0.0,
            zmax=w_max,
            contours=dict(
                start=0.0,
                end=w_max,
                size=w_k_limit / 5,
                showlabels=True,
                labelfont=dict(size=10),
            ),
            colorbar=dict(title="w_k (mm)"),
            hovertemplate="M: %{x:.1f} kN·m<br>N: %{y:.1f} kN<br>w_k: %{z:.3f} mm<extra></extra>",
            name="Crack width",
        ))

        # --- Boundary curve (root-found w_k = w_k,lim line) ---
        for boundary, label_suffix in [
            (pos_boundary, ""),
            (neg_boundary, " (hogging)"),
        ]:
            if boundary:
                bM = [p[0] for p in boundary]
                bN = [p[1] for p in boundary]
                fig.add_trace(go.Scatter(
                    x=bM,
                    y=bN,
                    mode="lines",
                    line=dict(width=3, color="red", dash="dash"),
                    name=f"w_k = {w_k_limit:.2f} mm limit{label_suffix}",
                    hovertemplate=(
                        f"w_k = {w_k_limit:.2f} mm boundary<br>"
                        "M: %{x:.1f} kN·m<br>N: %{y:.1f} kN<extra></extra>"
                    ),
                ))

        # --- Optional load case markers ---
        if load_cases:
            for idx, lc in enumerate(load_cases):
                M_Ed = float(lc["M_Ed"])
                N_Ed = float(lc.get("N_Ed", 0.0))
                name = str(lc.get("name", f"LC {idx + 1}"))
                r = _compute_load_case_result(self.check, M_Ed, N_Ed, name)
                color = "green" if r.passes else "red"
                status = "PASS" if r.passes else "FAIL"
                hover = (
                    f"<b>{r.name}</b><br>"
                    f"M_Ed: {r.M_Ed:.1f} kN·m<br>"
                    f"N_Ed: {r.N_Ed:.1f} kN<br>"
                    f"w_k: {r.w_k:.3f} mm<br>"
                    f"Limit: {r.w_k_limit:.2f} mm<br>"
                    f"Status: {status}"
                )
                fig.add_trace(go.Scatter(
                    x=[M_Ed],
                    y=[N_Ed],
                    mode="markers",
                    marker=dict(size=10, color=color, line=dict(color="black", width=1)),
                    name=name,
                    hovertemplate=hover + "<extra></extra>",
                ))

        fig.update_layout(
            title=dict(text=title or "Crack Width — M-N Contour Map"),
            xaxis_title="Moment M (kN·m)",
            yaxis_title="Axial Force N (kN)",
            template="plotly_white",
            width=width,
            height=height,
            legend=dict(x=1.12, y=1, xanchor="left"),
            margin=dict(r=160),
        )

        if show:
            fig.show()
        return fig
