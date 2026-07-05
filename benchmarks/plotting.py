"""
Plotly-based comparison plots for benchmark results.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from benchmarks.schemas import AnalysisResult


def plot_comparison(
    result: AnalysisResult,
    *,
    save_path: Optional[str | Path] = None,
    show: bool = False,
    title: Optional[str] = None,
    width: int = 1000,
    height: int = 750,
) -> Any:
    """
    Create an overlay plot of external vs internal M-N curves.

    External reference points are shown as discrete markers.
    Internal computed curve is shown as a solid line.
    Per-point error lines connect each external point to its nearest
    internal point, colour-coded by distance.

    Args:
        result: AnalysisResult from the benchmark runner.
        save_path: If provided, save plot as HTML to this path.
        show: If True, display the figure interactively.
        title: Custom plot title.
        width: Figure width in pixels.
        height: Figure height in pixels.

    Returns:
        Plotly Figure object.
    """
    import plotly.graph_objects as go

    fig = go.Figure()

    ext_M = [p[0] for p in result.external_points]
    ext_N = [p[1] for p in result.external_points]
    int_M = [p[0] for p in result.internal_points]
    int_N = [p[1] for p in result.internal_points]

    # Internal curve (solid line)
    fig.add_trace(go.Scatter(
        x=int_M, y=int_N,
        mode="lines",
        name="Internal (this package)",
        line=dict(color="blue", width=2),
        hovertemplate="M: %{x:.1f} kN·m<br>N: %{y:.1f} kN<extra></extra>",
    ))

    # External reference points (markers)
    fig.add_trace(go.Scatter(
        x=ext_M, y=ext_N,
        mode="markers",
        name=f"External reference",
        marker=dict(color="red", size=5, symbol="circle"),
        hovertemplate="M: %{x:.1f} kN·m<br>N: %{y:.1f} kN<extra></extra>",
    ))

    # Error connector lines (from external to nearest internal)
    if result.per_point_errors:
        max_dist = max(pe.distance for pe in result.per_point_errors) if result.per_point_errors else 1.0
        for pe in result.per_point_errors:
            # Colour intensity based on error magnitude
            intensity = min(pe.distance / max(max_dist, 0.01), 1.0)
            r = int(255 * intensity)
            g = int(255 * (1 - intensity))
            colour = f"rgba({r}, {g}, 0, 0.4)"

            fig.add_trace(go.Scatter(
                x=[pe.ext_M, pe.int_M],
                y=[pe.ext_N, pe.int_N],
                mode="lines",
                line=dict(color=colour, width=1),
                showlegend=False,
                hoverinfo="skip",
            ))

    # Status annotation
    status_colour = "green" if result.status == "PASS" else "red"
    status_text = (
        f"{result.status} — Hausdorff: {result.hausdorff_distance:.2f} "
        f"(tol: {result.tolerance:.1f})"
    )

    plot_title = title or f"Benchmark: {result.type} ({result.limit_state})"

    fig.update_layout(
        title=dict(text=plot_title),
        xaxis_title="Moment M (kN·m)",
        yaxis_title="Axial Force N (kN)",
        xaxis=dict(gridcolor="lightgray", zeroline=True),
        yaxis=dict(gridcolor="lightgray", zeroline=True),
        template="plotly_white",
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
        width=width,
        height=height,
        annotations=[
            dict(
                text=status_text,
                xref="paper", yref="paper",
                x=0.02, y=0.02,
                showarrow=False,
                font=dict(size=14, color=status_colour),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor=status_colour,
                borderwidth=1,
                borderpad=4,
            )
        ],
    )

    if save_path:
        fig.write_html(str(save_path))
    if show:
        fig.show()

    return fig
