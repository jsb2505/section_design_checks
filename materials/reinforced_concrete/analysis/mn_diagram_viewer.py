from typing import Any, Dict, List, Optional
from pathlib import Path
from materials.reinforced_concrete.analysis import MNInteractionDiagram


class MNDiagramViewer:
    def __init__(self, diagram: "MNInteractionDiagram") -> None:
        self.diagram = diagram

    def plot(
        self,
        *,
        load_points: Optional[List[Dict[str, Any]]] = None,
        show_vectors: bool = False,
        show_metadata: bool = True,
        n_points: int = 120,
        save_path: Optional[str | Path] = None,
        show: bool = True,
        title: Optional[str] = None,
        width: int = 900,
        height: int = 700,
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
            width: Figure width in pixels
            height: Figure height in pixels

        Returns:
            Plotly Figure object
        """
        try:
            import plotly.graph_objects as go
        except ImportError as e:
            raise ImportError(
                "Plotly is required for plotting. Install with: pip install plotly"
            ) from e

        # 1. Generate core diagram data
        diagram_points = self.diagram.generate_diagram_points(n_points=n_points)
        M_curve = [p.M for p in diagram_points]
        N_curve = [p.N for p in diagram_points]

        fig = go.Figure()

        # Initialize tracking for axis limits to avoid re-calculating later
        xs = list(M_curve) + [0.0]
        ys = list(N_curve) + [0.0]

        # 2. Plot Capacity Curve
        fig.add_trace(go.Scatter(
            x=M_curve, y=N_curve,
            mode="lines",
            name="M-N Capacity",
            line=dict(color="black", width=2),
            hovertemplate="M: %{x:.1f} kN·m<br>N: %{y:.1f} kN<extra></extra>",
        ))

        # 3. Add Origin Marker
        fig.add_trace(go.Scatter(
            x=[0.0], y=[0.0],
            mode="markers",
            name="Origin",
            marker=dict(color="black", size=4, symbol="circle"),
            hovertemplate="Origin (0,0)<extra></extra>",
        ))

        # 4. Process Load Points
        if load_points:
            for idx, lp in enumerate(load_points):
                N_Ed = float(lp.get("N_Ed", 0.0))
                M_Ed = float(lp.get("M_Ed", 0.0))
                name_lp = str(lp.get("name", f"Load Case {idx + 1}"))

                # Calculate capacity ONCE per load case
                capacity = self.diagram.get_capacity_vector(
                    N_Ed=N_Ed, M_Ed=M_Ed, n_points=n_points, return_details=False
                )

                # Update bounds trackers
                xs.append(M_Ed)
                ys.append(N_Ed)
                if capacity.M_Rd is not None and capacity.N_Rd is not None:
                    xs.append(capacity.M_Rd)
                    ys.append(capacity.N_Rd)

                # Color logic based on utilization
                if capacity.utilization <= 0.8:
                    color = "green"
                elif capacity.utilization <= 1.0:
                    color = "orange"
                else:
                    color = "red"

                # 5. Draw Vectors (if requested and valid)
                if show_vectors and capacity.M_Rd is not None and capacity.N_Rd is not None:
                    legend_grp = f"lc_{idx}"
                    # Demand Vector: Origin to Load (Solid)
                    fig.add_trace(go.Scatter(
                        x=[0.0, M_Ed], y=[0.0, N_Ed],
                        mode="lines",
                        line=dict(color=color, width=1.5, dash="solid"),
                        legendgroup=legend_grp,
                        showlegend=False,
                        hoverinfo="skip",
                    ))
                    # Reserve Vector: Load to Capacity (Dashed)
                    fig.add_trace(go.Scatter(
                        x=[M_Ed, capacity.M_Rd], y=[N_Ed, capacity.N_Rd],
                        mode="lines",
                        line=dict(color=color, width=1.5, dash="dash"),
                        legendgroup=legend_grp,
                        showlegend=False,
                        hoverinfo="skip",
                    ))

                # 6. Build Hover Metadata (Respecting show_metadata arg)
                if show_metadata:
                    hover_text = (
                        f"<b>{name_lp}</b><br>"
                        f"N_Ed: {N_Ed:.1f} kN<br>"
                        f"M_Ed: {M_Ed:.1f} kN·m<br>"
                    )
                    if capacity.N_Rd is not None:
                        hover_text += (
                            f"N_Rd: {capacity.N_Rd:.1f} kN<br>"
                            f"M_Rd: {capacity.M_Rd:.1f} kN·m<br>"
                            f"Utilization: {capacity.utilization:.1%}<br>"
                            f"Status: {'✓ PASS' if capacity.is_safe else '✗ FAIL'}"
                        )
                else:
                    hover_text = name_lp

                # 7. Plot Load Point Marker
                fig.add_trace(go.Scatter(
                    x=[M_Ed], y=[N_Ed],
                    mode="markers",
                    name=name_lp,
                    legendgroup=f"lc_{idx}",
                    marker=dict(color=color, size=8, symbol="circle", line=dict(color="black", width=1)),
                    hovertemplate=hover_text + "<extra></extra>",
                ))

        # 8. Axis Range and Layout
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        xpad = 0.05 * (xmax - xmin) if xmax > xmin else 1.0
        ypad = 0.05 * (ymax - ymin) if ymax > ymin else 1.0

        fig.update_layout(
            title=dict(text=title or "M-N Interaction Diagram"),
            xaxis_title="Moment M (kN·m)",
            yaxis_title="Axial Force N (kN)",
            xaxis=dict(range=[xmin - xpad, xmax + xpad], gridcolor="lightgray", zeroline=True),
            yaxis=dict(range=[ymin - ypad, ymax + ypad], gridcolor="lightgray", zeroline=True),
            template="plotly_white",
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
            width=width, height=height,
        )

        if save_path:
            fig.write_html(str(save_path))
        if show:
            fig.show()

        return fig