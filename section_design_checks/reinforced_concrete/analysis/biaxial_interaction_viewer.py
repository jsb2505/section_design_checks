from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from section_design_checks.reinforced_concrete.analysis.biaxial_interaction import (
        BiaxialMNInteractionSurface,
    )


class BiaxialInteractionViewer:
    def __init__(self, surface: BiaxialMNInteractionSurface) -> None:
        self.surface = surface

    def plot(
        self,
        *,
        load_points: list[dict[str, Any]] | None = None,
        show_vectors: bool = False,
        show_metadata: bool = True,
        n_angles: int = 36,
        n_axial_levels: int = 20,
        save_path: str | None = None,
        show: bool = True,
        title: str | None = None,
    ) -> Any:
        """
        Plot biaxial M-M-N interaction surface with optional load points using Plotly.

        Args:
            load_points: List of load case dictionaries with format:
                {"N_Ed": float, "My_Ed": float, "Mz_Ed": float, "name": str}
            show_vectors: If True, show vector projection rays
            show_metadata: If True, show metadata in hover tooltips
            n_angles: Number of angles for surface generation
            n_axial_levels: Number of N levels for surface generation
            save_path: If provided, save plot to this file path (HTML format)
            show: If True, display plot in browser
            title: Custom plot title (optional)

        Returns:
            Plotly Figure object
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            raise ImportError(
                "Plotly is required for plotting. Install with: pip install plotly"
            )

        surface_pts = self.surface.generate_surface_pivot(
            n_angles=n_angles,
            n_axial_levels=n_axial_levels,
        )

        My_mat, Mz_mat, N_mat = self.surface._prepare_surface_matrices(
            surface_pts, n_axial_levels, n_angles
        )

        fig = go.Figure()

        fig.add_trace(go.Surface(
            x=My_mat,
            y=Mz_mat,
            z=N_mat,
            colorscale='Viridis',
            opacity=0.5,
            name='M-M-N Surface',
            showlegend=True,
            showscale=False,
            hoverinfo='skip',
        ))

        fig.add_trace(go.Scatter3d(
            x=[0],
            y=[0],
            z=[0],
            mode='markers',
            name='Origin',
            marker=dict(color='black', size=3, symbol='circle'),
            hovertemplate='Origin<extra></extra>',
        ))

        if load_points:
            for idx, lp in enumerate(load_points):
                N_Ed = lp.get("N_Ed", 0.0)
                My_Ed = lp.get("My_Ed", 0.0)
                Mz_Ed = lp.get("Mz_Ed", 0.0)
                name = lp.get("name", f"Load Case {idx + 1}")

                N_Rd, My_Rd, Mz_Rd, is_safe, utilization = self.surface.get_capacity_vector(
                    N_Ed=N_Ed, My_Ed=My_Ed, Mz_Ed=Mz_Ed,
                    surface_points=list(surface_pts),
                    n_angles=n_angles,
                    n_axial_levels=n_axial_levels
                )

                if utilization <= 0.8:
                    color = 'green'
                elif utilization <= 1.0:
                    color = 'orange'
                else:
                    color = 'red'

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

                legend_grp = f"lc_{idx}"
                fig.add_trace(go.Scatter3d(
                    x=[My_Ed],
                    y=[Mz_Ed],
                    z=[N_Ed],
                    mode='markers',
                    name=name,
                    legendgroup=legend_grp,
                    marker=dict(
                        color=color,
                        size=5,
                        symbol='circle',
                        line=dict(color='black', width=1)
                    ),
                    hovertemplate=hover_text + '<extra></extra>',
                    showlegend=True,
                ))

                if show_vectors and N_Rd is not None and My_Rd is not None and Mz_Rd is not None:
                    fig.add_trace(go.Scatter3d(
                        x=[0, My_Ed],
                        y=[0, Mz_Ed],
                        z=[0, N_Ed],
                        mode='lines',
                        line=dict(color=color, width=3, dash='solid'),
                        legendgroup=legend_grp,
                        showlegend=False,
                        hoverinfo='skip',
                    ))

                    fig.add_trace(go.Scatter3d(
                        x=[My_Ed, My_Rd],
                        y=[Mz_Ed, Mz_Rd],
                        z=[N_Ed, N_Rd],
                        mode='lines',
                        line=dict(color=color, width=3, dash='dash'),
                        legendgroup=legend_grp,
                        showlegend=False,
                        hoverinfo='skip',
                    ))

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
                aspectmode='cube',
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

        if save_path:
            fig.write_html(save_path)

        if show:
            fig.show()

        return fig
