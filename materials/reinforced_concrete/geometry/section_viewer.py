from typing import TYPE_CHECKING, Any, Optional
import numpy as np
from shapely.geometry import Polygon

if TYPE_CHECKING:
    from materials.reinforced_concrete.geometry.section import RCSection
    from materials.reinforced_concrete.materials.concrete import ConcreteMaterial


class SectionViewer:
    def __init__(self, section: "RCSection") -> None:
        self.section = section

    def plot(
        self,
        *,
        concrete: Optional["ConcreteMaterial"] = None,
        show: bool = True,
        title: Optional[str] = None,
        width: int = 700,
        height: int = 700,
    ) -> Any:
        """
        Create an interactive Plotly figure of the section cross-section.

        Shows:
        - Concrete outline with fill and optional hatch pattern
        - Holes/voids (if any) shown as white areas
        - Rebar circles with detailed hover tooltips
        - Gross centroid marker
        - Transformed centroid marker (if concrete material provided)

        Hover tooltips include:
        - Concrete: Section name, grade (if provided), gross area, centroid
        - Rebars: Diameter, grade, area, position, layer name, group index

        Args:
            concrete: Optional ConcreteMaterial for hover info and transformed centroid
            show: If True, display the figure immediately
            title: Optional plot title (defaults to section name or "RC Section")
            width: Figure width in pixels
            height: Figure height in pixels

        Returns:
            Plotly Figure object for further customization

        Example:
            >>> from materials.reinforced_concrete.materials import ConcreteMaterial
            >>> section = create_rectangular_section(300, 500)
            >>> concrete = ConcreteMaterial(grade="C30/37")
            >>> # Add rebars...
            >>> fig = section.plot(concrete=concrete, show=True)
        """
        import plotly.graph_objects as go

        fig = go.Figure()

        # Get section bounds for axis limits
        min_x, min_y, max_x, max_y = self.section.get_bounding_box()
        cx, cy = self.section.get_centroid()
        area = self.section.get_area()
        I_xx, I_yy, I_xy = self.section.get_second_moment_area()            

        # Padding for plot limits
        pad_x = (max_x - min_x) * 0.1
        pad_y = (max_y - min_y) * 0.1

        # ===========================
        # 1. Draw concrete outline
        # ===========================

        # Build hover text for concrete
        concrete_hover_parts = [
            f"<b>Concrete Section</b>",
            f"Section Name: {self.section.section_name or 'unnamed'}",
        ]

        if concrete is not None:
            concrete_hover_parts.append(f"Material Name: {concrete.name}")
            concrete_hover_parts.append(f"Grade: {concrete.grade}")
            
        concrete_hover_parts.extend([
            f"Gross I_xx: {I_xx/10**6:,.0f} x10⁶ mm⁴",
            f"Gross I_yy: {I_yy/10**6:,.0f} x10⁶ mm⁴",
            f"Reinforcement Ratio: {self.section.reinforcement_ratio:.4f}",
        ])

        # Initialize default values
        transformed_data = None
        
        # 1. Perform transformed calculations ONCE
        if self.section.rebar_groups and concrete:
            E_cm = concrete.E_cm
            A_tr, cx_tr, cy_tr = self.section.get_transformed_centroid(E_cm=E_cm)
            I_tr_xx, I_tr_yy, I_tr_xy = self.section.get_transformed_second_moment_area(E_cm=E_cm)
            E_cm_GPa = E_cm / 1000.0

            # Store in a local dictionary or namedtuple to pass around
            transformed_data = {
                "A_tr": A_tr, "cx_tr": cx_tr, "cy_tr": cy_tr,
                "I_xx": I_tr_xx, "I_yy": I_tr_yy, "E_cm_GPa": E_cm_GPa
            }

            # Add to hover parts immediately
            concrete_hover_parts.extend([
                f"Elastic Modulus E_cm: {E_cm_GPa:,.2f} GPa",
                f"Transformed I_xx: {I_tr_xx/10**6:,.0f} x10⁶ mm⁴",
                f"Transformed I_yy: {I_tr_yy/10**6:,.0f} x10⁶ mm⁴",
            ])

        concrete_hover = "<br>".join(concrete_hover_parts)

        # Exterior ring
        ext_coords = np.asarray(self.section.outline.exterior.coords, dtype=float)
        x_ext = ext_coords[:, 0].tolist()
        y_ext = ext_coords[:, 1].tolist()

        # Add concrete fill (exterior)
        fig.add_trace(go.Scatter(
            x=x_ext,
            y=y_ext,
            fill="toself",
            fillcolor="rgba(180, 180, 180, 0.5)",  # Light gray with transparency
            line=dict(color="black", width=2),
            mode="lines",
            name="Concrete",
            hoverinfo="text",
            hovertext=concrete_hover,
            hoveron="fills+points",
        ))

        # Add interior rings (holes/voids) as white fill
        for i, interior in enumerate(self.section.outline.interiors):
            int_coords = np.asarray(interior.coords, dtype=float)
            x_int = int_coords[:, 0].tolist()
            y_int = int_coords[:, 1].tolist()

            void_hover = f"<b>Void {i+1}</b><br>Area: {Polygon(int_coords).area:,.0f} mm²"

            fig.add_trace(go.Scatter(
                x=x_int,
                y=y_int,
                fill="toself",
                fillcolor="white",
                line=dict(color="black", width=1.5, dash="dash"),
                mode="lines",
                name=f"Void {i+1}",
                hoverinfo="text",
                hovertext=void_hover,
                hoveron="fills+points",
            ))

        # ===========================
        # 2. Draw rebars
        # ===========================

        # Generate circle points for each rebar
        n_circle_pts = 32
        theta = np.linspace(0, 2 * np.pi, n_circle_pts, endpoint=True)

        for group_idx, group in enumerate(self.section.rebar_groups):
            r = float(group.rebar.diameter) / 2.0

            for bar_idx, pos in enumerate(group.positions):
                # Circle coordinates
                x_circle = (pos.x + r * np.cos(theta)).tolist()
                y_circle = (pos.y + r * np.sin(theta)).tolist()

                # Build detailed hover text
                hover_parts = [
                    f"<b>Rebar</b>",
                    f"Diameter: ϕ{group.rebar.diameter} mm",
                    f"Grade: {group.rebar.grade}",
                    f"Area: {group.rebar.area:.1f} mm²",
                    f"Position: ({pos.x:.1f}, {pos.y:.1f}) mm",
                ]
                if group.layer_name:
                    hover_parts.append(f"Layer: {group.layer_name}")

                hover_parts.extend([
                    f"Total Steel Area: {self.section.total_steel_area:.1f} mm²",
                ])

                rebar_hover = "<br>".join(hover_parts)

                # Determine legend group for cleaner legend
                legend_name = f"ϕ{group.rebar.diameter}"
                if group.layer_name:
                    legend_name += f" ({group.layer_name})"

                # Only show in legend for first bar of each group
                show_legend = (bar_idx == 0)

                fig.add_trace(go.Scatter(
                    x=x_circle,
                    y=y_circle,
                    fill="toself",
                    fillcolor="rgba(139, 0, 0, 0.9)",  # Dark red
                    line=dict(color="black", width=1),
                    mode="lines",
                    name=legend_name,
                    legendgroup=f"group_{group_idx}",
                    showlegend=show_legend,
                    hoverinfo="text",
                    hovertext=rebar_hover,
                    hoveron="fills+points",
                ))

        # ===========================
        # 3. Add centroid markers
        # ===========================
        fig.add_trace(go.Scatter(
            x=[cx],
            y=[cy],
            mode="markers",
            marker=dict(
                symbol="cross",
                size=12,
                color="blue",
                line=dict(width=2, color="blue"),
            ),
            name="Gross Centroid",
            hoverinfo="text",
            hovertext=(f"<b>Gross Centroid</b><br>"
                       f"Position: ({cx:.1f}, {cy:.1f}) mm<br>"
                       f"Gross Area: {area:,.0f} mm²"
            )
        ))

        # Add transformed centroid if concrete material is provided
        if transformed_data:
            fig.add_trace(go.Scatter(
                x=[transformed_data["cx_tr"]],
                y=[transformed_data["cy_tr"]],
                mode="markers",
                marker=dict(
                    symbol="x",
                    size=12,
                    color="green",
                    line=dict(width=2, color="green"),
                ),
                name="Transformed Centroid",
                hoverinfo="text",
                hovertext=(
                    f"<b>Transformed Centroid</b><br>"
                    f"Position: ({transformed_data['cx_tr']:.1f}, {transformed_data['cy_tr']:.1f}) mm<br>"
                    f"Transformed Area: {transformed_data['A_tr']:,.0f} mm²<br>"
                    f"Elastic Modulus: {transformed_data['E_cm_GPa']:,.2f} GPa"
                ),
            ))

        # ===========================
        # 4. Layout configuration
        # ===========================
        plot_title = title or self.section.section_name or "RC Section Cross-Section"

        fig.update_layout(
            title=dict(
                text=plot_title,
                font=dict(size=16, family="Arial, sans-serif"),
                x=0.5,
                xanchor="center",
            ),
            xaxis=dict(
                title="Width (mm)",
                range=[min_x - pad_x, max_x + pad_x],
                scaleanchor="y",
                scaleratio=1,
                showgrid=True,
                gridcolor="rgba(200, 200, 200, 0.5)",
                zeroline=False,
            ),
            yaxis=dict(
                title="Height (mm)",
                range=[min_y - pad_y, max_y + pad_y],
                showgrid=True,
                gridcolor="rgba(200, 200, 200, 0.5)",
                zeroline=False,
            ),
            width=width,
            height=height,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.02,
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="rgba(0, 0, 0, 0.3)",
                borderwidth=1,
            ),
            plot_bgcolor="white",
            hovermode="closest",
        )

        if show:
            fig.show()

        return fig
