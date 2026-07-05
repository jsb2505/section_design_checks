from pathlib import Path
from typing import Any, Optional, Tuple, Literal
from dataclasses import dataclass
import numpy as np

from materials.reinforced_concrete.analysis import MNInteractionDiagram
from materials.core.units import FORCE_TO_KN, ForceUnit, MomentUnit, to_kn, to_knm


# ------------------------------------------
# Helper Class for Stress-Strain Plot State
# ------------------------------------------

@dataclass(frozen=True)
class _StressStrainPlotState:
    # inputs
    M_Ed: float
    N_Ed: float

    # solved end strains
    eps_top: float
    eps_bottom: float

    # fibre fields (all same length)
    forces_N: np.ndarray
    areas: np.ndarray
    x: np.ndarray
    y: np.ndarray
    strains: np.ndarray
    stresses: np.ndarray
    conc_mask: np.ndarray
    steel_mask: np.ndarray

    # section geometry
    y_top: float
    y_bottom: float
    h: float

    # neutral axis
    y_na: float | None
    na_in_section: bool

    # resultants (kN)
    F_c_comp: float
    F_c_tens: float
    F_s_comp: float
    F_s_tens: float

    # centroids (mm)
    y_c_comp: float | None
    y_c_tens: float | None
    y_s_comp: float | None
    y_s_tens: float | None

    # overall compression/tension centroids + lever arm
    y_C: float | None
    y_T: float | None
    z: float | None

    # stress range info for scaling / axes
    max_stress_pos: float
    min_stress_neg: float
    force_scale: float

    fibre_i: np.ndarray | None
    fibre_j: np.ndarray | None
    bbox: tuple[float, float, float, float]

    # Equilibrium check: if solver hit bounds, the achieved forces won't match applied
    section_failed: bool  # True if equilibrium cannot be achieved (loads exceed capacity)
    achieved_N: float  # Actual axial force from the computed strain state
    achieved_M: float  # Actual moment from the computed strain state
    equilibrium_error_N: float  # |N_achieved - N_Ed| in kN
    equilibrium_error_M: float  # |M_achieved - M_Ed| in kN·m

    # Capacity at the applied axial force level
    M_Rd_pos: float | None  # positive moment capacity (kN·m)
    M_Rd_neg: float | None  # negative moment capacity (kN·m)
    N_Rd: float | None  # capped axial level used (kN)
    utilisation: float | None  # utilisation ratio


class StressStrainViewer:
    def __init__(self, diagram: "MNInteractionDiagram") -> None:
        self.diagram = diagram
 
    def plot(
        self,
        M_Ed: float,
        N_Ed: float,
        *,
        save_path: Optional[str | Path] = None,
        show: bool = True,
        title: Optional[str] = None,
        width: int = 1100,
        height: int = 1000,
        section_render: Literal["points", "filled"] = "points",
    ) -> Any:
        """
        Visualize stress and strain distribution for a given load case.

        Solves for the strain state (ε_top, ε_bottom) that produces the target (M_Ed, N_Ed),
        then displays in a 2×2 quadrant layout:
        - Top-left: Section cross-section with stress colour map (true 1:1 aspect ratio)
        - Top-right: Results annotation (load case, capacity, resultants)
        - Bottom-left: Strain profile (linear distribution across depth)
        - Bottom-right: Stress profile (concrete stress block and steel forces)

        Args:
            M_Ed: Applied moment (kN·m)
            N_Ed: Applied axial force (kN, compression positive)
            save_path: If provided, save plot to this file path (HTML format)
            show: If True, display plot (fig.show())
            title: Custom plot title (optional)
            width: Figure width in pixels
            height: Figure height in pixels
            section_render: Concrete rendering mode ('points' or 'filled')
        """
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError as e:
            raise ImportError(
                "Plotly is required for plotting. Install with: pip install plotly"
            ) from e

        state = self._build_stress_strain_plot_state(M_Ed=M_Ed, N_Ed=N_Ed)

        fig = make_subplots(
            rows=2,
            cols=2,
            row_heights=[0.45, 0.55],
            column_widths=[0.5, 0.5],
            subplot_titles=("Section (Stress)", "", "Strain Profile", "Stress Profile"),
            horizontal_spacing=0.08,
            vertical_spacing=0.12,
            shared_yaxes=False,
            specs=[
                [{"type": "xy"}, {"type": "xy"}],
                [{"type": "xy"}, {"type": "xy"}],
            ],
        )

        self._add_section_subplot(fig, go, state, section_render=section_render, row=1, col=1)
        self._add_strain_subplot(fig, go, state, row=2, col=1)
        self._add_stress_subplot(fig, go, state, row=2, col=2)

        self._apply_stress_strain_layout(fig, state, title=title, width=width, height=height)

        if save_path:
            fig.write_html(str(save_path))
        if show:
            fig.show()

        return fig

    def _get_capacity(self, *, M_Ed: float, N_Ed: float) -> dict:
        """Get M_Rd, N_Rd and utilisation via the capacity vector method."""
        result = self.diagram.get_capacity_vector(N_Ed=N_Ed, M_Ed=M_Ed)
        return dict(
            M_Rd_pos=float(result.M_Rd) if result.M_Rd is not None else None,
            M_Rd_neg=None,
            N_Rd=float(result.N_Rd) if result.N_Rd is not None else None,
            utilisation=float(result.utilization) if result.utilization is not None and result.utilization != float("inf") else None,
        )

    # -----------------------------
    # Build plot state
    # -----------------------------
    def _build_stress_strain_plot_state(self, *, M_Ed: float, N_Ed: float) -> _StressStrainPlotState:
        d = self.diagram  # shorthand

        # 1) Solve for strain state
        try:
            eps_top, eps_bottom = d.find_strains_for_MN(M_target=M_Ed, N_target=N_Ed)
        except ValueError as e:
            raise ValueError(
                f"Cannot find strain state for M_Ed={M_Ed:.1f} kN·m, N_Ed={N_Ed:.1f} kN. "
                f"Solver failed numerically. Original error: {e}"
            ) from e

        # 2) Fibre-level data
        forces_N, y_coords, areas = d.get_fibre_forces_from_end_strains(eps_top, eps_bottom)
        strains = d._strain_field_from_end_strains(eps_top=eps_top, eps_bottom=eps_bottom)
        stresses = np.divide(forces_N, areas, out=np.zeros_like(forces_N), where=areas > 0)

        material_type = d._fibre_mat
        x_coords = d._fibre_x

        conc_mask = material_type == "concrete"
        steel_mask = material_type == "steel"

        # 3) Section geometry
        y_top = float(d.section_top)
        y_bottom = float(d.section_bottom)
        h = float(d.section_height)

        # 4) Neutral axis
        y_na, na_in_section = self._neutral_axis(
            eps_top=float(eps_top),
            eps_bottom=float(eps_bottom),
            y_top=y_top,
            y_bottom=y_bottom,
            h=h,
        )

        # 5) Resultants (kN)
        _N_to_kN = FORCE_TO_KN[ForceUnit.N]
        conc_forces = forces_N[conc_mask] * _N_to_kN
        steel_forces = forces_N[steel_mask] * _N_to_kN

        F_c_comp = float(np.sum(conc_forces[conc_forces > 0.0])) if conc_forces.size else 0.0
        F_c_tens = float(np.sum(conc_forces[conc_forces < 0.0])) if conc_forces.size else 0.0
        F_s_comp = float(np.sum(steel_forces[steel_forces > 0.0])) if steel_forces.size else 0.0
        F_s_tens = float(np.sum(steel_forces[steel_forces < 0.0])) if steel_forces.size else 0.0

        # 5b) Equilibrium check: compute achieved N and M from the strain state
        # This tells us if the solver hit bounds and couldn't find an exact solution
        achieved_N = to_kn(float(np.sum(forces_N)), ForceUnit.N)
        # Moment about centroid (using section centroid y-coordinate)
        centroid_x, centroid_y = d.section.get_centroid()
        achieved_M_raw = float(np.sum(forces_N * (y_coords - centroid_y)))
        achieved_M = to_knm(achieved_M_raw, MomentUnit.NMM)

        equilibrium_error_N = abs(achieved_N - N_Ed)
        equilibrium_error_M = abs(achieved_M - M_Ed)

        # Determine if section failed: if errors exceed small tolerance, the solver
        # hit strain bounds and couldn't achieve equilibrium with the applied loads
        # Use relative tolerance where possible, absolute for near-zero values
        tol_N = max(1.0, 0.01 * abs(N_Ed))  # 1% or 1 kN
        tol_M = max(1.0, 0.01 * abs(M_Ed))  # 1% or 1 kN·m
        section_failed = equilibrium_error_N > tol_N or equilibrium_error_M > tol_M

        # 6) Centroids (mm) — weight by forces in N
        conc_comp_mask = conc_mask & (forces_N > 0.0)
        conc_tens_mask = conc_mask & (forces_N < 0.0)
        steel_comp_mask = steel_mask & (forces_N > 0.0)
        steel_tens_mask = steel_mask & (forces_N < 0.0)

        y_c_comp = self._weighted_centroid_y(forces_N, y_coords, conc_comp_mask)
        y_c_tens = self._weighted_centroid_y(forces_N, y_coords, conc_tens_mask)
        y_s_comp = self._weighted_centroid_y(forces_N, y_coords, steel_comp_mask)
        y_s_tens = self._weighted_centroid_y(forces_N, y_coords, steel_tens_mask)

        # 7) Overall C/T centroids & lever arm
        comp_mask = forces_N > 0.0
        tens_mask = forces_N < 0.0
        y_C = self._weighted_centroid_y(forces_N, y_coords, comp_mask)
        y_T = self._weighted_centroid_y(forces_N, y_coords, tens_mask)
        z = float(abs(y_C - y_T)) if (y_C is not None and y_T is not None) else None

        # 8) Stress range (concrete only) + force scaling for arrows
        max_stress_pos, min_stress_neg = self._concrete_stress_range(stresses, conc_mask)
        force_scale = self._force_scale(
            max_stress_pos=max_stress_pos,
            min_stress_neg=min_stress_neg,
            F_c_comp=F_c_comp,
            F_c_tens=F_c_tens,
            F_s_comp=F_s_comp,
            F_s_tens=F_s_tens,
        )

        # 9) IMPORTANT: pull i/j indices from diagram (for filled render)
        fibre_i = getattr(d, "_fibre_i", None)
        fibre_j = getattr(d, "_fibre_j", None)

        min_x, min_y, max_x, max_y = d.section.get_bounding_box()
        bbox: tuple[float, float, float, float] = (
            float(min_x),
            float(min_y),
            float(max_x),
            float(max_y),
        )

        return _StressStrainPlotState(
            M_Ed=float(M_Ed),
            N_Ed=float(N_Ed),
            eps_top=float(eps_top),
            eps_bottom=float(eps_bottom),
            forces_N=forces_N,
            areas=areas,
            x=x_coords,
            y=y_coords,
            strains=strains,
            stresses=stresses,
            conc_mask=conc_mask,
            steel_mask=steel_mask,
            y_top=y_top,
            y_bottom=y_bottom,
            h=h,
            y_na=y_na,
            na_in_section=na_in_section,
            F_c_comp=F_c_comp,
            F_c_tens=F_c_tens,
            F_s_comp=F_s_comp,
            F_s_tens=F_s_tens,
            y_c_comp=y_c_comp,
            y_c_tens=y_c_tens,
            y_s_comp=y_s_comp,
            y_s_tens=y_s_tens,
            y_C=y_C,
            y_T=y_T,
            z=z,
            max_stress_pos=max_stress_pos,
            min_stress_neg=min_stress_neg,
            force_scale=force_scale,
            fibre_i=fibre_i,
            fibre_j=fibre_j,
            bbox=bbox,
            section_failed=section_failed,
            achieved_N=achieved_N,
            achieved_M=achieved_M,
            equilibrium_error_N=equilibrium_error_N,
            equilibrium_error_M=equilibrium_error_M,
            **self._get_capacity(M_Ed=M_Ed, N_Ed=N_Ed),
        )


    # -----------------------------
    # Subplot builders
    # -----------------------------
    def _add_section_subplot(
        self,
        fig: Any,
        go: Any,
        s: _StressStrainPlotState,
        *,
        section_render: Literal["points", "filled"] = "points",
        row: int = 1,
        col: int = 1,
    ) -> None:
        # Concrete stress map
        if np.any(s.conc_mask):
            if section_render == "filled":
                self._add_concrete_filled_field(fig, go, s, row=row, col=col)
            else:
                # points
                conc_x = s.x[s.conc_mask]
                conc_y = s.y[s.conc_mask]
                conc_stresses = s.stresses[s.conc_mask]

                cmin_val, cmax_val, colorscale = self._concrete_colorscale(conc_stresses)

                scatter_cls = getattr(go, "Scattergl", go.Scatter)

                fig.add_trace(
                    scatter_cls(
                        x=conc_x,
                        y=conc_y,
                        mode="markers",
                        marker=dict(
                            size=6,
                            color=conc_stresses,
                            colorscale=colorscale,
                            cmin=cmin_val,
                            cmax=cmax_val,
                            colorbar=dict(
                                title=dict(text="σ<br>(MPa)", side="bottom"),
                                thickness=15,
                            ),
                        ),
                        hovertemplate=(
                            "x: %{x:.1f} mm<br>"
                            "y: %{y:.1f} mm<br>"
                            "σ: %{marker.color:.2f} MPa<br>"
                            "<extra>Concrete</extra>"
                        ),
                        name="Concrete",
                        showlegend=False,
                    ),
                    row=row,
                    col=col,
                )

        # Rebar markers split by sign so legend can toggle compression/tension independently.
        if np.any(s.steel_mask):
            sx = s.x[s.steel_mask].astype(float)
            sy = s.y[s.steel_mask].astype(float)
            ss = s.stresses[s.steel_mask].astype(float)
            sf = (s.forces_N[s.steel_mask] * FORCE_TO_KN[ForceUnit.N]).astype(float)
            se = (s.strains[s.steel_mask] * 1000.0).astype(float)   # ‰
            comp_mask = ss >= 0.0
            tens_mask = ss < 0.0

            if np.any(comp_mask):
                comp_custom = np.column_stack([ss[comp_mask], se[comp_mask], sf[comp_mask]])
                fig.add_trace(
                    go.Scatter(
                        x=sx[comp_mask],
                        y=sy[comp_mask],
                        mode="markers",
                        marker=dict(size=12, color="darkorange", line=dict(color="black", width=1)),
                        customdata=comp_custom,
                        hovertemplate=(
                            "x: %{x:.1f} mm<br>"
                            "y: %{y:.1f} mm<br>"
                            "σ: %{customdata[0]:.1f} MPa<br>"
                            "ε: %{customdata[1]:.3f} ‰<br>"
                            "F: %{customdata[2]:.1f} kN<br>"
                            "<extra>Rebar (compression)</extra>"
                        ),
                        name="Rebar (compression)",
                        showlegend=True,
                    ),
                    row=row,
                    col=col,
                )

            if np.any(tens_mask):
                tens_custom = np.column_stack([ss[tens_mask], se[tens_mask], sf[tens_mask]])
                fig.add_trace(
                    go.Scatter(
                        x=sx[tens_mask],
                        y=sy[tens_mask],
                        mode="markers",
                        marker=dict(size=12, color="green", line=dict(color="black", width=1)),
                        customdata=tens_custom,
                        hovertemplate=(
                            "x: %{x:.1f} mm<br>"
                            "y: %{y:.1f} mm<br>"
                            "σ: %{customdata[0]:.1f} MPa<br>"
                            "ε: %{customdata[1]:.3f} ‰<br>"
                            "F: %{customdata[2]:.1f} kN<br>"
                            "<extra>Rebar (tension)</extra>"
                        ),
                        name="Rebar (tension)",
                        showlegend=True,
                    ),
                    row=row,
                    col=col,
                )

        # Neutral axis line on section (clipped)
        if s.y_na is not None and s.na_in_section:
            segs = self._section_horizontal_segments_at_y(s.y_na)
            for i, (xa, xb) in enumerate(segs):
                fig.add_trace(
                    go.Scatter(
                        x=[xa, xb],
                        y=[s.y_na, s.y_na],
                        mode="lines",
                        line=dict(color="purple", width=2, dash="dash"),
                        name="Neutral Axis" if i == 0 else None,
                        showlegend=(i == 0),
                    ),
                    row=row,
                    col=col,
                )

        # Draw the section boundary last so it stays crisp above the stress field.
        outline_x, outline_y = self._get_outline_xy()
        fig.add_trace(
            go.Scatter(
                x=outline_x,
                y=outline_y,
                mode="lines",
                line=dict(color="black", width=2),
                name="Section outline",
                showlegend=False,
                hoverinfo="skip",
            ),
            row=row,
            col=col,
        )

    def _add_strain_subplot(self, fig: Any, go: Any, s: _StressStrainPlotState, *, row: int = 2, col: int = 1) -> None:
        eps_bottom_permille = s.eps_bottom * 1000.0
        eps_top_permille = s.eps_top * 1000.0

        strain_polygon_x = [0.0, eps_bottom_permille, eps_top_permille, 0.0, 0.0]
        strain_polygon_y = [s.y_bottom, s.y_bottom, s.y_top, s.y_top, s.y_bottom]

        # Use red when section fails (bounded solution), blue otherwise
        if s.section_failed:
            strain_line_color = "red"
            strain_fill_color = "rgba(255, 0, 0, 0.15)"
            strain_name = "Strain (FAILED)"
        else:
            strain_line_color = "blue"
            strain_fill_color = "rgba(0, 0, 255, 0.15)"
            strain_name = "Strain"

        fig.add_trace(
            go.Scatter(
                x=strain_polygon_x,
                y=strain_polygon_y,
                mode="lines",
                line=dict(color=strain_line_color, width=2),
                fill="toself",
                fillcolor=strain_fill_color,
                name=strain_name,
                hovertemplate="ε: %{x:.3f} ‰<br>y: %{y:.1f} mm<extra></extra>",
            ),
            row=row,
            col=col,
        )

        fig.add_trace(
            go.Scatter(
                x=[eps_bottom_permille, eps_top_permille],
                y=[s.y_bottom, s.y_top],
                mode="markers",
                marker=dict(size=8, color=strain_line_color),
                showlegend=False,
                hovertemplate="ε: %{x:.3f} ‰<br>y: %{y:.1f} mm<extra></extra>",
            ),
            row=row,
            col=col,
        )

        fig.add_trace(
            go.Scatter(
                x=[0, 0],
                y=[s.y_bottom - 20, s.y_top + 20],
                mode="lines",
                line=dict(color="gray", width=1, dash="dot"),
                showlegend=False,
            ),
            row=row,
            col=col,
        )

        if s.y_na is not None and s.na_in_section:
            fig.add_trace(
                go.Scatter(
                    x=[0],
                    y=[s.y_na],
                    mode="markers",
                    marker=dict(size=10, color="purple", symbol="diamond"),
                    name="NA",
                    showlegend=False,
                    hovertemplate=f"Neutral Axis<br>y: {s.y_na:.1f} mm<extra></extra>",
                ),
                row=row,
                col=col,
            )

    def _add_stress_subplot(self, fig: Any, go: Any, s: _StressStrainPlotState, *, row: int = 2, col: int = 2) -> None:
        # Concrete stress profile polygon + hover markers
        if np.any(s.conc_mask):
            # Use interpolated profile for smooth stress block visualization
            # This gives much better resolution in compression zones which are typically
            # only 20-30% of section depth but contain the important stress block shape
            interp_y, interp_strains, interp_stresses = self._interpolate_concrete_stress_profile(s, n_points=100)

            # Build polygon: start at zero, trace up the stress curve, back to zero
            stress_polygon_x = np.concatenate([[0.0], interp_stresses, [0.0]])
            stress_polygon_y = np.concatenate([[interp_y[0]], interp_y, [interp_y[-1]]])

            # Use red/pink when section fails, gray otherwise
            if s.section_failed:
                stress_line_color = "darkred"
                stress_fill_color = "rgba(255, 100, 100, 0.3)"
                stress_marker_color = "darkred"
                stress_name = "Concrete σ (FAILED)"
            else:
                stress_line_color = "gray"
                stress_fill_color = "rgba(128, 128, 128, 0.3)"
                stress_marker_color = "gray"
                stress_name = "Concrete σ"

            fig.add_trace(
                go.Scatter(
                    x=stress_polygon_x,
                    y=stress_polygon_y,
                    mode="lines",
                    line=dict(color=stress_line_color, width=2),
                    fill="toself",
                    fillcolor=stress_fill_color,
                    name=stress_name,
                    hoverinfo="skip",
                    legendgroup="concrete_stress",
                ),
                row=row,
                col=col,
            )

            # Add hover markers using interpolated points for smooth hover experience
            hover_texts = [
                f"σ: {interp_stresses[i]:.2f} MPa<br>"
                f"ε: {interp_strains[i]*1000:.3f} ‰<br>"
                f"y: {interp_y[i]:.1f} mm"
                for i in range(len(interp_stresses))
            ]

            fig.add_trace(
                go.Scatter(
                    x=interp_stresses,
                    y=interp_y,
                    mode="markers",
                    marker=dict(size=3, color=stress_marker_color, opacity=0.3),
                    text=hover_texts,
                    hovertemplate="%{text}<extra>Concrete</extra>",
                    showlegend=False,
                    legendgroup="concrete_stress",
                ),
                row=row,
                col=col,
            )

        # Resultant arrows
        self._add_resultant_arrow(fig, go, row=row, col=col, name="F<sub>cc</sub>", force=s.F_c_comp, y=s.y_c_comp, force_scale=s.force_scale, line_color="red", tip_symbol="triangle-right", extra="Concrete Compression")
        self._add_resultant_arrow(fig, go, row=row, col=col, name="F<sub>ct</sub>", force=s.F_c_tens, y=s.y_c_tens, force_scale=s.force_scale, line_color="blue", tip_symbol="triangle-left", extra="Concrete Tension")
        self._add_resultant_arrow(fig, go, row=row, col=col, name="F<sub>sc</sub>", force=s.F_s_comp, y=s.y_s_comp, force_scale=s.force_scale, line_color="darkorange", tip_symbol="triangle-right", extra="Steel Compression")
        self._add_resultant_arrow(fig, go, row=row, col=col, name="F<sub>st</sub>", force=s.F_s_tens, y=s.y_s_tens, force_scale=s.force_scale, line_color="green", tip_symbol="triangle-left", extra="Steel Tension")

        # Zero stress line
        fig.add_trace(
            go.Scatter(
                x=[0, 0],
                y=[s.y_bottom - 20, s.y_top + 20],
                mode="lines",
                line=dict(color="gray", width=1, dash="dot"),
                showlegend=False,
            ),
            row=row,
            col=col,
        )

        # NA marker on stress plot
        if s.y_na is not None and s.na_in_section:
            fig.add_trace(
                go.Scatter(
                    x=[0],
                    y=[s.y_na],
                    mode="markers",
                    marker=dict(size=10, color="purple", symbol="diamond"),
                    showlegend=False,
                    hovertemplate=f"Neutral Axis<br>y: {s.y_na:.1f} mm<extra></extra>",
                ),
                row=row,
                col=col,
            )

    # -----------------------------
    # Layout / axes / annotation
    # -----------------------------
    def _apply_stress_strain_layout(
        self,
        fig: Any,
        s: _StressStrainPlotState,
        *,
        title: Optional[str],
        width: int,
        height: int,
    ) -> None:
        if title is None:
            title = f"Stress-Strain Distribution: M<sub>Ed</sub> = {s.M_Ed:.1f} kN·m, N<sub>Ed</sub> = {s.N_Ed:.1f} kN"
            if s.section_failed:
                title += (
                    "<br><span style='color:red; font-size:11px'><b>"
                    "SECTION FAILS: Applied loads exceed capacity. Strains shown are "
                    "bounded approximation NOT in equilibrium with applied forces."
                    "</b></span>"
                )

        margin_top = 110 if s.section_failed else 80

        fig.update_layout(
            title=dict(text=title, x=0.5),
            width=width,
            height=height,
            showlegend=True,
            legend=dict(x=1.02, y=1),
            margin=dict(l=60, r=120, t=margin_top, b=60),
        )

        # ---- Top-left: Section with true 1:1 aspect ratio ----
        fig.update_xaxes(title_text="x (mm)", row=1, col=1)
        fig.update_yaxes(
            title_text="y (mm)",
            scaleanchor="x",
            scaleratio=1,
            row=1, col=1,
        )

        # ---- Top-right: Annotation only (hide axes) ----
        fig.update_xaxes(visible=False, row=1, col=2)
        fig.update_yaxes(visible=False, row=1, col=2)

        # ---- Bottom row: Strain + Stress (shared y-axes) ----
        y_range_min = s.y_bottom - 20
        y_range_max = s.y_top + 20

        fig.update_yaxes(title_text="y (mm)", range=[y_range_min, y_range_max], row=2, col=1)
        fig.update_yaxes(matches="y3", showticklabels=False, row=2, col=2)

        fig.update_xaxes(title_text="Strain (‰)", row=2, col=1)

        stress_x_min, stress_x_max = self._stress_x_range(s)
        fig.update_xaxes(
            title_text="Stress (MPa)",
            range=[stress_x_min, stress_x_max],
            row=2, col=2,
        )

        # Two-column annotation in top-right quadrant, both anchored at same top y.
        ann_left_x, ann_right_x, ann_y = self._top_right_annotation_anchors(fig)
        left_text, right_text = self._build_annotation_columns(s, include_failure_banner=False)
        if left_text:
            fig.add_annotation(
                xref="paper",
                yref="paper",
                x=ann_left_x,
                y=ann_y,
                text=left_text,
                showarrow=False,
                font=dict(size=10),
                align="left",
                xanchor="left",
                yanchor="top",
            )
        if right_text:
            fig.add_annotation(
                xref="paper",
                yref="paper",
                x=ann_right_x,
                y=ann_y,
                text=right_text,
                showarrow=False,
                font=dict(size=10),
                align="left",
                xanchor="left",
                yanchor="top",
            )

        # Scale and position the concrete colorbar relative to the section subplot.
        self._position_section_colorbar(fig)


    # -----------------------------
    # Helpers
    # -----------------------------
    @staticmethod
    def _axis_domain(layout: Any, axis_name: str, fallback: tuple[float, float]) -> tuple[float, float]:
        """Return axis domain in paper coordinates with a safe fallback."""
        axis = getattr(layout, axis_name, None)
        domain = getattr(axis, "domain", None) if axis is not None else None
        if isinstance(domain, (tuple, list)) and len(domain) == 2:
            try:
                a = float(domain[0])
                b = float(domain[1])
                if np.isfinite(a) and np.isfinite(b) and b > a:
                    return (a, b)
            except (TypeError, ValueError):
                pass
        return fallback

    def _top_right_annotation_anchor(self, fig: Any) -> tuple[float, float]:
        """Get paper-coordinate anchor for top-left of top-right quadrant."""
        left_x, _, y = self._top_right_annotation_anchors(fig)
        return (left_x, y)

    def _top_right_annotation_anchors(self, fig: Any) -> tuple[float, float, float]:
        """Get paper-coordinate anchors for left/right annotation columns."""
        layout = getattr(fig, "layout", None)
        if layout is None:
            return (0.52, 0.63, 0.98)

        x0, x1 = self._axis_domain(layout, "xaxis2", (0.52, 0.98))
        y0, y1 = self._axis_domain(layout, "yaxis2", (0.56, 0.98))

        span_x = max(1e-9, x1 - x0)
        pad_x = max(0.01, 0.04 * span_x)
        gap_x = max(0.008, 0.02 * span_x)
        pad_y = 0.01 * (y1 - y0)

        left_x = x0 + pad_x
        # Keep columns between previous and current spacing (midpoint adjustment).
        old_right_x = min(x1 - pad_x, x0 + 0.5 * span_x + gap_x)
        half_sep = 0.5 * (old_right_x - left_x)
        min_sep = max(0.006, 0.02 * span_x)
        mid_sep = 0.5 * ((old_right_x - left_x) + half_sep)
        right_x = min(x1 - pad_x, left_x + max(min_sep, mid_sep))
        top_y = y1 - pad_y
        return (left_x, right_x, top_y)

    def _position_section_colorbar(self, fig: Any) -> None:
        """
        Position concrete stress colorbar to the right of the section subplot and
        scale it to that subplot's height.
        """
        layout = getattr(fig, "layout", None)
        if layout is None:
            return

        x0, x1 = self._axis_domain(layout, "xaxis", (0.0, 0.45))
        y0, y1 = self._axis_domain(layout, "yaxis", (0.55, 1.0))

        colorbar_cfg = dict(
            x=min(0.98, x1 + 0.012),
            y=0.5 * (y0 + y1),
            len=max(0.08, 0.95 * (y1 - y0)),
            lenmode="fraction",
            xanchor="left",
            yanchor="middle",
            thickness=15,
            title=dict(text="σ<br>(MPa)", side="bottom"),
        )

        # Real Plotly figure path.
        if hasattr(fig, "for_each_trace"):
            def _update_trace(trace: Any) -> None:
                if getattr(trace, "name", None) != "Concrete":
                    return

                marker = getattr(trace, "marker", None)
                marker_cb = getattr(marker, "colorbar", None) if marker is not None else None
                if marker_cb is not None:
                    for key, value in colorbar_cfg.items():
                        setattr(marker_cb, key, value)

                trace_cb = getattr(trace, "colorbar", None)
                if trace_cb is not None:
                    for key, value in colorbar_cfg.items():
                        setattr(trace_cb, key, value)

            fig.for_each_trace(_update_trace)
            return

        # Fallback path for lightweight fake figures used in tests.
        traces = getattr(fig, "traces", None)
        if traces is None:
            return

        for item in traces:
            trace = item[0] if isinstance(item, tuple) and item else item
            if not isinstance(trace, dict) or trace.get("name") != "Concrete":
                continue

            marker = trace.get("marker")
            if isinstance(marker, dict):
                marker_cb = marker.get("colorbar")
                if isinstance(marker_cb, dict):
                    marker_cb.update(colorbar_cfg)

            trace_cb = trace.get("colorbar")
            if isinstance(trace_cb, dict):
                trace_cb.update(colorbar_cfg)

    def _add_concrete_filled_field(self, fig: Any, go: Any, s: _StressStrainPlotState, *, row: int, col: int) -> None:
        """
        Render concrete stresses as a filled cell-based field using i/j fibre indices.

        Requires:
        - s.fibre_i, s.fibre_j arrays aligned with s.stresses
        - s.bbox = (min_x, min_y, max_x, max_y)

        Notes:
        - Uses cell centres derived from bbox and inferred nx/ny from max(i/j)+1.
        - Cells outside the section remain NaN, so they don't render.
        """
        if getattr(s, "fibre_i", None) is None or getattr(s, "fibre_j", None) is None:
            raise ValueError(
                "section_render='filled' requires fibre i/j indices on the diagram.\n"
                "Expose aligned arrays on the diagram, e.g. self.diagram._fibre_i and self.diagram._fibre_j."
            )

        fi = np.asarray(s.fibre_i)
        fj = np.asarray(s.fibre_j)

        # Concrete-only i/j + stresses
        ci = fi[s.conc_mask]
        cj = fj[s.conc_mask]
        conc_stresses = s.stresses[s.conc_mask]

        if ci.size == 0 or cj.size == 0:
            return

        # Infer base grid size from max indices (+1)
        nx_base = int(np.max(ci)) + 1
        ny_base = int(np.max(cj)) + 1
        if nx_base <= 0 or ny_base <= 0:
            return

        # Upsample the display grid to reduce staircase artifacts on curved boundaries.
        upsample = 6
        nx = int(nx_base * upsample)
        ny = int(ny_base * upsample)

        min_x, min_y, max_x, max_y = s.bbox
        dx = (max_x - min_x) / float(nx)
        dy = (max_y - min_y) / float(ny)
        if dx <= 0.0 or dy <= 0.0:
            return

        # Cell centres
        x_centres = (min_x + (np.arange(nx) + 0.5) * dx).astype(float)
        y_centres = (min_y + (np.arange(ny) + 0.5) * dy).astype(float)

        # Build Z grid: shape (ny, nx). Keep NaN outside section so nothing renders there.
        Z = np.full((ny, nx), np.nan, dtype=float)

        # Concrete stress in this 2D viewer varies only with y (linear strain field in depth).
        if s.h > 0.0:
            row_strains = s.eps_bottom + (s.eps_top - s.eps_bottom) * (
                (y_centres - s.y_bottom) / s.h
            )
        else:
            row_strains = np.full_like(y_centres, s.eps_bottom, dtype=float)

        row_stresses = self._get_concrete_stress_for_plot(row_strains)

        # Clip each horizontal band to the exact section intersection at that y.
        for j, y_mid in enumerate(y_centres):
            segs = self._section_horizontal_segments_at_y(float(y_mid))
            if not segs:
                continue
            sig = float(row_stresses[j])
            for xa, xb in segs:
                if xb < xa:
                    xa, xb = xb, xa

                # Indices of cells whose horizontal span overlaps [xa, xb].
                # This avoids inset gaps at curved boundaries.
                i0 = int(np.floor((xa - min_x) / dx))
                i1 = int(np.ceil((xb - min_x) / dx)) - 1
                if i1 < 0 or i0 > nx - 1:
                    continue
                i0 = max(i0, 0)
                i1 = min(i1, nx - 1)
                if i1 >= i0:
                    Z[j, i0 : i1 + 1] = sig

        finite = np.isfinite(Z)
        if not np.any(finite):
            return

        cmin_val, cmax_val, colorscale = self._concrete_colorscale(Z[finite])

        # Contour with heatmap coloring gives the "filled" look without triangulation
        fig.add_trace(
            go.Contour(
                x=x_centres,
                y=y_centres,
                z=Z,
                contours=dict(coloring="heatmap", showlines=False),
                colorscale=colorscale,
                zmin=cmin_val,
                zmax=cmax_val,
                showscale=True,
                colorbar=dict(
                    title=dict(text="σ<br>(MPa)", side="bottom"),
                    thickness=15,
                ),
                hovertemplate="x: %{x:.1f} mm<br>y: %{y:.1f} mm<br>σ: %{z:.2f} MPa<extra>Concrete</extra>",
                name="Concrete",
                showlegend=False,
            ),
            row=row,
            col=col,
        )

    def _section_horizontal_segments_at_y(self, y: float) -> list[tuple[float, float]]:
        """
        Return x-intervals [x0, x1] where the section outline intersects a horizontal
        line at elevation y. Handles Polygon / MultiPolygon and all intersection outputs.
        """
        try:
            from shapely.geometry import LineString
        except ImportError as e:
            raise ImportError("Shapely is required for section clipping.") from e

        geom = self.diagram.section.outline
        x_min, _, x_max, _ = self.diagram.section.get_bounding_box()

        # make the cut line slightly longer than bbox
        cut = LineString([(float(x_min) - 1.0, float(y)), (float(x_max) + 1.0, float(y))])
        inter = geom.intersection(cut)

        segs: list[tuple[float, float]] = []

        def add_geom(g: object) -> None:
            # Shapely geometry types expose .geom_type at runtime
            gt = getattr(g, "geom_type", None)

            if gt == "LineString":
                coords = getattr(g, "coords", None)
                if coords is None:
                    return
                xs = [float(p[0]) for p in coords]
                if xs:
                    segs.append((min(xs), max(xs)))
                return

            # MultiLineString / GeometryCollection / MultiPolygon intersection cases
            parts = getattr(g, "geoms", None)
            if parts is not None:
                for part in parts:
                    add_geom(part)
                return

            # Points happen if the cut just kisses a corner — ignore for drawing
            return

        add_geom(inter)

        # merge/clean
        if not segs:
            return []

        segs.sort()
        merged: list[tuple[float, float]] = [segs[0]]
        for a, b in segs[1:]:
            pa, pb = merged[-1]
            if a <= pb + 1e-6:
                merged[-1] = (pa, max(pb, b))
            else:
                merged.append((a, b))

        return merged

    @staticmethod
    def _weighted_centroid_y(forces: np.ndarray, y: np.ndarray, mask: np.ndarray, *, tol: float = 1e-9) -> float | None:
        if not np.any(mask):
            return None
        f = forces[mask]
        denom = float(np.sum(f))
        if abs(denom) < tol:
            return None
        return float(np.sum(f * y[mask]) / denom)

    @staticmethod
    def _neutral_axis(*, eps_top: float, eps_bottom: float, y_top: float, y_bottom: float, h: float) -> Tuple[float | None, bool]:
        # strain(y) = eps_bottom + (eps_top-eps_bottom)*(y - y_bottom)/h
        # set strain=0 => y = y_bottom - eps_bottom*h/(eps_top-eps_bottom)
        if abs(eps_top - eps_bottom) <= 1e-12:
            return None, False
        y_na = y_bottom - eps_bottom * h / (eps_top - eps_bottom)
        na_in_section = (y_bottom <= y_na <= y_top)
        return float(y_na), bool(na_in_section)

    def _interpolate_concrete_stress_profile(
        self,
        s: _StressStrainPlotState,
        n_points: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate a smooth concrete stress profile by interpolating across section height.

        The fibre mesh may have sparse points in the compression zone (e.g., only 6-9 points
        if compression is 20-30% of section depth with 30 fibres total). This method generates
        a dense set of points for smoother visualization.

        Args:
            s: Plot state containing strain endpoints and section geometry
            n_points: Number of interpolation points across section height

        Returns:
            (y_coords, strains, stresses): Arrays of y-coordinates, strains, and stresses (MPa)
        """
        # Generate dense y-coordinates across section height
        y_coords = np.linspace(s.y_bottom, s.y_top, n_points)

        # Compute strain at each y using linear interpolation
        # strain(y) = eps_bottom + (eps_top - eps_bottom) * (y - y_bottom) / h
        h = s.h
        if h > 0:
            strains = s.eps_bottom + (s.eps_top - s.eps_bottom) * (y_coords - s.y_bottom) / h
        else:
            strains = np.full_like(y_coords, s.eps_bottom)

        # Compute concrete stress using the same option-aware path as the solver.
        stresses = self._get_concrete_stress_for_plot(strains)

        return y_coords, strains, stresses

    def _get_concrete_stress_for_plot(self, strains: np.ndarray) -> np.ndarray:
        """
        Concrete stress for plotting, aligned with diagram analysis options.

        Uses the diagram's option-aware concrete stress path when available
        (e.g. tension stiffening, crack-to-NA policies). Falls back to the
        constitutive model directly for lightweight test doubles.
        """
        stress_fn = getattr(self.diagram, "_concrete_stress_with_options", None)
        if callable(stress_fn):
            return np.asarray(stress_fn(strains), dtype=float)
        return np.asarray(self.diagram.concrete_model.get_stress_array(strains), dtype=float)

    def _get_outline_xy(self) -> Tuple[list[float], list[float]]:
        """
        Return section outline x/y suitable for Plotly.
        Assumes section.outline is a shapely Polygon.
        """
        outline_coords = list(self.diagram.section.outline.exterior.coords)
        outline_x = [float(c[0]) for c in outline_coords]
        outline_y = [float(c[1]) for c in outline_coords]
        return outline_x, outline_y

    @staticmethod
    def _concrete_colorscale(conc_stresses: np.ndarray) -> Tuple[float, float, Any]:
        """
        Returns (cmin, cmax, colorscale) for concrete stress coloring.
        If no tension: [0..max] with white->red. If tension exists: symmetric with RdBu_r.
        """
        if conc_stresses.size == 0:
            return 0.0, 1.0, [[0, "white"], [1, "red"]]

        smin = float(np.min(conc_stresses))
        smax = float(np.max(conc_stresses))

        if smin >= 0.0:
            cmin_val = 0.0
            cmax_val = smax if smax > 0.0 else 1.0
            colorscale = [[0, "white"], [1, "red"]]
            return cmin_val, cmax_val, colorscale

        abs_max = max(abs(smin), abs(smax))
        return -abs_max, abs_max, "RdBu_r"

    @staticmethod
    def _concrete_stress_range(stresses: np.ndarray, conc_mask: np.ndarray) -> Tuple[float, float]:
        if np.any(conc_mask):
            conc = stresses[conc_mask]
            max_pos = max(0.0, float(np.max(conc)))
            min_neg = min(0.0, float(np.min(conc)))
            return max_pos, min_neg
        return 1.0, -1.0

    @staticmethod
    def _force_scale(
        *,
        max_stress_pos: float,
        min_stress_neg: float,
        F_c_comp: float,
        F_c_tens: float,
        F_s_comp: float,
        F_s_tens: float,
    ) -> float:
        max_force = max(abs(F_c_comp), abs(F_c_tens), abs(F_s_comp), abs(F_s_tens), 1.0)
        stress_range = max(max_stress_pos, abs(min_stress_neg), 1.0)
        return (stress_range * 0.5) / max_force

    @staticmethod
    def _add_resultant_arrow(
        fig: Any,
        go: Any,
        *,
        row: int,
        col: int,
        name: str,
        force: float,
        y: float | None,
        force_scale: float,
        line_color: str,
        tip_symbol: str,
        extra: str,
        tol: float = 0.001,
    ) -> None:
        if y is None or abs(force) <= tol:
            return

        arrow_x = float(force * force_scale)

        fig.add_trace(
            go.Scatter(
                x=[0.0, arrow_x],
                y=[y, y],
                mode="lines",
                line=dict(color=line_color, width=3),
                name=f"{name} ({force:.0f} kN)",
                legendgroup=name,
                hovertemplate=f"{name} = {force:.1f} kN<br>y = {y:.1f} mm<extra>{extra}</extra>",
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=[arrow_x],
                y=[y],
                mode="markers",
                marker=dict(size=14, color=line_color, symbol=tip_symbol),
                showlegend=False,
                legendgroup=name,
                hoverinfo="skip",
            ),
            row=row,
            col=col,
        )

    def _build_annotation_text(self, s: _StressStrainPlotState) -> str:
        left, right = self._build_annotation_columns(s, include_failure_banner=True)
        if left and right:
            return f"{left}<br><br>{right}"
        return left or right

    def _build_annotation_columns(
        self,
        s: _StressStrainPlotState,
        *,
        include_failure_banner: bool = True,
    ) -> tuple[str, str]:
        indent = "&nbsp;&nbsp;&nbsp;&nbsp;"
        block_gap = "<br><br>"
        left_blocks: list[str] = []
        right_blocks: list[str] = []

        # Show prominent warning if section fails
        if s.section_failed and include_failure_banner:
            left_blocks.append(
                '<span style="color:red; font-size:12px"><b>⚠ SECTION FAILS: '
                'Applied loads exceed capacity. Strains shown are bounded '
                'approximation NOT in equilibrium with applied forces.</b></span>'
            )

        left_blocks.append(
            "<b>Load Case:</b><br>"
            f"{indent}M<sub>Ed</sub> = {s.M_Ed:.1f} kN·m<br>"
            f"{indent}N<sub>Ed</sub> = {s.N_Ed:.1f} kN<br>"
        )
        if s.M_Rd_pos is not None and s.N_Rd is not None:
            right_blocks.append(
                "<b>Capacity:</b><br>"
                f"{indent}M<sub>Rd</sub> = {s.M_Rd_pos:.1f} kN·m<br>"
                f"{indent}N<sub>Rd</sub> = {s.N_Rd:.1f} kN<br>"
            )

        left_blocks.append(
            "<b>Strains:</b><br>"
            f"{indent}ε<sub>top</sub> = {s.eps_top*1000:.3f}‰<br>"
            f"{indent}ε<sub>bot</sub> = {s.eps_bottom*1000:.3f}‰<br>"
        )
        compression_at_top = s.eps_top >= s.eps_bottom
        y_comp_face = s.y_top if compression_at_top else s.y_bottom
        section_info_lines: list[str] = []
        if s.utilisation is not None:
            section_info_lines.append(f"<b>Utilisation:</b> {100.0 * s.utilisation:.1f}%")
        if s.y_na is not None:
            x_na = (y_comp_face - s.y_na) if compression_at_top else (s.y_na - y_comp_face)
            na_line = f"<b>Neutral Axis:</b> x = {x_na:.1f} mm"
            if not s.na_in_section:
                na_line += " (outside section)"
            section_info_lines.append(na_line)
        if s.z is not None:
            section_info_lines.append(f"<b>Lever Arm:</b> z = {s.z:.1f} mm")
            if s.y_T is not None:
                d_eff = abs(y_comp_face - s.y_T)
                section_info_lines.append(f"<b>Effective Depth:</b> d = {d_eff:.1f} mm")
        if section_info_lines:
            right_blocks.append("<br>".join(section_info_lines))

        parts = []
        if abs(s.F_c_comp) > 0.001:
            parts.append(f"F<sub>cc</sub> = {s.F_c_comp:.1f} kN")
        if abs(s.F_c_tens) > 0.001:
            parts.append(f"F<sub>ct</sub> = {s.F_c_tens:.1f} kN")
        if abs(s.F_s_comp) > 0.001:
            parts.append(f"F<sub>sc</sub> = {s.F_s_comp:.1f} kN")
        if abs(s.F_s_tens) > 0.001:
            parts.append(f"F<sub>st</sub> = {s.F_s_tens:.1f} kN")

        resultants_block = "<b>Resultants:</b><br>"
        if parts:
            for part in parts:
                resultants_block += f"{indent}{part}<br>"
        else:
            resultants_block += f"{indent}No forces<br>"
        left_blocks.append(resultants_block)

        F_total = s.F_c_comp + s.F_c_tens + s.F_s_comp + s.F_s_tens
        right_blocks.append(
            "<b>ΣF:</b><br>"
            f"{indent}{F_total:.1f} kN<br>"
            f"{indent}(≈ N<sub>Ed</sub> = {s.N_Ed:.1f} kN)<br>"
        )

        # Show equilibrium error if section failed
        if s.section_failed:
            left_blocks.append(
                '<span style="color:red"><b>Equilibrium Error:</b><br>'
                f"{indent}ΔN = {s.equilibrium_error_N:.1f} kN<br>"
                f"{indent}ΔM = {s.equilibrium_error_M:.1f} kN·m</span>"
            )
            right_blocks.append(
                '<br><span style="color:red"><b>Achieved:</b><br>'
                f"{indent}N = {s.achieved_N:.1f} kN<br>"
                f"{indent}M = {s.achieved_M:.1f} kN·m</span>"
            )

        return (block_gap.join(left_blocks), block_gap.join(right_blocks))

    def _stress_x_range(self, s: _StressStrainPlotState) -> Tuple[float, float]:
        # base range from concrete stresses
        x_min = s.min_stress_neg if s.min_stress_neg < 0.0 else 0.0
        x_max = s.max_stress_pos if s.max_stress_pos > 0.0 else 0.0

        # include arrow extents if present
        arrow_x_vals: list[float] = []
        for F, y in (
            (s.F_c_comp, s.y_c_comp),
            (s.F_c_tens, s.y_c_tens),
            (s.F_s_comp, s.y_s_comp),
            (s.F_s_tens, s.y_s_tens),
        ):
            if y is not None and abs(F) > 0.001:
                arrow_x_vals.append(float(F * s.force_scale))

        if arrow_x_vals:
            x_min = min(x_min, min(arrow_x_vals))
            x_max = max(x_max, max(arrow_x_vals))

        # pad
        rng = x_max - x_min
        pad = rng * 0.1 if rng > 0.0 else 1.0
        return (x_min - pad, x_max + pad)
