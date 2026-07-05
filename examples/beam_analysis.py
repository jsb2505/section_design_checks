"""
Beam analysis module for demonstrating tension shift logic.

This module provides simple beam analysis classes for uniformly distributed loads (UDL)
with interactive Plotly-based visualization. Primarily used for educational purposes
to demonstrate the EC2 tension shift rule.

Classes:
    BeamNode: Dataclass storing position, moment, and shear at a node
    Beam: Base class for beam analysis with UDL
    SimplySupportedBeam: Simply supported beam (pin-roller)
    FixedPinnedBeam: Fixed-pinned beam (cantilever with roller)

Functions:
    plot_beam_diagrams: Interactive Plotly plots for M and V diagrams
    plot_tension_shift_comparison: Compare original vs shifted moments
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, TYPE_CHECKING

from materials.core.units import MomentUnit, to_knm
import numpy as np

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

if TYPE_CHECKING:
    from materials.reinforced_concrete.analysis.interaction_diagram import MNInteractionDiagram
    from materials.reinforced_concrete.materials.rebar import ShearRebar


@dataclass
class BeamNode:
    """
    Represents a node along the beam with position and internal forces.

    Attributes:
        x: Position along beam from left support (m)
        M: Bending moment at node (kN·m), positive = sagging
        V: Shear force at node (kN), positive = clockwise rotation
    """
    x: float
    M: float
    V: float


@dataclass
class Beam:
    """
    Base class for beam analysis with uniformly distributed load.

    Sign conventions:
        - Positive moment: Sagging (tension at bottom)
        - Negative moment: Hogging (tension at top)
        - Positive shear: Clockwise rotation on element

    Attributes:
        length_m: Beam span length (m)
        udl_kN_m: Uniformly distributed load (kN/m), positive = downward
        n_nodes: Number of nodes along beam for analysis
        nodes: List of BeamNode objects after analysis
    """
    length_m: float
    udl_kN_m: float
    n_nodes: int = 50
    nodes: List[BeamNode] = field(default_factory=list, init=False)

    def __post_init__(self):
        """Run analysis after initialization."""
        self._analyze()

    def _analyze(self) -> None:
        """Calculate M and V at each node. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement _analyze()")

    def get_max_positive_moment(self) -> Tuple[float, float]:
        """
        Get maximum positive (sagging) moment and its location.

        Returns:
            Tuple of (max_M, x) or (0.0, 0.0) if no positive moments
        """
        positive_nodes = [n for n in self.nodes if n.M > 0]
        if not positive_nodes:
            return (0.0, 0.0)
        max_node = max(positive_nodes, key=lambda n: n.M)
        return (max_node.M, max_node.x)

    def get_max_negative_moment(self) -> Tuple[float, float]:
        """
        Get maximum negative (hogging) moment and its location.

        Returns:
            Tuple of (max_M, x) or (0.0, 0.0) if no negative moments
        """
        negative_nodes = [n for n in self.nodes if n.M < 0]
        if not negative_nodes:
            return (0.0, 0.0)
        min_node = min(negative_nodes, key=lambda n: n.M)
        return (min_node.M, min_node.x)

    def get_M_cap_positive(self) -> float:
        """Get M_cap for positive moment region (max positive moment magnitude)."""
        return abs(self.get_max_positive_moment()[0])

    def get_M_cap_negative(self) -> float:
        """Get M_cap for negative moment region (max negative moment magnitude)."""
        return abs(self.get_max_negative_moment()[0])

    @property
    def x_values(self) -> np.ndarray:
        """Array of x positions along beam (m)."""
        return np.array([n.x for n in self.nodes])

    @property
    def M_values(self) -> np.ndarray:
        """Array of moments along beam (kN·m)."""
        return np.array([n.M for n in self.nodes])

    @property
    def V_values(self) -> np.ndarray:
        """Array of shear forces along beam (kN)."""
        return np.array([n.V for n in self.nodes])


@dataclass
class SimplySupportedBeam(Beam):
    """
    Simply supported beam with pin-roller supports.

    Reactions:
        R_A = R_B = w*L/2

    Moment equation:
        M(x) = R_A*x - w*x²/2 = w*x*(L-x)/2

    Shear equation:
        V(x) = R_A - w*x = w*(L/2 - x)

    Maximum moment at midspan:
        M_max = w*L²/8
    """

    def _analyze(self) -> None:
        """Calculate M and V at each node for simply supported beam."""
        L = self.length_m
        w = self.udl_kN_m

        # Reactions
        R_A = w * L / 2  # Left support reaction

        # Generate nodes
        x_positions = np.linspace(0, L, self.n_nodes)
        self.nodes = []

        for x in x_positions:
            # Moment: M = R_A*x - w*x²/2
            M = R_A * x - w * x**2 / 2

            # Shear: V = R_A - w*x (positive = clockwise)
            V = R_A - w * x

            self.nodes.append(BeamNode(x=x, M=M, V=V))


@dataclass
class FixedPinnedBeam(Beam):
    """
    Fixed-pinned beam (propped cantilever).

    Fixed at left (A), pinned at right (B).

    Using superposition/compatibility:
        R_B = 3*w*L/8  (reaction at pinned end)
        R_A = 5*w*L/8  (reaction at fixed end)
        M_A = -w*L²/8  (fixed end moment, hogging)

    Moment equation:
        M(x) = M_A + R_A*x - w*x²/2

    Shear equation:
        V(x) = R_A - w*x

    Maximum positive moment at x = 5L/8:
        M_max_pos = 9*w*L²/128
    """

    def _analyze(self) -> None:
        """Calculate M and V at each node for fixed-pinned beam."""
        L = self.length_m
        w = self.udl_kN_m

        # Reactions for fixed-pinned beam with UDL
        R_A = 5 * w * L / 8  # Reaction at fixed end
        M_A = -w * L**2 / 8   # Fixed end moment (negative = hogging)

        # Generate nodes
        x_positions = np.linspace(0, L, self.n_nodes)
        self.nodes = []

        for x in x_positions:
            # Moment: M = M_A + R_A*x - w*x²/2
            M = M_A + R_A * x - w * x**2 / 2

            # Shear: V = R_A - w*x
            V = R_A - w * x

            self.nodes.append(BeamNode(x=x, M=M, V=V))


def _check_plotly() -> None:
    """Raise ImportError if plotly is not available."""
    if not PLOTLY_AVAILABLE:
        raise ImportError(
            "plotly is required for interactive beam diagrams. "
            "Install with: pip install plotly"
        )


def plot_beam_diagrams(
    beam: Beam,
    title: str = "Beam Analysis",
    show_grid: bool = True,
    height: int = 600,
) -> "go.Figure":
    """
    Create interactive Plotly plots for bending moment and shear force diagrams.

    Args:
        beam: Analyzed Beam object
        title: Plot title
        show_grid: Whether to show grid lines
        height: Figure height in pixels

    Returns:
        Plotly Figure object with M and V subplots
    """
    _check_plotly()

    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Bending Moment Diagram", "Shear Force Diagram"),
        vertical_spacing=0.12,
    )

    # Moment diagram - convention: plot negative values upward (hogging at top)
    # For structural engineers, sagging is typically shown below the beam
    fig.add_trace(
        go.Scatter(
            x=beam.x_values,
            y=-beam.M_values,  # Negative so sagging plots below axis
            mode="lines",
            name="Bending Moment",
            line=dict(color="blue", width=2),
            fill="tozeroy",
            fillcolor="rgba(0, 100, 255, 0.2)",
            hovertemplate=(
                "<b>Position:</b> %{x:.3f} m<br>"
                "<b>Moment:</b> %{customdata:.2f} kN·m<br>"
                "<extra></extra>"
            ),
            customdata=beam.M_values,
        ),
        row=1, col=1
    )

    # Add markers at nodes for better hover interaction
    fig.add_trace(
        go.Scatter(
            x=beam.x_values,
            y=-beam.M_values,
            mode="markers",
            name="Nodes",
            marker=dict(color="blue", size=5),
            hovertemplate=(
                "<b>Position:</b> %{x:.3f} m<br>"
                "<b>Moment:</b> %{customdata:.2f} kN·m<br>"
                "<extra></extra>"
            ),
            customdata=beam.M_values,
            showlegend=False,
        ),
        row=1, col=1
    )

    # Shear diagram
    fig.add_trace(
        go.Scatter(
            x=beam.x_values,
            y=beam.V_values,
            mode="lines",
            name="Shear Force",
            line=dict(color="red", width=2),
            fill="tozeroy",
            fillcolor="rgba(255, 0, 0, 0.2)",
            hovertemplate=(
                "<b>Position:</b> %{x:.3f} m<br>"
                "<b>Shear:</b> %{y:.2f} kN<br>"
                "<extra></extra>"
            ),
        ),
        row=2, col=1
    )

    # Add markers for shear
    fig.add_trace(
        go.Scatter(
            x=beam.x_values,
            y=beam.V_values,
            mode="markers",
            marker=dict(color="red", size=5),
            hovertemplate=(
                "<b>Position:</b> %{x:.3f} m<br>"
                "<b>Shear:</b> %{y:.2f} kN<br>"
                "<extra></extra>"
            ),
            showlegend=False,
        ),
        row=2, col=1
    )

    # Update layout
    fig.update_layout(
        title=dict(text=title, x=0.5),
        height=height,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        hovermode="closest",
    )

    # Update axes
    fig.update_xaxes(title_text="Position along beam (m)", row=2, col=1)
    fig.update_yaxes(title_text="Moment (kN·m) [↑ hogging, ↓ sagging]", row=1, col=1)
    fig.update_yaxes(title_text="Shear (kN)", row=2, col=1)

    if show_grid:
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")

    # Add zero line
    fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=1, row=1, col=1)
    fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=1, row=2, col=1)

    return fig


def plot_tension_shift_comparison(
    beam: Beam,
    shifted_moments: np.ndarray,
    shift_distances: np.ndarray | None = None,
    title: str = "Tension Shift Comparison",
    height: int = 500,
) -> "go.Figure":
    """
    Create interactive comparison plot of original vs shifted moments.

    Args:
        beam: Analyzed Beam object with original moments
        shifted_moments: Array of shifted moment values (kN·m)
        shift_distances: Optional array of a_l values at each node (mm)
        title: Plot title
        height: Figure height in pixels

    Returns:
        Plotly Figure object comparing original and shifted moments
    """
    _check_plotly()

    fig = go.Figure()

    # Original moment envelope
    hover_orig = (
        "<b>Position:</b> %{x:.3f} m<br>"
        "<b>Original M:</b> %{customdata:.2f} kN·m<br>"
        "<extra></extra>"
    )

    fig.add_trace(
        go.Scatter(
            x=beam.x_values,
            y=-beam.M_values,  # Negative for structural convention
            mode="lines",
            name="Original Moment",
            line=dict(color="blue", width=2),
            fill="tozeroy",
            fillcolor="rgba(0, 100, 255, 0.15)",
            hovertemplate=hover_orig,
            customdata=beam.M_values,
        )
    )

    # Shifted moment envelope
    if shift_distances is not None:
        hover_shifted = (
            "<b>Position:</b> %{x:.3f} m<br>"
            "<b>Shifted M:</b> %{customdata[0]:.2f} kN·m<br>"
            "<b>Shift a_l:</b> %{customdata[1]:.1f} mm<br>"
            "<extra></extra>"
        )
        customdata_shifted = np.column_stack([shifted_moments, shift_distances])
    else:
        hover_shifted = (
            "<b>Position:</b> %{x:.3f} m<br>"
            "<b>Shifted M:</b> %{customdata:.2f} kN·m<br>"
            "<extra></extra>"
        )
        customdata_shifted = shifted_moments

    fig.add_trace(
        go.Scatter(
            x=beam.x_values,
            y=-shifted_moments,  # Negative for structural convention
            mode="lines",
            name="Shifted Moment (EC2 §9.2.1.3)",
            line=dict(color="orange", width=2, dash="dash"),
            hovertemplate=hover_shifted,
            customdata=customdata_shifted,
        )
    )

    # Add markers for interaction
    fig.add_trace(
        go.Scatter(
            x=beam.x_values,
            y=-beam.M_values,
            mode="markers",
            marker=dict(color="blue", size=4),
            hovertemplate=hover_orig,
            customdata=beam.M_values,
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=beam.x_values,
            y=-shifted_moments,
            mode="markers",
            marker=dict(color="orange", size=4),
            hovertemplate=hover_shifted,
            customdata=customdata_shifted,
            showlegend=False,
        )
    )

    # Layout
    fig.update_layout(
        title=dict(text=title, x=0.5),
        height=height,
        xaxis_title="Position along beam (m)",
        yaxis_title="Moment (kN·m) [↑ hogging, ↓ sagging]",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        hovermode="closest",
        showlegend=True,
    )

    # Grid and zero line
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")
    fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=1)

    return fig


def calculate_tension_shift_envelope_simplified(
    beam: Beam,
    effective_depth: float,
    lever_arm: float | None = None,
    cot_theta: float | None = None,
    stirrup_angle_degrees: float = 90.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate tension-shifted moments along a beam using EC2 §9.2.1.3.

    This is a SIMPLIFIED implementation for educational/visualization purposes only.
    For actual design calculations, use apply_tension_shift_to_beam() which calls
    the full MNInteractionDiagram.apply_tension_shift() API.

    The shift distance a_l is calculated as:
        - Without shear reinforcement: a_l = d (effective depth)
        - With shear reinforcement: a_l = z(cot θ - cot α)/2

    Args:
        beam: Analysed Beam object
        effective_depth: Effective depth d (mm)
        lever_arm: Internal lever arm z (mm), required if cot_theta provided
        cot_theta: cot(θ) from shear design, None = no shear reinforcement
        stirrup_angle_degrees: Stirrup angle α in degrees (default 90° for vertical)

    Returns:
        Tuple of (shifted_moments, shift_distances) arrays
    """
    # Calculate shift distance
    if cot_theta is not None:
        if lever_arm is None:
            raise ValueError("lever_arm required when cot_theta is provided")
        # EC2 §9.2.1.3: a_l = z(cot θ - cot α)/2
        if stirrup_angle_degrees == 90.0:
            cot_alpha = 0.0
        else:
            import math
            cot_alpha = 1.0 / math.tan(math.radians(stirrup_angle_degrees))
        a_l = lever_arm * (cot_theta - cot_alpha) / 2.0
        a_l = max(a_l, 0.0)  # Ensure non-negative
    else:
        a_l = effective_depth

    # Get M_cap for each sign (max magnitudes)
    M_cap_pos = beam.get_M_cap_positive()
    M_cap_neg = beam.get_M_cap_negative()

    shifted_moments = []
    shift_distances = []

    for node in beam.nodes:
        M_orig = node.M
        V = node.V

        # M_add = |V| * a_l / 1000 (convert mm to m)
        M_add = to_knm(abs(V) * a_l, MomentUnit.NM)  # kN·mm = N·m → kN·m

        # Apply shift: increase magnitude, then cap
        sign = 1 if M_orig >= 0 else -1
        M_cap = M_cap_pos if sign > 0 else M_cap_neg

        # Shifted magnitude, capped at M_cap
        M_shifted_magnitude = min(abs(M_orig) + M_add, M_cap)
        M_shifted = sign * M_shifted_magnitude

        shifted_moments.append(M_shifted)
        shift_distances.append(a_l)

    return np.array(shifted_moments), np.array(shift_distances)


# Keep old name as alias for backwards compatibility
calculate_tension_shift_envelope = calculate_tension_shift_envelope_simplified


def apply_tension_shift_to_beam(
    beam: Beam,
    diagram: "MNInteractionDiagram",
    N_Ed: float = 0.0,
    shear_reinforcement: "ShearRebar | None" = None,
    cot_theta_override: "float | None" = None,
    iterate_z: bool = False,
    use_mechanical_lever_arm: bool = False,
    z_d_upper: float = 0.95,
    z_d_lower: float = 0.65,
    z_d_approx: float = 0.9,
    warn_on_fallback: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply tension shift to all nodes along a beam using the actual API.

    This function calls MNInteractionDiagram.apply_tension_shift() for each node,
    exercising the full tension shift implementation including strain solving for z.

    Args:
        beam: Analysed Beam object with M and V at each node
        diagram: MNInteractionDiagram instance for the section
        N_Ed: Axial force (kN), constant along beam (default 0.0)
        shear_reinforcement: ShearRebar instance, or None for no shear reinforcement
        cot_theta_override: Optional user-supplied cot(θ) value. When provided with
            shear_reinforcement, this value is used directly instead of calculating
            cot(θ) from V_Ed and V_Rd,max. Must be in range [1.0, 2.5].
        iterate_z: If True, iterate to find z at each section (more accurate but slower)
        use_mechanical_lever_arm: If True, attempt to compute the rigorous centroid-based
            lever arm from strain analysis. If False (default), use the simplified
            z_d_approx * d approach per EC2 §6.2.3(1).
        z_d_upper: Upper bound for z/d in rigorous mode (default 0.95).
        z_d_lower: Lower bound for z/d in rigorous mode (default 0.65).
        z_d_approx: Approximate z/d ratio for non-rigorous mode (default 0.9).
        warn_on_fallback: If True, emit a warning when the rigorous lever arm
            calculation falls back. Default False to avoid noise.

    Returns:
        Tuple of (shifted_moments, shift_distances, cot_theta_values) arrays
        - shifted_moments: M_design at each node (kN·m)
        - shift_distances: a_l at each node (mm)
        - cot_theta_values: cot(θ) at each node (None if no shear reinforcement)
    """
    # Get M_cap for each sign (max magnitudes from beam analysis)
    M_cap_pos = beam.get_M_cap_positive()
    M_cap_neg = beam.get_M_cap_negative()

    shifted_moments = []
    shift_distances = []
    cot_theta_values = []

    for node in beam.nodes:
        M_Ed = node.M
        V_Ed = node.V

        # Determine M_cap based on sign of moment
        M_cap = M_cap_pos if M_Ed >= 0 else M_cap_neg

        # Call the actual API
        result = diagram.apply_tension_shift(
            M_Ed=M_Ed,
            V_Ed=V_Ed,
            N_Ed=N_Ed,
            M_cap=M_cap,
            shear_reinforcement=shear_reinforcement,
            cot_theta_override=cot_theta_override,
            iterate_z=iterate_z,
            use_mechanical_lever_arm=use_mechanical_lever_arm,
            z_d_upper=z_d_upper,
            z_d_lower=z_d_lower,
            z_d_approx=z_d_approx,
            warn_on_fallback=warn_on_fallback,
        )

        shifted_moments.append(result.M_design)
        shift_distances.append(result.shift_distance_a_l)
        cot_theta_values.append(result.cot_theta)

    return (
        np.array(shifted_moments),
        np.array(shift_distances),
        np.array(cot_theta_values, dtype=object),  # object dtype to allow None
    )


if __name__ == "__main__":
    # Quick demonstration
    print("Beam Analysis Module")
    print("=" * 50)

    # Simply supported beam
    ss_beam = SimplySupportedBeam(length_m=8.0, udl_kN_m=25.0, n_nodes=50)
    print(f"\nSimply Supported Beam (L={ss_beam.length_m}m, w={ss_beam.udl_kN_m}kN/m)")
    print(f"  Max positive moment: {ss_beam.get_max_positive_moment()[0]:.2f} kN·m")
    print(f"  Max negative moment: {ss_beam.get_max_negative_moment()[0]:.2f} kN·m")
    print(f"  Theoretical M_max = wL²/8 = {ss_beam.udl_kN_m * ss_beam.length_m**2 / 8:.2f} kN·m")

    # Fixed-pinned beam
    fp_beam = FixedPinnedBeam(length_m=8.0, udl_kN_m=25.0, n_nodes=50)
    print(f"\nFixed-Pinned Beam (L={fp_beam.length_m}m, w={fp_beam.udl_kN_m}kN/m)")
    print(f"  Max positive moment: {fp_beam.get_max_positive_moment()[0]:.2f} kN·m")
    print(f"  Max negative moment: {fp_beam.get_max_negative_moment()[0]:.2f} kN·m")
    print(f"  Theoretical M_A = -wL²/8 = {-fp_beam.udl_kN_m * fp_beam.length_m**2 / 8:.2f} kN·m")
