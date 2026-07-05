"""
Shear visualization helpers for EC2 shear checks.

Provides plotting routines for comparative studies of:
- cot(theta) sweeps
- shear link angle sweeps
- cot(theta) vs link angle heatmaps
- axial force vs cot(theta) heatmaps
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import radians, sin
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from materials.core.units import ForceUnit, to_kn
from materials.reinforced_concrete.code_checks.ec2_2004.flexure_utils import LoadCase
from materials.reinforced_concrete.code_checks.ec2_2004.shear_utils import (
    calculate_tension_shift,
    find_alpha_cw,
    find_nu_1_factor,
    find_nu_1_factor_note_2,
)
from materials.reinforced_concrete.ndp import get_ndp
from materials.utils.helpers import cot

if TYPE_CHECKING:
    from materials.reinforced_concrete.analysis.interaction_diagram import MNInteractionDiagram
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
    cot_intersection: float | None


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


@dataclass(frozen=True)
class _MNPlotLoadPoint:
    """Normalized demand point plotted on an M-N contour figure."""

    M_Ed: float
    N_Ed: float
    name: str


@dataclass(frozen=True)
class _AxialMomentPlotDomain:
    """Resolved M-N boundary geometry and the masked heatmap sampling grid."""

    m_curve: np.ndarray
    n_curve: np.ndarray
    m_vals: np.ndarray
    n_vals: np.ndarray
    m_edges: np.ndarray
    n_edges: np.ndarray
    center_left: np.ndarray
    center_right: np.ndarray
    valid_mask: np.ndarray


def _as_load_case(load_case: LoadCase | dict[str, Any]) -> LoadCase:
    """Normalize supported load-case inputs to LoadCase."""
    if isinstance(load_case, LoadCase):
        return load_case
    if isinstance(load_case, dict):
        return LoadCase(
            Vz_Ed=float(load_case["V_Ed"]),
            My_Ed=float(load_case.get("M_Ed", 0.0)),
            N_Ed=float(load_case.get("N_Ed", 0.0)),
        )
    raise TypeError("load_case must be a LoadCase or dict with V_Ed/M_Ed/N_Ed keys.")


def _normalize_mn_loadcases(
    loadcases: Sequence[dict[str, Any] | Sequence[float]] | None,
) -> list[_MNPlotLoadPoint]:
    """Normalize supported M/N load-point inputs for contour overlays."""
    if not loadcases:
        return []

    normalized: list[_MNPlotLoadPoint] = []
    for idx, item in enumerate(loadcases):
        default_name = f"Load Case {idx + 1}"
        if isinstance(item, dict):
            normalized.append(
                _MNPlotLoadPoint(
                    M_Ed=float(item["M_Ed"]),
                    N_Ed=float(item["N_Ed"]),
                    name=str(item.get("name", default_name)),
                ),
            )
            continue

        if isinstance(item, Sequence) and not isinstance(item, (str, bytes)) and len(item) == 2:
            normalized.append(
                _MNPlotLoadPoint(
                    M_Ed=float(item[0]),
                    N_Ed=float(item[1]),
                    name=default_name,
                ),
            )
            continue

        raise TypeError(
            "loadcases entries must be dicts with M_Ed/N_Ed keys or 2-item (M_Ed, N_Ed) sequences.",
        )

    return normalized


def _build_axial_moment_plot_domain(
    diagram: MNInteractionDiagram,
    *,
    n_diagram_points: int,
    n_moment: int,
    n_axial: int,
) -> _AxialMomentPlotDomain:
    """Build the M-N plotting boundary and a rectangular grid masked to the envelope."""
    diagram_points = diagram.generate_diagram_points(n_points=n_diagram_points)
    if not diagram_points:
        raise ValueError("Interaction diagram must contain at least one point.")

    m_curve = np.array([float(point.M) for point in diagram_points], dtype=float)
    n_curve = np.array([float(point.N) for point in diagram_points], dtype=float)
    m_vals = np.linspace(float(np.min(m_curve)), float(np.max(m_curve)), max(2, int(n_moment)))
    n_vals = np.linspace(float(np.min(n_curve)), float(np.max(n_curve)), max(2, int(n_axial)))
    m_edges = _build_axis_edges(m_vals)
    n_edges = _build_axis_edges(n_vals)
    center_left = np.full(len(n_vals), np.nan, dtype=float)
    center_right = np.full(len(n_vals), np.nan, dtype=float)
    edge_left = np.full(len(n_edges), np.nan, dtype=float)
    edge_right = np.full(len(n_edges), np.nan, dtype=float)
    valid_mask = np.zeros((len(n_vals), len(m_vals)), dtype=bool)

    for i_n, n_ed in enumerate(n_vals):
        _, m_rd_pos, m_rd_neg = diagram.get_capacity_fixed_n(
            N_Ed=float(n_ed),
            n_points=n_diagram_points,
        )
        if m_rd_pos is None or m_rd_neg is None:
            continue
        center_left[i_n] = min(float(m_rd_neg), float(m_rd_pos))
        center_right[i_n] = max(float(m_rd_neg), float(m_rd_pos))

    for i_n, n_ed in enumerate(n_edges):
        _, m_rd_pos, m_rd_neg = diagram.get_capacity_fixed_n(
            N_Ed=float(n_ed),
            n_points=n_diagram_points,
        )
        if m_rd_pos is None or m_rd_neg is None:
            continue
        edge_left[i_n] = min(float(m_rd_neg), float(m_rd_pos))
        edge_right[i_n] = max(float(m_rd_neg), float(m_rd_pos))

    for i_n, n_ed in enumerate(n_vals):
        left_center = center_left[i_n]
        right_center = center_right[i_n]
        if not np.isfinite(left_center) or not np.isfinite(right_center):
            continue
        left_band = float(np.nanmin(edge_left[i_n : i_n + 2]))
        right_band = float(np.nanmax(edge_right[i_n : i_n + 2]))
        tol = 1e-9 * max(1.0, abs(left_band), abs(right_band))
        for i_m in range(len(m_vals)):
            cell_left = float(m_edges[i_m])
            cell_right = float(m_edges[i_m + 1])
            valid_mask[i_n, i_m] = bool(
                cell_right >= left_band - tol and cell_left <= right_band + tol,
            )

    return _AxialMomentPlotDomain(
        m_curve=m_curve,
        n_curve=n_curve,
        m_vals=m_vals,
        n_vals=n_vals,
        m_edges=m_edges,
        n_edges=n_edges,
        center_left=center_left,
        center_right=center_right,
        valid_mask=valid_mask,
    )


def _build_axis_edges(values: np.ndarray) -> np.ndarray:
    """Return cell-edge coordinates for a monotonic array of cell centers."""
    vals = np.asarray(values, dtype=float)
    if vals.size == 0:
        return np.asarray([], dtype=float)
    if vals.size == 1:
        return np.asarray([vals[0] - 0.5, vals[0] + 0.5], dtype=float)

    midpoints = 0.5 * (vals[:-1] + vals[1:])
    first = vals[0] - 0.5 * (vals[1] - vals[0])
    last = vals[-1] + 0.5 * (vals[-1] - vals[-2])
    return np.concatenate(([first], midpoints, [last]))


def _subdivide_axis(values: np.ndarray, factor: int) -> np.ndarray:
    """Subdivide a cell-edge axis into a denser display-only grid."""
    if factor <= 1:
        return np.asarray(values, dtype=float)

    vals = np.asarray(values, dtype=float)
    segments: list[float] = [float(vals[0])]
    for start, end in zip(vals[:-1], vals[1:]):
        subdivided = np.linspace(float(start), float(end), factor + 1, dtype=float)
        segments.extend(float(v) for v in subdivided[1:])
    return np.asarray(segments, dtype=float)


def _axis_centers_from_edges(edges: np.ndarray) -> np.ndarray:
    """Return cell centers from edge coordinates."""
    edge_vals = np.asarray(edges, dtype=float)
    return 0.5 * (edge_vals[:-1] + edge_vals[1:])


def _build_outside_clip_masks(
    diagram: MNInteractionDiagram,
    *,
    y_edges: np.ndarray,
    x_min: float,
    x_max: float,
    n_diagram_points: int,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Return left and right white mask polygons covering the area outside the boundary."""
    left_bounds: list[float] = []
    right_bounds: list[float] = []
    for y_val in y_edges:
        _, m_rd_pos, m_rd_neg = diagram.get_capacity_fixed_n(
            N_Ed=float(y_val),
            n_points=n_diagram_points,
        )
        if m_rd_pos is None or m_rd_neg is None:
            left_bounds.append(float(x_min))
            right_bounds.append(float(x_max))
            continue
        left_bounds.append(min(float(m_rd_neg), float(m_rd_pos)))
        right_bounds.append(max(float(m_rd_neg), float(m_rd_pos)))

    y = np.asarray(y_edges, dtype=float)
    left = np.asarray(left_bounds, dtype=float)
    right = np.asarray(right_bounds, dtype=float)
    left_mask = (
        np.concatenate(([x_min, x_min], left[::-1], [x_min])),
        np.concatenate(([y[0], y[-1]], y[::-1], [y[0]])),
    )
    right_mask = (
        np.concatenate(([x_max, x_max], right[::-1], [x_max])),
        np.concatenate(([y[0], y[-1]], y[::-1], [y[0]])),
    )
    return left_mask, right_mask


def _get_closed_mn_polyline(
    diagram: MNInteractionDiagram,
    *,
    n_points: int,
) -> list[tuple[float, float]]:
    """Return the M-N diagram polyline as a closed list of (M, N) points."""
    diagram_points = diagram.generate_diagram_points(n_points=n_points)
    if not diagram_points:
        raise ValueError("Interaction diagram must contain at least one point.")

    pts = [(float(point.M), float(point.N)) for point in diagram_points]
    if pts[0] != pts[-1]:
        pts.append(pts[0])
    return pts


def _intersections_with_vertical(
    pts: Sequence[tuple[float, float]],
    M0: float,
    tol: float = 1e-9,
) -> list[float]:
    """Intersect a polyline (M,N) with the vertical line M=M0 and return N intersections."""
    intersections: list[float] = []
    if len(pts) < 2:
        return intersections

    for (M1, N1), (M2, N2) in zip(pts[:-1], pts[1:]):
        if abs(M2 - M1) <= tol:
            if abs(M1 - M0) <= tol:
                intersections.append(float(N1))
                intersections.append(float(N2))
            continue

        if (M0 - M1) * (M0 - M2) > tol:
            continue

        t = (M0 - M1) / (M2 - M1)
        if t < -1e-12 or t > 1.0 + 1e-12:
            continue
        t = min(max(t, 0.0), 1.0)
        Ny = N1 + t * (N2 - N1)
        intersections.append(float(Ny))

    intersections.sort()
    deduped: list[float] = []
    for n_val in intersections:
        if not deduped or abs(n_val - deduped[-1]) > 1e-7:
            deduped.append(n_val)
    return deduped


def _get_force_band_fixed_m(
    pts: Sequence[tuple[float, float]],
    M_Ed: float,
) -> tuple[float | None, float | None, float | None]:
    """Return the vertical-line M-N slice as (M_cap, N_upper, N_lower)."""
    if len(pts) < 4:
        return (None, None, None)

    m_vals = [m for m, _ in pts]
    m_min = float(min(m_vals))
    m_max = float(max(m_vals))
    m_cap = float(min(max(M_Ed, m_min), m_max))

    intersections = _intersections_with_vertical(pts, M0=m_cap, tol=1e-9)
    if not intersections:
        return (None, None, None)

    return (m_cap, float(max(intersections)), float(min(intersections)))


def _build_horizontal_clip_masks(
    *,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    y_lower: float,
    y_upper: float,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Return top and bottom white mask polygons outside a horizontal valid band."""
    lower = float(np.clip(y_lower, y_min, y_max))
    upper = float(np.clip(y_upper, y_min, y_max))
    if lower > upper:
        lower, upper = upper, lower

    top_mask = (
        np.asarray([x_min, x_max, x_max, x_min, x_min], dtype=float),
        np.asarray([upper, upper, y_max, y_max, upper], dtype=float),
    )
    bottom_mask = (
        np.asarray([x_min, x_max, x_max, x_min, x_min], dtype=float),
        np.asarray([y_min, y_min, lower, lower, y_min], dtype=float),
    )
    return top_mask, bottom_mask


def _show_or_save(fig: Any, *, save_path: str | Path | None, show: bool) -> None:
    """Apply standard save/show behaviour used by viewer methods."""
    if save_path:
        fig.write_html(str(save_path))
    if show:
        fig.show()


def _utilization_colorscale(*, zmin: float, zmax: float) -> list[list[float | str]]:
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


def _build_slider_values(
    *,
    value_min: float,
    value_max: float,
    n_points: int,
    step: float | None,
) -> np.ndarray:
    """
    Build monotonic slider values.

    If ``step`` is positive, values are sampled on that uniform increment and
    the upper bound is always included. Otherwise falls back to ``n_points``
    linear spacing.
    """
    v_min = float(value_min)
    v_max = float(value_max)
    if v_min > v_max:
        v_min, v_max = v_max, v_min

    if np.isclose(v_min, v_max):
        return np.asarray([v_min], dtype=float)

    if step is None or step <= 0.0:
        return np.linspace(v_min, v_max, max(2, int(n_points)))

    step_f = float(step)
    n_full = int(np.floor((v_max - v_min) / step_f + 1e-12))
    # Guard against a pathologically small step (relative to the range) allocating
    # an unbounded array; fall back to a capped uniform sampling instead.
    _MAX_SLIDER_POINTS = 10000
    if n_full + 1 > _MAX_SLIDER_POINTS:
        return np.linspace(v_min, v_max, _MAX_SLIDER_POINTS)
    vals = v_min + step_f * np.arange(n_full + 1, dtype=float)
    if vals.size == 0:
        vals = np.asarray([v_min], dtype=float)
    if not np.isclose(vals[-1], v_max):
        vals = np.append(vals, v_max)

    vals[0] = v_min
    vals[-1] = v_max
    return np.round(vals, 12)


def _build_nice_force_slider_values(
    *,
    value_min: float,
    value_max: float,
    target_intervals: int = 50,
) -> np.ndarray:
    """
    Build force slider values using "nice" round increments near a target count.

    Step sizes are chosen from integer engineering increments and the returned
    values are snapped to that step grid so the slider labels stay round.
    """
    v_min = float(value_min)
    v_max = float(value_max)
    if v_min > v_max:
        v_min, v_max = v_max, v_min

    if np.isclose(v_min, v_max):
        return np.asarray([v_min], dtype=float)

    span = v_max - v_min
    target = max(1, int(target_intervals))
    raw_step = span / target

    # Prefer steps that produce integer-valued labels in engineering-style
    # increments such as 5, 10, 20, 50, 100, ...
    nice_bases = np.asarray([1.0, 2.0, 5.0, 10.0], dtype=float)
    raw_exp = int(np.floor(np.log10(raw_step))) if raw_step > 0.0 else 0
    min_step = 5.0 if span >= 5.0 else 1.0

    candidates: list[float] = []
    for exp in range(raw_exp - 2, raw_exp + 3):
        scale = 10.0**exp
        for base in nice_bases:
            step = float(base * scale)
            if step >= min_step:
                candidates.append(step)

    if not candidates:
        candidates = [min_step]

    tol = 1e-9 * max(1.0, abs(v_min), abs(v_max))

    def _aligned_bounds(step: float) -> tuple[float, float]:
        start = float(np.ceil((v_min - tol) / step) * step)
        end = float(np.floor((v_max + tol) / step) * step)
        return (start, end)

    def _interval_count(step: float) -> int:
        start, end = _aligned_bounds(step)
        if end + tol < start:
            return 0
        return int(np.floor((end - start) / step + 1e-12))

    def _coverage_loss(step: float) -> float:
        start, end = _aligned_bounds(step)
        if end + tol < start:
            return float("inf")
        return max(0.0, start - v_min) + max(0.0, v_max - end)

    valid_candidates = [step for step in candidates if _interval_count(step) >= 1]
    if not valid_candidates:
        rounded_mid = 5.0 * round((0.5 * (v_min + v_max)) / 5.0)
        rounded_mid = min(max(rounded_mid, v_min), v_max)
        rounded_mid = round(rounded_mid)
        return np.asarray([float(rounded_mid)], dtype=float)

    def _score(step: float) -> tuple[float, float, float]:
        intervals = _interval_count(step)
        return (abs(intervals - target), _coverage_loss(step), step)

    step_f = min(valid_candidates, key=_score)
    start, end = _aligned_bounds(step_f)
    n_intervals = _interval_count(step_f)
    vals = start + step_f * np.arange(n_intervals + 1, dtype=float)
    return np.round(vals, 12)


def _format_slider_numeric_label(value: float) -> str:
    """Format slider labels without unnecessary trailing zeros."""
    rounded = int(round(float(value)))
    if np.isclose(float(value), float(rounded)):
        return str(rounded)
    return f"{float(value):.6f}".rstrip("0").rstrip(".")


def _build_slider_animation_controls(
    *,
    steps: Sequence[dict[str, Any]],
    currentvalue_prefix: str,
) -> dict[str, list[dict[str, Any]]]:
    """Build standard slider controls with top-right horizontal play/pause buttons."""
    return {
        "sliders": [
            {
                "active": 0,
                "x": 0.1,
                "len": 0.82,
                "currentvalue": {"prefix": currentvalue_prefix},
                "steps": list(steps),
            },
        ],
        "updatemenus": [
            {
                "type": "buttons",
                "showactive": False,
                "direction": "left",
                "x": 1.0,
                "y": 1.16,
                "xanchor": "right",
                "yanchor": "top",
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 200, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                ],
            },
        ],
    }


class ShearViewer:
    """Plotting utilities for ``ShearCheck`` comparative studies."""

    def __init__(self, check: ShearCheck) -> None:
        self.check = check

    def _require_shear_reinforcement(self) -> None:
        if self.check.shear_reinforcement is None:
            raise ValueError("Shear reinforcement is required for shear study plots.")

    def _resolve_plot_diagram(
        self,
    ) -> MNInteractionDiagram:
        """Return the default cached interaction diagram for plotting."""
        return self.check._get_diagram(ignore_compression_steel=False)

    def _build_context(
        self,
        *,
        load_case: LoadCase,
        use_uncracked_V_Rd_c: bool = False,
        ignore_compression_steel: bool = False,
        diagram: MNInteractionDiagram | None = None,
    ) -> _StudyContext:
        """Compute shared parameters for a load case once."""
        V_Ed = abs(float(load_case.V_Ed))
        M_Ed = float(load_case.M_Ed)
        N_Ed = float(load_case.N_Ed)

        if abs(M_Ed) > 1e-6:
            interaction_diagram = diagram
            if interaction_diagram is None:
                interaction_diagram = self.check._get_diagram(ignore_compression_steel)
            eps_top, eps_bottom = interaction_diagram.find_strains_for_MN(M_Ed, N_Ed)
            # Projected strains from strict=False represent the section's
            # failure mode at the nearest envelope point.  The resulting
            # mechanical lever arm is always more meaningful (and more
            # conservative for shear, since z_mech < 0.95d) than the
            # virtual 0.95d fallback — even for load cases outside the
            # M-N envelope.  Using force_virtual=True would create
            # discontinuities wherever the load sweep crosses the
            # envelope boundary.
            force_virtual = False
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
            My_Ed=M_Ed,
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
    ) -> tuple[float, float]:
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
    ) -> float | None:
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
        load_case: LoadCase | dict[str, Any],
        n_points: int,
        cot_theta_min: float | None,
        cot_theta_max: float | None,
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
        load_case: LoadCase | dict[str, Any],
        n_points: int = 60,
        cot_theta_min: float | None = None,
        cot_theta_max: float | None = None,
        use_uncracked_V_Rd_c: bool = False,
        use_note_2: bool = False,
        save_path: str | Path | None = None,
        show: bool = True,
        title: str | None = None,
        width: int = 1000,
        height: int = 560,
    ) -> Any:
        """
        Plot shear-capacity components over a cot(theta) sweep.

        The figure contains demand and capacity references (`V_Ed`, `V_Rd,c`),
        variable capacities (`V_Rd,s`, `V_Rd,max`) and fixed design reference lines
        at the governing code limits for cot(theta).

        Args:
            load_case: Shear demand definition as either ``LoadCase`` or a
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
            TypeError: If ``load_case`` is not a ``LoadCase`` or compatible dict.
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
                line=dict(color="black", dash="solid"),
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
        load_case: LoadCase | dict[str, Any],
        n_points: int = 60,
        cot_theta_min: float | None = None,
        cot_theta_max: float | None = None,
        use_uncracked_V_Rd_c: bool = False,
        use_note_2: bool = False,
        save_path: str | Path | None = None,
        show: bool = True,
        title: str | None = None,
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
            load_case: Shear demand definition as either ``LoadCase`` or a
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
            TypeError: If ``load_case`` is not a ``LoadCase`` or compatible dict.
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
        fig.update_yaxes(title_text="Utilization", secondary_y=False)
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
        load_case: LoadCase | dict[str, Any],
        cot_theta: float | None,
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
        load_case: LoadCase | dict[str, Any],
        cot_theta_min: float | None = None,
        cot_theta_max: float | None = None,
        n_cot: int = 20,
        cot_theta_step: float | None = 0.05,
        angle_min: float = 45.0,
        angle_max: float = 90.0,
        n_points: int = 46,
        use_uncracked_V_Rd_c: bool = False,
        use_note_2: bool = False,
        save_path: str | Path | None = None,
        show: bool = True,
        title: str | None = None,
        width: int = 1000,
        height: int = 560,
    ) -> Any:
        """
        Plot shear-capacity components over a link-angle sweep.

        The x-axis is link angle and the figure includes a cot(theta) slider.
        Each slider step recomputes ``V_Rd,s(alpha)`` and ``V_Rd,max(alpha)``
        for that cot(theta) value.

        Args:
            load_case: Shear demand definition as either ``LoadCase`` or a
                ``dict`` with keys ``V_Ed`` and optional ``M_Ed``/``N_Ed`` (kN, kN·m).
            cot_theta_min: Optional lower bound for cot(theta). If ``None``,
                the NDP lower limit is used.
            cot_theta_max: Optional upper bound for cot(theta). If ``None``,
                the NDP upper limit is used.
            n_cot: Number of cot(theta) slider samples used only when
                ``cot_theta_step`` is ``None`` or non-positive.
            cot_theta_step: Uniform cot(theta) increment for slider values.
                Defaults to ``0.05``. Set to ``None`` or non-positive to use
                ``n_cot`` linear spacing.
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
            TypeError: If ``load_case`` is not a ``LoadCase`` or compatible dict.
            ImportError: If Plotly is not installed.
        """
        self._require_shear_reinforcement()

        try:
            import plotly.graph_objects as go
        except ImportError as e:
            raise ImportError("Plotly is required for plotting. Install with: pip install plotly") from e

        case = _as_load_case(load_case)
        base_context = self._build_context(load_case=case, use_uncracked_V_Rd_c=use_uncracked_V_Rd_c)

        ndp_cot_min = get_ndp("cot_theta_lower_lim")
        default_cot_min = float(ndp_cot_min() if callable(ndp_cot_min) else ndp_cot_min)
        ndp_cot_max = get_ndp("cot_theta_upper_lim")
        if callable(ndp_cot_max):
            default_cot_max = base_context.cot_max
        else:
            default_cot_max = float(ndp_cot_max)

        cot_min = default_cot_min if cot_theta_min is None else float(cot_theta_min)
        cot_max = default_cot_max if cot_theta_max is None else float(cot_theta_max)
        if cot_min > cot_max:
            cot_min, cot_max = cot_max, cot_min

        cot_vals = _build_slider_values(
            value_min=cot_min,
            value_max=cot_max,
            n_points=n_cot,
            step=cot_theta_step,
        )
        cot_labels = [f"{float(cot_theta):.2f}" for cot_theta in cot_vals]
        series_by_cot = [
            self._compute_link_angle_study_series(
                load_case=case,
                cot_theta=float(cot_theta),
                angle_min=angle_min,
                angle_max=angle_max,
                n_points=n_points,
                use_uncracked_V_Rd_c=use_uncracked_V_Rd_c,
                use_note_2=use_note_2,
            )
            for cot_theta in cot_vals
        ]
        first_series = series_by_cot[0]
        context = first_series.context
        all_cap_vals = [context.V_Ed, context.V_Rd_c]
        for series in series_by_cot:
            all_cap_vals.extend(series.V_Rd_s_vals)
            all_cap_vals.extend(series.V_Rd_max_vals)
        finite_cap_vals = [float(v) for v in all_cap_vals if np.isfinite(v)]
        cap_min = min(finite_cap_vals) if finite_cap_vals else 0.0
        cap_max = max(finite_cap_vals) if finite_cap_vals else 1.0
        y_min = min(0.0, cap_min)
        y_span = max(1e-6, cap_max - y_min)
        y_max = cap_max + 0.05 * y_span

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=first_series.angle_vals,
                y=[context.V_Ed] * len(first_series.angle_vals),
                mode="lines",
                name="V_Ed",
                line=dict(color="black", dash="dash"),
                hovertemplate="alpha: %{x:.1f}°<br>V_Ed: %{y:.1f} kN<extra></extra>",
            ),
        )
        fig.add_trace(
            go.Scatter(
                x=first_series.angle_vals,
                y=[context.V_Rd_c] * len(first_series.angle_vals),
                mode="lines",
                name="V_Rd,c",
                line=dict(color="#8c564b", dash="dot"),
                hovertemplate="alpha: %{x:.1f}°<br>V_Rd,c: %{y:.1f} kN<extra></extra>",
            ),
        )
        fig.add_trace(
            go.Scatter(
                x=first_series.angle_vals,
                y=first_series.V_Rd_s_vals,
                mode="lines",
                name="V_Rd,s(alpha)",
                line=dict(color="#1f77b4"),
                hovertemplate="alpha: %{x:.1f}°<br>V_Rd,s: %{y:.1f} kN<extra></extra>",
            ),
        )
        fig.add_trace(
            go.Scatter(
                x=first_series.angle_vals,
                y=first_series.V_Rd_max_vals,
                mode="lines",
                name="V_Rd,max(alpha)",
                line=dict(color="#ff7f0e"),
                hovertemplate="alpha: %{x:.1f}°<br>V_Rd,max: %{y:.1f} kN<extra></extra>",
            ),
        )

        frames = []
        for i, cot_label in enumerate(cot_labels):
            series = series_by_cot[i]
            frames.append(
                {
                    "name": cot_label,
                    "data": [
                        go.Scatter(
                            x=series.angle_vals,
                            y=[series.context.V_Ed] * len(series.angle_vals),
                            mode="lines",
                            name="V_Ed",
                            line=dict(color="black", dash="dash"),
                            hovertemplate="alpha: %{x:.1f}°<br>V_Ed: %{y:.1f} kN<extra></extra>",
                        ),
                        go.Scatter(
                            x=series.angle_vals,
                            y=[series.context.V_Rd_c] * len(series.angle_vals),
                            mode="lines",
                            name="V_Rd,c",
                            line=dict(color="#8c564b", dash="dot"),
                            hovertemplate="alpha: %{x:.1f}°<br>V_Rd,c: %{y:.1f} kN<extra></extra>",
                        ),
                        go.Scatter(
                            x=series.angle_vals,
                            y=series.V_Rd_s_vals,
                            mode="lines",
                            name="V_Rd,s(alpha)",
                            line=dict(color="#1f77b4"),
                            hovertemplate="alpha: %{x:.1f}°<br>V_Rd,s: %{y:.1f} kN<extra></extra>",
                        ),
                        go.Scatter(
                            x=series.angle_vals,
                            y=series.V_Rd_max_vals,
                            mode="lines",
                            name="V_Rd,max(alpha)",
                            line=dict(color="#ff7f0e"),
                            hovertemplate="alpha: %{x:.1f}°<br>V_Rd,max: %{y:.1f} kN<extra></extra>",
                        ),
                    ],
                },
            )
        fig.frames = frames

        layout_kwargs: dict[str, Any] = dict(
            title=title or "Shear Capacity Study vs Link Angle (cot(theta) slider)",
            template="plotly_white",
            width=width,
            height=height,
            uirevision="link_angle_study",
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1.0,
                xanchor="left",
                x=1.02,
            ),
            margin=dict(r=240),
            xaxis=dict(title="Link angle alpha (degrees)"),
            yaxis=dict(title="Capacity (kN)", range=[y_min, y_max], autorange=False),
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

    def plot_link_angle_moment_shift_study(
        self,
        *,
        load_case: LoadCase | dict[str, Any],
        cot_theta_min: float | None = None,
        cot_theta_max: float | None = None,
        n_cot: int = 20,
        cot_theta_step: float | None = 0.05,
        angle_min: float = 45.0,
        angle_max: float = 90.0,
        n_points: int = 46,
        use_uncracked_V_Rd_c: bool = False,
        use_note_2: bool = False,
        save_path: str | Path | None = None,
        show: bool = True,
        title: str | None = None,
        width: int = 1000,
        height: int = 560,
    ) -> Any:
        """
        Plot utilization and tension-shift add-on versus link angle.

        The x-axis is link angle and the figure includes a cot(theta) slider.
        Each slider step recomputes utilization and ``M_add`` for that
        cot(theta) value.

        Args:
            load_case: Shear demand definition as either ``LoadCase`` or a
                ``dict`` with keys ``V_Ed`` and optional ``M_Ed``/``N_Ed`` (kN, kN·m).
            cot_theta_min: Optional lower bound for cot(theta). If ``None``,
                the NDP lower limit is used.
            cot_theta_max: Optional upper bound for cot(theta). If ``None``,
                the NDP upper limit is used.
            n_cot: Number of cot(theta) slider samples used only when
                ``cot_theta_step`` is ``None`` or non-positive.
            cot_theta_step: Uniform cot(theta) increment for slider values.
                Defaults to ``0.05``. Set to ``None`` or non-positive to use
                ``n_cot`` linear spacing.
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
            TypeError: If ``load_case`` is not a ``LoadCase`` or compatible dict.
            ImportError: If Plotly is not installed.
        """
        self._require_shear_reinforcement()

        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError as e:
            raise ImportError("Plotly is required for plotting. Install with: pip install plotly") from e

        case = _as_load_case(load_case)
        base_context = self._build_context(load_case=case, use_uncracked_V_Rd_c=use_uncracked_V_Rd_c)

        ndp_cot_min = get_ndp("cot_theta_lower_lim")
        default_cot_min = float(ndp_cot_min() if callable(ndp_cot_min) else ndp_cot_min)
        ndp_cot_max = get_ndp("cot_theta_upper_lim")
        if callable(ndp_cot_max):
            default_cot_max = base_context.cot_max
        else:
            default_cot_max = float(ndp_cot_max)

        cot_min = default_cot_min if cot_theta_min is None else float(cot_theta_min)
        cot_max = default_cot_max if cot_theta_max is None else float(cot_theta_max)
        if cot_min > cot_max:
            cot_min, cot_max = cot_max, cot_min

        cot_vals = _build_slider_values(
            value_min=cot_min,
            value_max=cot_max,
            n_points=n_cot,
            step=cot_theta_step,
        )
        cot_labels = [f"{float(cot_theta):.2f}" for cot_theta in cot_vals]
        series_by_cot = [
            self._compute_link_angle_study_series(
                load_case=case,
                cot_theta=float(cot_theta),
                angle_min=angle_min,
                angle_max=angle_max,
                n_points=n_points,
                use_uncracked_V_Rd_c=use_uncracked_V_Rd_c,
                use_note_2=use_note_2,
            )
            for cot_theta in cot_vals
        ]
        first_series = series_by_cot[0]
        all_util_vals = [1.0]
        all_m_add_vals: list[float] = []
        for series in series_by_cot:
            all_util_vals.extend(series.util_vals)
            all_m_add_vals.extend(series.M_add_vals)

        finite_util = [float(v) for v in all_util_vals if np.isfinite(v)]
        util_max = max(finite_util) if finite_util else 1.0
        util_span = max(1e-6, util_max)
        util_range = [0.0, util_max + 0.05 * util_span]

        finite_m_add = [float(v) for v in all_m_add_vals if np.isfinite(v)]
        if finite_m_add:
            m_min = min(finite_m_add)
            m_max = max(finite_m_add)
        else:
            m_min, m_max = 0.0, 1.0
        m_span = max(1e-6, m_max - m_min)
        m_range = [m_min - 0.05 * m_span, m_max + 0.05 * m_span]

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Scatter(
                x=first_series.angle_vals,
                y=first_series.util_vals,
                mode="lines",
                name="Utilization",
                line=dict(color="#d62728"),
                hovertemplate="alpha: %{x:.1f}°<br>Utilization: %{y:.3f}<extra></extra>",
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=first_series.angle_vals,
                y=first_series.M_add_vals,
                mode="lines",
                name="M_add (tension shift)",
                line=dict(color="#9467bd"),
                hovertemplate="alpha: %{x:.1f}°<br>M_add: %{y:.2f} kN·m<extra></extra>",
            ),
            secondary_y=True,
        )
        fig.add_trace(
            go.Scatter(
                x=first_series.angle_vals,
                y=[1.0] * len(first_series.angle_vals),
                mode="lines",
                name="Utilization = 1.0",
                line=dict(color="black", dash="dot"),
                hovertemplate="Utilization limit<extra></extra>",
            ),
            secondary_y=False,
        )

        frames = []
        for i, cot_label in enumerate(cot_labels):
            series = series_by_cot[i]
            frames.append(
                {
                    "name": cot_label,
                    "data": [
                        go.Scatter(
                            x=series.angle_vals,
                            y=series.util_vals,
                            mode="lines",
                            name="Utilization",
                            line=dict(color="#d62728"),
                            hovertemplate="alpha: %{x:.1f}°<br>Utilization: %{y:.3f}<extra></extra>",
                        ),
                        go.Scatter(
                            x=series.angle_vals,
                            y=series.M_add_vals,
                            mode="lines",
                            name="M_add (tension shift)",
                            line=dict(color="#9467bd"),
                            hovertemplate="alpha: %{x:.1f}°<br>M_add: %{y:.2f} kN·m<extra></extra>",
                        ),
                        go.Scatter(
                            x=series.angle_vals,
                            y=[1.0] * len(series.angle_vals),
                            mode="lines",
                            name="Utilization = 1.0",
                            line=dict(color="black", dash="dot"),
                            hovertemplate="Utilization limit<extra></extra>",
                        ),
                    ],
                },
            )
        fig.frames = frames

        layout_kwargs: dict[str, Any] = dict(
            title=title or "Tension-Shift Study vs Link Angle (cot(theta) slider)",
            template="plotly_white",
            width=width,
            height=height,
            uirevision="link_angle_moment_shift_study",
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1.0,
                xanchor="left",
                x=1.02,
            ),
            margin=dict(r=240),
            xaxis=dict(title="Link angle alpha (degrees)"),
            yaxis=dict(title="Utilization", range=util_range, autorange=False),
            yaxis2=dict(title="M_add (kN·m)", range=m_range, autorange=False),
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

    def plot_cot_theta_link_angle_heatmap(
        self,
        *,
        load_case: LoadCase | dict[str, Any],
        cot_theta_min: float | None = None,
        cot_theta_max: float | None = None,
        angle_min: float = 45.0,
        angle_max: float = 90.0,
        n_cot: int = 40,
        n_angles: int = 40,
        metric: str = "utilization",
        use_uncracked_V_Rd_c: bool = False,
        use_note_2: bool = False,
        save_path: str | Path | None = None,
        show: bool = True,
        title: str | None = None,
        width: int = 980,
        height: int = 760,
    ) -> Any:
        """
        Plot a cot(theta)-vs-link-angle heatmap for shear response metrics.

        Args:
            load_case: Shear demand definition as either ``LoadCase`` or a
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
            TypeError: If ``load_case`` is not a ``LoadCase`` or compatible dict.
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
            colorbar_title = "Utilization"
            zmin = 0.0
            # Non-computable cells (V_Rd <= 0) are set to +inf above. np.nanmax
            # does NOT ignore inf, so derive zmax from finite cells only and then
            # render the inf cells at the saturated colour (mirrors the sibling
            # contour methods).
            finite_vals = Z[np.isfinite(Z)]
            zmax = max(1.5, float(np.nanmax(finite_vals))) if finite_vals.size else 1.5
            Z = np.array(Z, copy=True)
            Z[np.isinf(Z)] = zmax
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
            colorbar_title = "kN" if metric_key != "utilization" else "Utilization"
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

    def plot_force_cot_theta_contour(
        self,
        *,
        load_case: LoadCase | dict[str, Any],
        n_axial: int = 31,
        n_moment: int = 31,
        moment_on_y_axis: bool = False,
        cot_theta_min: float | None = None,
        cot_theta_max: float | None = None,
        n_cot: int = 40,
        metric: str = "utilization",
        use_uncracked_V_Rd_c: bool = False,
        use_note_2: bool = False,
        save_path: str | Path | None = None,
        show: bool = True,
        title: str | None = None,
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
                ``LoadCase`` or a ``dict`` with keys ``V_Ed`` and optional
                ``M_Ed``/``N_Ed``.
            n_axial: Number of axial-force samples used on the heatmap force axis.
            n_moment: Number of moment samples used on the heatmap force axis.
            moment_on_y_axis: If ``True``, the y-axis is moment and the slider
                controls axial force. If ``False``, the y-axis is axial force and
                the slider controls moment. Ranges are derived from the current
                M-N interaction diagram, and slider values are auto-generated
                with nice round increments targeting about 50 steps.
            cot_theta_min: Optional lower bound for cot(theta). If ``None``,
                the minimum across sampled valid M-N states is used.
            cot_theta_max: Optional upper bound for cot(theta). If ``None``,
                the maximum across sampled valid M-N states is used.
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
            y_label = "Moment My_Ed (kN·m)"
            y_hover_name = "My_Ed"
            y_hover_format = ".2f"
            y_hover_unit = "kN·m"
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
            slider_prefix = "My_Ed (kN·m): "

        y_edges = _build_axis_edges(y_vals)
        display_oversample = 3
        display_y_edges = _subdivide_axis(y_edges, display_oversample)
        display_y_vals = _axis_centers_from_edges(display_y_edges)

        context_grid: list[list[_StudyContext | None]] = [
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
                    sweep_case = LoadCase(Vz_Ed=case.V_Ed, My_Ed=y_eval, N_Ed=fixed_force)
                else:
                    sweep_case = LoadCase(Vz_Ed=case.V_Ed, My_Ed=fixed_force, N_Ed=y_eval)

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
                    V_Rd_s = self.check.find_V_Rd_s(cot_f, context.z, use_note_2=use_note_2)
                    V_Rd_max = self.check.find_V_Rd_max(
                        cot_f,
                        context.z,
                        context.sigma_cp,
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
            title=title or f"Force vs cot(theta): {metric_key} ({mode_title})",
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
        loadcases: Sequence[dict[str, Any] | Sequence[float]] | None = None,
        n_diagram_points: int = 120,
        n_moment: int = 41,
        n_axial: int = 31,
        cot_theta_min: float | None = None,
        cot_theta_max: float | None = None,
        n_cot: int = 20,
        cot_theta_step: float | None = 0.05,
        use_uncracked_V_Rd_c: bool = False,
        use_note_2: bool = False,
        save_path: str | Path | None = None,
        show: bool = True,
        title: str | None = None,
        width: int = 1000,
        height: int = 760,
    ) -> Any:
        """
        Plot an M-N utilization heatmap clipped to the interaction diagram.

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

        context_grid: list[list[_StudyContext | None]] = [
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
                case = LoadCase(Vz_Ed=V_Ed, My_Ed=m_eval, N_Ed=float(n_ed))
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
                load_case=LoadCase(Vz_Ed=V_Ed, My_Ed=0.0, N_Ed=0.0),
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
                    V_Rd_s = self.check.find_V_Rd_s(cot_f, context.z, use_note_2=use_note_2)
                    V_Rd_max = self.check.find_V_Rd_max(
                        cot_f,
                        context.z,
                        context.sigma_cp,
                        use_note_2=use_note_2,
                    )
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
                    "My_Ed: %{x:.2f} kN·m<br>"
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
                    "M_Rd: %{x:.2f} kN·m<br>"
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
                        f"My_Ed: {plotted_case.M_Ed:.2f} kN·m<br>"
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
                                "My_Ed: %{x:.2f} kN·m<br>"
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
            title=title or "Axial-Moment Utilization (cot(theta) slider)",
            xaxis_title="Moment My_Ed (kN·m)",
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
