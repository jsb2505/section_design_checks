"""
Benchmark runner — reads JSON benchmark files, recreates analyses, and
compares results against external reference data.

Usage:
    python -m benchmarks.runner benchmarks/sbd_1.json
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.schemas import (
    AnalysisResult,
    AnalysisSpec,
    AnalysisType,
    BenchmarkFile,
    BenchmarkResult,
    LimitState,
    PointError,
)
from materials.core.geometry import Point2D
from materials.reinforced_concrete.analysis import create_interaction_diagram
from materials.reinforced_concrete.constitutive import ConcreteModelType, SteelModelType
from materials.reinforced_concrete.geometry import RCSection, RebarGroup
from materials.reinforced_concrete.materials import ConcreteMaterial, Rebar


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def _build_concrete(spec: BenchmarkFile) -> ConcreteMaterial:
    """Create ConcreteMaterial from benchmark spec."""
    return ConcreteMaterial(
        grade=spec.concrete.grade,
        alpha_cc=spec.concrete.alpha_cc,
        alpha_ct=spec.concrete.alpha_ct,
        gamma_c=spec.concrete.gamma_c,
        gamma_c_accidental=spec.concrete.gamma_c_accidental,
    )


def _build_section(spec: BenchmarkFile) -> RCSection:
    """Create RCSection with rebar groups from benchmark spec."""
    outline = tuple(Point2D(x=c[0], y=c[1]) for c in spec.outline_coords)

    kwargs: dict[str, Any] = {
        "outline_coords": outline,
        "section_name": spec.name or "benchmark_section",
    }
    if spec.holes:
        kwargs["voids_coords"] = tuple(
            tuple(Point2D(x=c[0], y=c[1]) for c in hole)
            for hole in spec.holes
        )

    section = RCSection(**kwargs)

    # Group bars by diameter for efficient RebarGroup creation
    bars_by_dia: dict[float, list[Point2D]] = defaultdict(list)
    for bar in spec.reinforcement.layout:
        x, y, dia = bar[0], bar[1], bar[2]
        bars_by_dia[dia].append(Point2D(x=x, y=y))

    for dia, positions in bars_by_dia.items():
        rebar = Rebar(
            diameter=dia,
            grade=spec.reinforcement.grade,
            gamma_s=spec.reinforcement.gamma_s,
            gamma_s_accidental=spec.reinforcement.gamma_s_accidental,
            E_s=spec.reinforcement.E_s,
        )
        group = RebarGroup(rebar=rebar, positions=positions)
        section.add_rebar_group(group)

    return section


# ---------------------------------------------------------------------------
# Hausdorff distance
# ---------------------------------------------------------------------------

def _hausdorff_distance(
    curve_a: np.ndarray,
    curve_b: np.ndarray,
) -> tuple[float, list[PointError]]:
    """
    Compute the directed Hausdorff distance from curve_a to curve_b.

    Points are in (M, N) space. The distance is normalised by the range
    of each axis so that M and N contribute equally.

    Args:
        curve_a: shape (n, 2) — external reference points
        curve_b: shape (m, 2) — internal computed points

    Returns:
        (hausdorff_distance, per_point_errors) where per_point_errors
        maps each point in curve_a to its nearest point in curve_b.
    """
    all_pts = np.vstack([curve_a, curve_b])
    ranges = np.ptp(all_pts, axis=0)
    # Avoid division by zero for constant axes
    ranges = np.where(ranges > 0, ranges, 1.0)

    a_norm = curve_a / ranges
    b_norm = curve_b / ranges

    per_point: list[PointError] = []
    max_dist = 0.0

    for i in range(len(a_norm)):
        diffs = b_norm - a_norm[i]
        dists = np.sqrt(np.sum(diffs**2, axis=1))
        j = int(np.argmin(dists))
        d = float(dists[j]) * np.linalg.norm(ranges)  # scale back to original units

        per_point.append(PointError(
            ext_M=float(curve_a[i, 0]),
            ext_N=float(curve_a[i, 1]),
            int_M=float(curve_b[j, 0]),
            int_N=float(curve_b[j, 1]),
            distance=round(d, 4),
        ))
        max_dist = max(max_dist, d)

    return round(max_dist, 4), per_point


# ---------------------------------------------------------------------------
# Analysis runners
# ---------------------------------------------------------------------------

def _resolve_model_types(
    spec: BenchmarkFile,
) -> tuple[ConcreteModelType, SteelModelType]:
    """Map string model_type values to their StrEnum equivalents."""
    concrete_map = {
        "parabola_rectangle": ConcreteModelType.PARABOLA_RECTANGLE,
        "parabola-rectangle": ConcreteModelType.PARABOLA_RECTANGLE,
        "bilinear": ConcreteModelType.BILINEAR,
        "schematic": ConcreteModelType.SCHEMATIC,
    }
    steel_map = {
        "inclined": SteelModelType.INCLINED,
        "horizontal": SteelModelType.HORIZONTAL,
    }
    c_model = concrete_map.get(spec.concrete.model_type, ConcreteModelType.PARABOLA_RECTANGLE)
    s_model = steel_map.get(spec.reinforcement.model_type, SteelModelType.INCLINED)
    return c_model, s_model


def _run_interaction_diagram(
    section: RCSection,
    concrete: ConcreteMaterial,
    spec: BenchmarkFile,
    analysis: AnalysisSpec,
) -> AnalysisResult:
    """Run interaction diagram analysis and compare to external results."""
    c_model, s_model = _resolve_model_types(spec)

    use_accidental = analysis.limit_state == LimitState.ULS_ACC
    use_characteristic = analysis.limit_state in (LimitState.SLS_CHAR, LimitState.SLS_QP)

    diagram = create_interaction_diagram(
        section=section,
        concrete=concrete,
        concrete_model_type=c_model,
        steel_model_type=s_model,
        use_accidental=use_accidental,
        use_characteristic=use_characteristic,
    )

    n_points = analysis.n_points or 200
    points = diagram.generate_diagram_points(n_points=n_points)

    external_pts = np.array(analysis.results, dtype=float)

    # Sample the internal diagram at each external N level using
    # get_capacity_fixed_n.  This gives a direct M_Rd comparison at
    # matching axial force levels, avoiding reference-axis issues from
    # asymmetric reinforcement.  External points at M≈0 are curve
    # endpoints (not capacity values) and are excluded from comparison.
    M_EPS = 0.5  # kN·m — threshold to identify boundary points
    matched_internal: list[list[float]] = []
    matched_external: list[list[float]] = []
    for ext_M, ext_N in external_pts.tolist():
        if abs(ext_M) < M_EPS:
            continue  # skip boundary endpoints
        N_cap, M_pos, M_neg = diagram.get_capacity_fixed_n(
            ext_N, n_points=n_points,
        )
        if N_cap is None:
            continue
        if ext_M >= 0:
            m_int = float(M_pos) if M_pos is not None else 0.0
        else:
            m_int = float(M_neg) if M_neg is not None else 0.0
        matched_internal.append([m_int, float(N_cap)])
        matched_external.append([ext_M, ext_N])

    external_pts = np.array(matched_external, dtype=float)

    matched_arr = np.array(matched_internal, dtype=float)

    # Also keep the full diagram for plotting
    internal_pts = np.array([[p.M, p.N] for p in points])

    hausdorff, per_point = _hausdorff_distance(external_pts, matched_arr)

    status = "PASS" if hausdorff <= analysis.tolerance else "FAIL"

    return AnalysisResult(
        type=analysis.type,
        limit_state=analysis.limit_state,
        status=status,
        hausdorff_distance=hausdorff,
        tolerance=analysis.tolerance,
        external_points=external_pts.tolist(),
        internal_points=matched_arr.tolist(),
        per_point_errors=per_point,
    )


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_RUNNERS: dict[AnalysisType, Any] = {
    AnalysisType.INTERACTION_DIAGRAM: _run_interaction_diagram,
}


def run_benchmark(path: str | Path) -> BenchmarkResult:
    """
    Run all analyses in a benchmark file and produce a comparison result.

    Args:
        path: Path to benchmark JSON file.

    Returns:
        BenchmarkResult with comparison data for each analysis.
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    spec = BenchmarkFile.model_validate(raw)

    concrete = _build_concrete(spec)
    section = _build_section(spec)

    result = BenchmarkResult(
        benchmark_name=spec.name or path.stem,
        source=spec.source,
    )

    for analysis in spec.analyses:
        runner = _RUNNERS.get(analysis.type)
        if runner is None:
            print(f"Skipping unsupported analysis type: {analysis.type}")
            continue

        analysis_result = runner(section, concrete, spec, analysis)
        result.analyses.append(analysis_result)

    # Save outputs
    results_dir = path.parent / "results"
    results_dir.mkdir(exist_ok=True)

    stem = spec.name or path.stem

    # Save JSON result
    result_path = results_dir / f"{stem}_results.json"
    with open(result_path, "w", encoding="utf-8") as f:
        f.write(result.model_dump_json(indent=2))

    # Save HTML plots
    try:
        from benchmarks.plotting import plot_comparison

        for ar in result.analyses:
            html_path = results_dir / f"{stem}_{ar.type}_{ar.limit_state}.html"
            plot_comparison(ar, save_path=html_path, title=f"{stem} — {ar.type} ({ar.limit_state})")
    except ImportError:
        pass  # plotly not available

    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m benchmarks.runner <benchmark.json> [benchmark2.json ...]")
        sys.exit(1)

    for arg in sys.argv[1:]:
        path = Path(arg)
        if not path.exists():
            print(f"File not found: {path}")
            continue

        print(f"\nRunning benchmark: {path.name}")
        print("=" * 60)

        result = run_benchmark(path)

        for ar in result.analyses:
            symbol = "PASS" if ar.status == "PASS" else "FAIL"
            print(
                f"  [{symbol}] {ar.type} ({ar.limit_state}): "
                f"Hausdorff = {ar.hausdorff_distance:.2f} "
                f"(tolerance = {ar.tolerance:.1f})"
            )

        results_dir = path.parent / "results"
        print(f"  Results saved to: {results_dir}")


if __name__ == "__main__":
    main()
