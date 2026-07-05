#!/usr/bin/env python3
"""
BENCH-001 + CACHE-301: CircularSectionCheck timing harness.

Measures:
  - Wall time per check call (warm strain cache — repeated identical loads)
  - Wall time per check call (cold strain cache — unique load cases)
  - Speedup from HOT-000 strain cache (warm/cold ratio)
  - Cost of with_updates() + diagram rebuild for config sweeps (ARCH-401 gate)
  - Snapshot (model_dump x2) overhead as % of _get_diagram() time (CACHE-301 gate)

Run from repo root::

    python benchmarks/bench_circular.py
    python benchmarks/bench_circular.py --n-warm 2000 --n-cold 200
    python benchmarks/bench_circular.py --json > results.json

CACHE-301 decision rule:  if snapshot_pct > 5% → implement fingerprinting.
ARCH-401 decision rule:   if diagram_build_ms > 20% of per-check total → share diagrams.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import warnings
from typing import Any

import numpy as np


# ------------------------------------------------------------------ fixtures

def _make_check(
    diameter: float = 600.0,
    grade: str = "C30/37",
    cover: float = 40.0,
    link_spacing: float = 200.0,
):
    """Build a representative CircularSectionCheck (C30/37, D=600, 12×T20, T12@200)."""
    from section_design_checks.reinforced_concrete.code_checks.ec2_2004.circular_section_check import (
        CircularSectionCheck,
    )
    from section_design_checks.reinforced_concrete.geometry import (
        create_circular_section,
        create_circular_perimeter_rebars,
    )
    from section_design_checks.reinforced_concrete.materials import ConcreteMaterial, Rebar, ShearRebar

    section = create_circular_section(diameter=diameter, hook_ref=0)
    perimeter = create_circular_perimeter_rebars(
        rebar=Rebar(diameter=20, grade="B500B"),
        diameter=diameter,
        cover=cover,
        n_bars=12,
        origin=(0.0, 0.0),
    )
    section.add_rebar_group(perimeter)
    return CircularSectionCheck(
        section=section,
        concrete=ConcreteMaterial(grade=grade),
        diameter=diameter,
        cover=cover,
        shear_reinforcement=ShearRebar(
            diameter=12,
            link_spacing=link_spacing,
            n_legs=2,
            grade="B500B",
        ),
    )


def _shear_loads(n: int, seed: int = 42) -> list[dict]:
    rng = np.random.default_rng(seed)
    return [
        {"V_Ed": float(v), "M_Ed": float(m), "N_Ed": float(n_)}
        for v, m, n_ in zip(
            rng.uniform(50.0, 300.0, n),
            rng.uniform(50.0, 250.0, n),
            rng.uniform(300.0, 2000.0, n),
        )
    ]


def _bending_loads(n: int, seed: int = 42) -> list[dict]:
    rng = np.random.default_rng(seed)
    return [
        {"M_Ed": float(m), "N_Ed": float(n_)}
        for m, n_ in zip(
            rng.uniform(50.0, 250.0, n),
            rng.uniform(300.0, 2000.0, n),
        )
    ]


# ------------------------------------------------------------------ timing helpers

def _run_shear(check, load: dict) -> None:
    from section_design_checks.reinforced_concrete.code_checks.ec2_2004.flexure_utils import LoadCase

    check.perform_shear_check(
        load_case=LoadCase(**load),
        suppress_warnings=True,
    )


def _run_bending(check, load: dict) -> None:
    check.perform_bending_check(suppress_warnings=True, **load)


def _run_cracking(check, load: dict) -> None:
    check.perform_cracking_check(**load)


def _time_ms(fn, args_list: list) -> list[float]:
    """Run fn(*args) for each entry, return times in ms (warnings suppressed)."""
    times: list[float] = []
    for args in args_list:
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fn(*args)
        times.append((time.perf_counter() - t0) * 1000.0)
    return times


def _stats(times: list[float]) -> dict[str, Any]:
    s = sorted(times)
    n = len(s)
    p95_idx = min(n - 1, int(n * 0.95))
    return {
        "n": n,
        "mean_ms": round(statistics.mean(s), 3),
        "median_ms": round(statistics.median(s), 3),
        "p95_ms": round(s[p95_idx], 3),
        "min_ms": round(s[0], 3),
        "max_ms": round(s[-1], 3),
        "total_s": round(sum(s) / 1000.0, 3),
    }


# ------------------------------------------------------------------ scenarios

def bench_warm_cache(
    check,
    n_warm: int,
    shear_load: dict,
    bending_load: dict,
    cracking_load: dict,
) -> dict[str, Any]:
    """
    Scenario A: warm strain cache (same load repeated).

    The strain solver ran once on the first call; every subsequent call hits
    _strain_cache and skips scipy.  This measures the per-call floor when
    geometry and loads are stable.
    """
    from section_design_checks.reinforced_concrete.code_checks.ec2_2004.flexure_utils import LoadCase

    shear_lc = LoadCase(**shear_load)

    shear_times = _time_ms(
        lambda lc: check.perform_shear_check(load_case=lc, suppress_warnings=True),
        [(shear_lc,)] * n_warm,
    )
    bending_times = _time_ms(
        lambda m, n: check.perform_bending_check(M_Ed=m, N_Ed=n, suppress_warnings=True),
        [(bending_load["M_Ed"], bending_load["N_Ed"])] * n_warm,
    )
    cracking_times = _time_ms(
        lambda m, n: check.perform_cracking_check(M_Ed=m, N_Ed=n),
        [(cracking_load["M_Ed"], cracking_load["N_Ed"])] * n_warm,
    )

    return {
        "scenario": "warm_strain_cache",
        "description": "Same load case repeated — strain cache always hits",
        "shear": _stats(shear_times),
        "bending": _stats(bending_times),
        "cracking": _stats(cracking_times),
    }


def bench_cold_cache(
    check,
    n_cold: int,
    shear_loads: list[dict],
    bending_loads: list[dict],
    cracking_loads: list[dict],
) -> dict[str, Any]:
    """
    Scenario B: cold strain cache (unique load cases).

    Every (M, N) pair is distinct so _strain_cache misses and scipy runs.
    This measures the realistic cost when iterating over many load combinations.
    """
    from section_design_checks.reinforced_concrete.code_checks.ec2_2004.flexure_utils import LoadCase

    shear_lcs = [LoadCase(**lc) for lc in shear_loads[:n_cold]]
    shear_times = _time_ms(
        lambda lc: check.perform_shear_check(load_case=lc, suppress_warnings=True),
        [(lc,) for lc in shear_lcs],
    )
    bending_times = _time_ms(
        lambda m, n: check.perform_bending_check(M_Ed=m, N_Ed=n, suppress_warnings=True),
        [(lc["M_Ed"], lc["N_Ed"]) for lc in bending_loads[:n_cold]],
    )
    cracking_times = _time_ms(
        lambda m, n: check.perform_cracking_check(M_Ed=m, N_Ed=n),
        [(lc["M_Ed"], lc["N_Ed"]) for lc in cracking_loads[:n_cold]],
    )

    return {
        "scenario": "cold_strain_cache",
        "description": "Unique load cases — strain solver runs every call",
        "shear": _stats(shear_times),
        "bending": _stats(bending_times),
        "cracking": _stats(cracking_times),
    }


def bench_config_sweep(n_configs: int, n_checks_per_config: int = 5) -> dict[str, Any]:
    """
    Scenario C: with_updates() config sweep.

    Simulates a design parameter study (varying link_spacing across n_configs
    distinct designs), running n_checks_per_config checks per design to force
    diagram build.  Reports:
      - construction time (with_updates + sub-check model construction)
      - first-check time (triggers diagram build for 4 sub-checks)
      - subsequent-check time (diagram cached, strain cache warm)

    This gates ARCH-401 (shared diagram registry).
    """
    from section_design_checks.reinforced_concrete.code_checks.ec2_2004.flexure_utils import LoadCase

    base_check = _make_check()
    reference_load = LoadCase(V_Ed=200.0, M_Ed=150.0, N_Ed=1000.0)

    # Vary link_spacing so with_updates() produces genuinely new sub-checks
    # (i.e. new shear_reinforcement objects — triggers sub-check rebuild)
    spacings = np.linspace(100.0, 400.0, n_configs)

    from section_design_checks.reinforced_concrete.materials import ShearRebar

    construct_times: list[float] = []
    first_check_times: list[float] = []
    warm_check_times: list[float] = []

    for s in spacings:
        new_rebar = ShearRebar(diameter=12, link_spacing=float(s), n_legs=2, grade="B500B")

        # Construction (delegates rebuilt, diagrams NOT yet built — lazy)
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            new_check = base_check.with_updates(shear_reinforcement=new_rebar)
        construct_times.append((time.perf_counter() - t0) * 1000.0)

        # First call — triggers diagram build on all sub-checks touched by shear path
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            new_check.perform_shear_check(
                load_case=reference_load, suppress_warnings=True
            )
        first_check_times.append((time.perf_counter() - t0) * 1000.0)

        # Subsequent calls — diagrams cached, strains cached
        t0 = time.perf_counter()
        for _ in range(n_checks_per_config - 1):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                new_check.perform_shear_check(
                    load_case=reference_load, suppress_warnings=True
                )
        warm_check_times.append((time.perf_counter() - t0) * 1000.0 / (n_checks_per_config - 1))

    return {
        "scenario": "config_sweep",
        "description": f"{n_configs} configs via with_updates(), {n_checks_per_config} checks each",
        "n_configs": n_configs,
        "n_checks_per_config": n_checks_per_config,
        "construction": _stats(construct_times),
        "first_check_shear": _stats(first_check_times),
        "warm_check_shear": _stats(warm_check_times),
        "diagram_build_estimate_ms": round(
            statistics.mean(first_check_times) - statistics.mean(warm_check_times), 3
        ),
    }


# ------------------------------------------------------------------ CACHE-301

def bench_snapshot_overhead(check, n: int = 5000) -> dict[str, Any]:
    """
    CACHE-301: Measure snapshot (model_dump) overhead vs total _get_diagram() time.

    On warm cache, _get_diagram() does:
      1. _take_snapshot()  → section.model_dump() + concrete.model_dump()
      2. snapshot == cached_snapshot   (dict equality check)
      3. return self._diagram          (no rebuild)

    Steps 1–2 are pure overhead for a stable geometry.  If they exceed 5% of
    per-check time, fingerprinting (CACHE-301) is justified.

    Tests ShearCheck because perform_shear_check calls _get_diagram() twice:
    once at the top of perform_shear_check (HOT-101) and once inside find_lever_arm.
    """
    sc = check._shear_check

    # Force diagram build so subsequent calls are pure snapshot + return
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc._get_diagram(False)

    # Time N calls to _get_diagram() (warm, no rebuild)
    t0 = time.perf_counter()
    for _ in range(n):
        sc._get_diagram(False)
    get_diag_total_ms = (time.perf_counter() - t0) * 1000.0

    # Time N pairs of model_dump() — exactly what _take_snapshot() costs
    # (_take_snapshot also has dict-key logic but the two model_dump calls dominate)
    t0 = time.perf_counter()
    for _ in range(n):
        sc.section.model_dump()
        sc.concrete.model_dump()
    model_dump_total_ms = (time.perf_counter() - t0) * 1000.0

    # Time a single model_dump for reference
    t0 = time.perf_counter()
    for _ in range(n):
        sc.section.model_dump()
    section_dump_ms = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    for _ in range(n):
        sc.concrete.model_dump()
    concrete_dump_ms = (time.perf_counter() - t0) * 1000.0

    get_diag_mean = get_diag_total_ms / n
    model_dump_mean = model_dump_total_ms / n
    snapshot_pct = (model_dump_mean / get_diag_mean * 100.0) if get_diag_mean > 0 else 0.0

    return {
        "scenario": "cache_301_snapshot_overhead",
        "description": "model_dump overhead in _get_diagram() for stable geometry (warm cache)",
        "n": n,
        "get_diagram_mean_us": round(get_diag_mean * 1000, 2),  # convert ms → µs
        "model_dump_2x_mean_us": round(model_dump_mean * 1000, 2),
        "section_dump_mean_us": round(section_dump_ms / n * 1000, 2),
        "concrete_dump_mean_us": round(concrete_dump_ms / n * 1000, 2),
        "snapshot_pct_of_get_diagram": round(snapshot_pct, 1),
        "implement_fingerprinting": snapshot_pct > 5.0,
        "note": (
            "snapshot_pct > 5% → implement CACHE-301 fingerprinting. "
            "Each perform_shear_check calls _get_diagram() ~2x (HOT-101 + find_lever_arm), "
            "so effective per-check snapshot cost ≈ 2 × model_dump_2x."
        ),
    }


# ------------------------------------------------------------------ reporting

def _print_section(title: str, width: int = 64) -> None:
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def _print_timing(label: str, stats: dict, indent: int = 2) -> None:
    pad = " " * indent
    print(
        f"{pad}{label:<28}  "
        f"median={stats['median_ms']:>8.3f} ms  "
        f"p95={stats['p95_ms']:>8.3f} ms  "
        f"mean={stats['mean_ms']:>8.3f} ms  "
        f"(n={stats['n']})"
    )


def _print_results(results: list[dict], as_json: bool) -> None:
    if as_json:
        print(json.dumps(results, indent=2))
        return

    warm = next(r for r in results if r["scenario"] == "warm_strain_cache")
    cold = next(r for r in results if r["scenario"] == "cold_strain_cache")
    sweep = next(r for r in results if r["scenario"] == "config_sweep")
    cache = next(r for r in results if r["scenario"] == "cache_301_snapshot_overhead")

    _print_section("Scenario A - Warm strain cache (same load repeated)")
    print(f"  {warm['description']}")
    _print_timing("Shear check", warm["shear"])
    _print_timing("Bending check", warm["bending"])
    _print_timing("Cracking check", warm["cracking"])

    _print_section("Scenario B - Cold strain cache (unique load cases)")
    print(f"  {cold['description']}")
    _print_timing("Shear check", cold["shear"])
    _print_timing("Bending check", cold["bending"])
    _print_timing("Cracking check", cold["cracking"])

    _print_section("HOT-000 strain cache speedup (cold/warm ratio)")
    for check_name in ("shear", "bending", "cracking"):
        ratio = cold[check_name]["median_ms"] / max(warm[check_name]["median_ms"], 1e-9)
        print(f"  {check_name.capitalize():<8}  {ratio:>6.1f}x  faster on warm cache")

    _print_section("Scenario C - Config sweep (with_updates())")
    print(f"  {sweep['description']}")
    _print_timing("Construction (with_updates)", sweep["construction"])
    _print_timing("First check (diagram build)", sweep["first_check_shear"])
    _print_timing("Warm check (diagram cached)", sweep["warm_check_shear"])
    print(
        f"\n  Estimated diagram build cost: {sweep['diagram_build_estimate_ms']:.1f} ms\n"
        f"  (first_check median - warm_check median)\n"
        f"\n  ARCH-401 gate: {'INVESTIGATE' if sweep['diagram_build_estimate_ms'] > 0.2 * cold['shear']['median_ms'] else 'DEFER'}"
        f"  (>20% of cold shear median = {0.2 * cold['shear']['median_ms']:.1f} ms)"
    )

    _print_section("CACHE-301 - Snapshot overhead investigation")
    print(f"  {cache['description']}")
    print(f"  _get_diagram() mean:          {cache['get_diagram_mean_us']:>8.1f} us")
    print(f"  model_dump x2 mean:           {cache['model_dump_2x_mean_us']:>8.1f} us")
    print(f"    section.model_dump():       {cache['section_dump_mean_us']:>8.1f} us")
    print(f"    concrete.model_dump():      {cache['concrete_dump_mean_us']:>8.1f} us")
    print(f"  Snapshot % of _get_diagram(): {cache['snapshot_pct_of_get_diagram']:>7.1f}%")
    print()
    decision = "IMPLEMENT fingerprinting" if cache["implement_fingerprinting"] else "DEFER (below 5% threshold)"
    print(f"  CACHE-301 decision: {decision}")
    print()
    warm_shear_us = warm["shear"]["median_ms"] * 1000.0
    effective_snap_us = cache["model_dump_2x_mean_us"] * 2  # 2x _get_diagram per shear call
    print(
        f"  Context: each perform_shear_check calls _get_diagram() ~2x,\n"
        f"  so effective snapshot cost ~= {effective_snap_us:.1f} us / call\n"
        f"  vs warm shear median ~= {warm_shear_us:.1f} us/call  "
        f"({effective_snap_us / max(warm_shear_us, 1e-9) * 100:.1f}% of per-call budget)."
    )


# ------------------------------------------------------------------ main

def main() -> None:
    parser = argparse.ArgumentParser(
        description="BENCH-001 + CACHE-301: CircularSectionCheck timing harness"
    )
    parser.add_argument(
        "--n-warm", type=int, default=1000,
        help="Warm-cache call count per check type (default: 1000)",
    )
    parser.add_argument(
        "--n-cold", type=int, default=100,
        help="Cold-cache (unique load) call count per check type (default: 100)",
    )
    parser.add_argument(
        "--n-configs", type=int, default=100,
        help="Config count for with_updates() sweep (default: 100)",
    )
    parser.add_argument(
        "--n-snapshot", type=int, default=5000,
        help="Iterations for CACHE-301 snapshot timing (default: 5000)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for load generation (default: 42)",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output results as JSON",
    )
    args = parser.parse_args()

    if not args.json:
        print("Building CircularSectionCheck ... ", end="", flush=True)

    t_build = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        check = _make_check()
    build_ms = (time.perf_counter() - t_build) * 1000.0

    if not args.json:
        print(f"done ({build_ms:.0f} ms)")
        print("Generating load cases ... ", end="", flush=True)

    n_cases = max(args.n_warm, args.n_cold) + 10
    shear_loads = _shear_loads(n_cases, seed=args.seed)
    bending_loads = _bending_loads(n_cases, seed=args.seed)
    cracking_loads = _bending_loads(n_cases, seed=args.seed + 1)

    if not args.json:
        print("done")
        print("Warming up (force diagram build) ... ", end="", flush=True)

    # Force diagram build on all 4 sub-checks before timing begins
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from section_design_checks.reinforced_concrete.code_checks.ec2_2004.flexure_utils import LoadCase
        _warmup_lc = LoadCase(**shear_loads[0])
        check.perform_shear_check(load_case=_warmup_lc, suppress_warnings=True)
        check.perform_bending_check(
            M_Ed=bending_loads[0]["M_Ed"],
            N_Ed=bending_loads[0]["N_Ed"],
            suppress_warnings=True,
        )
        check.perform_cracking_check(
            M_Ed=cracking_loads[0]["M_Ed"],
            N_Ed=cracking_loads[0]["N_Ed"],
        )

    if not args.json:
        print("done")

    results: list[dict] = []

    # ---- Scenario A: warm cache
    if not args.json:
        print(f"Running Scenario A (warm, n={args.n_warm}) ... ", end="", flush=True)
    results.append(
        bench_warm_cache(
            check,
            n_warm=args.n_warm,
            shear_load=shear_loads[1],
            bending_load=bending_loads[1],
            cracking_load=cracking_loads[1],
        )
    )
    if not args.json:
        print("done")

    # ---- Scenario B: cold cache
    if not args.json:
        print(f"Running Scenario B (cold, n={args.n_cold}) ... ", end="", flush=True)
    results.append(
        bench_cold_cache(
            check,
            n_cold=args.n_cold,
            # Use loads starting at index 2 so they are genuinely fresh (not in strain cache)
            shear_loads=shear_loads[2:],
            bending_loads=bending_loads[2:],
            cracking_loads=cracking_loads[2:],
        )
    )
    if not args.json:
        print("done")

    # ---- Scenario C: config sweep
    if not args.json:
        print(f"Running Scenario C (config sweep, n={args.n_configs}) ... ", end="", flush=True)
    results.append(bench_config_sweep(n_configs=args.n_configs))
    if not args.json:
        print("done")

    # ---- CACHE-301
    if not args.json:
        print(f"Running CACHE-301 (n={args.n_snapshot}) ... ", end="", flush=True)
    results.append(bench_snapshot_overhead(check, n=args.n_snapshot))
    if not args.json:
        print("done")

    # ---- Report
    _print_results(results, as_json=args.json)

    if not args.json:
        print()
        print(f"Build time (section + check construction): {build_ms:.0f} ms")


if __name__ == "__main__":
    main()
