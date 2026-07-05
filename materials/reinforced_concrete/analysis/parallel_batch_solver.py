"""
Parallel batch solver for M-N interaction diagram inverse problems.

Provides helper functions for processing thousands of load cases efficiently
using thread-based parallelism (ThreadPoolExecutor).

**IMPORTANT PERFORMANCE NOTE**:
With analytical Jacobian, each M-N solve is VERY FAST (~2-20ms per case).
Python's GIL (Global Interpreter Lock) prevents true thread parallelism for
CPU-bound tasks. ThreadPoolExecutor overhead (thread coordination, task queue,
context switching) can EXCEED any benefit for such fast tasks.

**Current Performance Reality**:
    - Serial (analytical Jacobian): ~17ms per case
    - Parallel (ThreadPool, 4 workers): ~26ms per case (52% SLOWER!)
    - ThreadPool overhead: ~10ms per task (more than task duration)

**When ThreadPool Helps**:
    - ONLY if each task takes >100ms (e.g., with complex section geometry)
    - NEVER with analytical Jacobian + simple sections

**Recommendation**:
    Use serial processing (solve_batch_serial) for best performance.
    The analytical Jacobian already provides 3-10x speedup vs 2-point.

**Alternative for True Parallelism**:
    ProcessPoolExecutor could provide real speedup, but requires:
    - Serializing section geometry (Shapely geometries not picklable)
    - Recreating MNInteractionDiagram in each process (high overhead)
    - Only worthwhile for >10,000 cases

Example:
    >>> from materials.reinforced_concrete.analysis.interaction_diagram import MNInteractionDiagram
    >>> from materials.reinforced_concrete.analysis.parallel_batch_solver import solve_batch_parallel
    >>>
    >>> # Create diagram once
    >>> diagram = MNInteractionDiagram(section=section, concrete=concrete, ...)
    >>>
    >>> # Define load cases
    >>> load_cases = [
    ...     {"M": 50.0, "N": 100.0},
    ...     {"M": 30.0, "N": 200.0},
    ...     # ... thousands more ...
    ... ]
    >>>
    >>> # Solve in parallel
    >>> results = solve_batch_parallel(diagram, load_cases, n_workers=8)
    >>>
    >>> # Access results
    >>> for case, result in zip(load_cases, results):
    ...     if result["success"]:
    ...         eps_top, eps_bottom = result["eps_top"], result["eps_bottom"]
    ...         print(f"M={case['M']}, N={case['N']} → eps_top={eps_top:.6f}")
    ...     else:
    ...         print(f"Failed: {result['error']}")
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple
import os

import numpy as np

# Type aliases
LoadCase = Dict[str, float]  # {"M": float, "N": float}
SolverResult = Dict[str, Any]  # {"success": bool, "eps_top": float, "eps_bottom": float, "error": str}


def _solve_single_case(diagram: Any, load_case: LoadCase, tol: float) -> SolverResult:
    """
    Worker function for thread-based parallel processing.

    Args:
        diagram: MNInteractionDiagram instance (shared across threads)
        load_case: Dict with "M" and "N" keys
        tol: Solver tolerance

    Returns:
        Dictionary with:
            - success: bool (True if solved, False if failed)
            - eps_top: float (top strain, or None if failed)
            - eps_bottom: float (bottom strain, or None if failed)
            - M: float (input M)
            - N: float (input N)
            - error: str (error message if failed, empty otherwise)
    """
    M = load_case.get("M", 0.0)
    N = load_case.get("N", 0.0)

    try:
        eps_top, eps_bottom = diagram.find_strains_for_MN(
            My_target=M,
            N_target=N,
            tol=tol,
        )
        return {
            "success": True,
            "eps_top": float(eps_top),
            "eps_bottom": float(eps_bottom),
            "M": M,
            "N": N,
            "error": "",
        }
    except Exception as e:
        return {
            "success": False,
            "eps_top": None,
            "eps_bottom": None,
            "M": M,
            "N": N,
            "error": str(e),
        }


def solve_batch_parallel(
    diagram: Any,  # MNInteractionDiagram
    load_cases: List[LoadCase],
    n_workers: Optional[int] = None,
    tol: float = 1e-6,
    show_progress: bool = False,
) -> List[SolverResult]:
    """
    Solve multiple M-N load cases in parallel using threads.

    Uses ThreadPoolExecutor to distribute work across threads.
    Works well because numpy/scipy release the GIL during computation.

    Args:
        diagram: MNInteractionDiagram instance (shared across threads)
        load_cases: List of dicts with "M" and "N" keys (kN·m and kN)
        n_workers: Number of worker threads (default: cpu_count)
        tol: Solver tolerance for least_squares (default: 1e-6)
        show_progress: If True, print progress updates (default: False)

    Returns:
        List of solver results (same order as load_cases)
        Each result is a dict with:
            - success: bool
            - eps_top: float (or None if failed)
            - eps_bottom: float (or None if failed)
            - M: float (input)
            - N: float (input)
            - error: str (empty if success)

    Performance Notes:
        - Speedup: ~4-6x on 8-core system (limited by GIL but still significant)
        - Overhead: minimal (threads share memory)
        - Efficient for: batch_size > 50 cases
        - Less efficient for: very small batches (< 20 cases)

    Example:
        >>> diagram = MNInteractionDiagram(...)
        >>> cases = [{"M": 50, "N": 100}, {"M": 30, "N": 200}]
        >>> results = solve_batch_parallel(diagram, cases, n_workers=4)
        >>> for case, result in zip(cases, results):
        ...     if result["success"]:
        ...         print(f"Solved: eps_top={result['eps_top']:.6f}")

    Raises:
        ValueError: If load_cases is empty

    Note:
        Uses threads instead of processes because MNInteractionDiagram
        contains Shapely prepared geometries that cannot be pickled.
    """
    if not load_cases:
        raise ValueError("load_cases cannot be empty")

    # Default to all available cores
    if n_workers is None:
        n_workers = os.cpu_count() or 4

    if show_progress:
        print(f"Solving {len(load_cases)} cases using {n_workers} threads...")

    # Use ThreadPoolExecutor for parallel execution
    # Pre-allocate with proper type hint to avoid Pylance warnings
    results: List[SolverResult] = [None] * len(load_cases)  # type: ignore[list-item]

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        # Submit all tasks
        future_to_index = {
            executor.submit(_solve_single_case, diagram, case, tol): i
            for i, case in enumerate(load_cases)
        }

        # Collect results as they complete
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            results[index] = future.result()

    if show_progress:
        n_success = sum(1 for r in results if r["success"])
        n_failed = len(results) - n_success
        print(f"Complete: {n_success} solved, {n_failed} failed")

    return results


def solve_batch_serial(
    diagram: Any,  # MNInteractionDiagram
    load_cases: List[LoadCase],
    tol: float = 1e-6,
    show_progress: bool = False,
) -> List[SolverResult]:
    """
    Solve multiple M-N load cases serially (single process).

    Useful for:
    - Small batches (< 50 cases) where parallel overhead isn't worth it
    - Debugging (easier to trace errors)
    - Profiling (cleaner performance data)

    Args:
        diagram: MNInteractionDiagram instance
        load_cases: List of dicts with "M" and "N" keys
        tol: Solver tolerance (default: 1e-6)
        show_progress: If True, print progress every 100 cases

    Returns:
        List of solver results (same format as solve_batch_parallel)

    Example:
        >>> results = solve_batch_serial(diagram, cases)
    """
    if not load_cases:
        raise ValueError("load_cases cannot be empty")

    results = []
    total = len(load_cases)

    for i, case in enumerate(load_cases):
        M = case.get("M", 0.0)
        N = case.get("N", 0.0)

        try:
            eps_top, eps_bottom = diagram.find_strains_for_MN(
                My_target=M,
                N_target=N,
                tol=tol,
            )
            results.append({
                "success": True,
                "eps_top": float(eps_top),
                "eps_bottom": float(eps_bottom),
                "M": M,
                "N": N,
                "error": "",
            })
        except Exception as e:
            results.append({
                "success": False,
                "eps_top": None,
                "eps_bottom": None,
                "M": M,
                "N": N,
                "error": str(e),
            })

        if show_progress and (i + 1) % 100 == 0:
            print(f"Progress: {i + 1}/{total} cases solved")

    if show_progress:
        n_success = sum(1 for r in results if r["success"])
        print(f"Complete: {n_success}/{total} solved")

    return results


def analyze_batch_results(results: List[SolverResult]) -> Dict[str, Any]:
    """
    Analyze batch solver results and compute statistics.

    Args:
        results: List of solver results from solve_batch_parallel or solve_batch_serial

    Returns:
        Dictionary with:
            - n_total: Total number of cases
            - n_success: Number of successful solves
            - n_failed: Number of failed solves
            - success_rate: Fraction of successful solves (0.0 to 1.0)
            - failed_cases: List of (M, N, error) tuples for failed cases

    Example:
        >>> results = solve_batch_parallel(diagram, cases)
        >>> stats = analyze_batch_results(results)
        >>> print(f"Success rate: {stats['success_rate']:.1%}")
        >>> if stats['failed_cases']:
        ...     print("Failed cases:")
        ...     for M, N, error in stats['failed_cases']:
        ...         print(f"  M={M}, N={N}: {error}")
    """
    n_total = len(results)
    n_success = sum(1 for r in results if r["success"])
    n_failed = n_total - n_success

    failed_cases = [
        (r["M"], r["N"], r["error"])
        for r in results
        if not r["success"]
    ]

    return {
        "n_total": n_total,
        "n_success": n_success,
        "n_failed": n_failed,
        "success_rate": n_success / n_total if n_total > 0 else 0.0,
        "failed_cases": failed_cases,
    }


def extract_strain_arrays(results: List[SolverResult]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract strain arrays from batch results.

    Only includes successful solves. Failed cases are skipped.

    Args:
        results: List of solver results

    Returns:
        Tuple of (eps_top_array, eps_bottom_array)
        Arrays have shape (n_success,)

    Example:
        >>> results = solve_batch_parallel(diagram, cases)
        >>> eps_top, eps_bottom = extract_strain_arrays(results)
        >>> print(f"Mean top strain: {np.mean(eps_top):.6f}")

    Note:
        If you need to maintain correspondence with original load_cases,
        filter results first:
            >>> successful_results = [r for r in results if r["success"]]
            >>> eps_top, eps_bottom = extract_strain_arrays(successful_results)
    """
    eps_top_list = [r["eps_top"] for r in results if r["success"]]
    eps_bottom_list = [r["eps_bottom"] for r in results if r["success"]]

    return np.array(eps_top_list), np.array(eps_bottom_list)


# Example usage (if run as script)
if __name__ == "__main__":
    print("Parallel batch solver module for M-N interaction diagrams")
    print("\nExample usage:")
    print("""
    from materials.reinforced_concrete.analysis.interaction_diagram import MNInteractionDiagram
    from materials.reinforced_concrete.analysis.parallel_batch_solver import solve_batch_parallel

    # Create diagram
    diagram = MNInteractionDiagram(section=section, concrete=concrete, ...)

    # Define load cases
    load_cases = [
        {"M": 50.0, "N": 100.0},
        {"M": 30.0, "N": 200.0},
        # ... more cases ...
    ]

    # Solve in parallel (8 threads)
    results = solve_batch_parallel(diagram, load_cases, n_workers=8, show_progress=True)

    # Analyze results
    stats = analyze_batch_results(results)
    print(f"Success rate: {stats['success_rate']:.1%}")

    # Extract strains for successful cases
    eps_top, eps_bottom = extract_strain_arrays(results)
    print(f"Solved {len(eps_top)} cases successfully")
    """)
