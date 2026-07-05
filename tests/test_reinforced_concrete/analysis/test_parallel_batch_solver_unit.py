"""
Deterministic unit tests for parallel batch solver helpers.
"""

from __future__ import annotations

import pytest

from section_design_checks.reinforced_concrete.analysis import parallel_batch_solver as pbs


class _FakeDiagram:
    def __init__(self, *, fail_for: set[tuple[float, float]] | None = None):
        self.fail_for = fail_for or set()
        self.calls = []

    def find_strains_for_MN(self, *, My_target, N_target, tol):
        self.calls.append((My_target, N_target, tol))
        if (My_target, N_target) in self.fail_for:
            raise RuntimeError(f"failed at M={My_target}, N={N_target}")
        return (My_target / 1000.0, N_target / 1000.0)


class TestSolveSingleCase:
    """Tests for TestSolveSingleCase."""
    def test_success(self):
        """Test success."""
        diagram = _FakeDiagram()
        out = pbs._solve_single_case(diagram, {"M": 50.0, "N": 100.0}, tol=1e-5)

        assert out["success"] is True
        assert out["eps_top"] == pytest.approx(0.05, rel=1e-12)
        assert out["eps_bottom"] == pytest.approx(0.1, rel=1e-12)
        assert out["M"] == pytest.approx(50.0, rel=1e-12)
        assert out["N"] == pytest.approx(100.0, rel=1e-12)
        assert out["error"] == ""

    def test_failure(self):
        """Test failure."""
        diagram = _FakeDiagram(fail_for={(50.0, 100.0)})
        out = pbs._solve_single_case(diagram, {"M": 50.0, "N": 100.0}, tol=1e-6)

        assert out["success"] is False
        assert out["eps_top"] is None
        assert out["eps_bottom"] is None
        assert out["M"] == pytest.approx(50.0, rel=1e-12)
        assert out["N"] == pytest.approx(100.0, rel=1e-12)
        assert "failed at M=50.0, N=100.0" in out["error"]


class TestParallelBatchSolver:
    """Tests for TestParallelBatchSolver."""
    def test_empty_load_cases_raise(self):
        """Test empty load cases raise."""
        with pytest.raises(ValueError, match="load_cases cannot be empty"):
            pbs.solve_batch_parallel(_FakeDiagram(), [])

    def test_parallel_solver_preserves_input_order_and_reports_progress(self, monkeypatch, capsys):
        """Test parallel solver preserves input order and reports progress."""
        monkeypatch.setattr(pbs.os, "cpu_count", lambda: None)  # Force fallback to 4 workers

        diagram = _FakeDiagram(fail_for={(30.0, 40.0)})
        load_cases = [
            {"M": 10.0, "N": 20.0},
            {"M": 30.0, "N": 40.0},
            {"M": 50.0, "N": 60.0},
        ]

        out = pbs.solve_batch_parallel(
            diagram=diagram,
            load_cases=load_cases,
            n_workers=None,
            tol=1e-7,
            show_progress=True,
        )

        # Ensure output aligns with original load case order, not completion order.
        assert [r["M"] for r in out] == [10.0, 30.0, 50.0]
        assert [r["N"] for r in out] == [20.0, 40.0, 60.0]
        assert out[0]["success"] is True
        assert out[1]["success"] is False
        assert out[2]["success"] is True

        captured = capsys.readouterr().out
        assert "Solving 3 cases using 4 threads" in captured
        assert "Complete: 2 solved, 1 failed" in captured


class TestSerialBatchSolver:
    """Tests for TestSerialBatchSolver."""
    def test_empty_load_cases_raise(self):
        """Test empty load cases raise."""
        with pytest.raises(ValueError, match="load_cases cannot be empty"):
            pbs.solve_batch_serial(_FakeDiagram(), [])

    def test_serial_solver_success_failure_and_progress_messages(self, capsys):
        """Test serial solver success failure and progress messages."""
        fail_case = (99.0, 198.0)
        diagram = _FakeDiagram(fail_for={fail_case})
        load_cases = [{"M": float(i), "N": float(2 * i)} for i in range(1, 101)]

        out = pbs.solve_batch_serial(
            diagram=diagram,
            load_cases=load_cases,
            tol=1e-8,
            show_progress=True,
        )

        assert len(out) == 100
        assert out[0]["success"] is True
        assert out[98]["M"] == pytest.approx(99.0, rel=1e-12)
        assert out[98]["success"] is False
        assert out[99]["success"] is True

        captured = capsys.readouterr().out
        assert "Progress: 100/100 cases solved" in captured
        assert "Complete: 99/100 solved" in captured


class TestBatchResultUtilities:
    """Tests for TestBatchResultUtilities."""
    def test_analyze_batch_results_handles_empty_and_mixed(self):
        """Test analyze batch results handles empty and mixed."""
        empty_stats = pbs.analyze_batch_results([])
        assert empty_stats["n_total"] == 0
        assert empty_stats["n_success"] == 0
        assert empty_stats["n_failed"] == 0
        assert empty_stats["success_rate"] == pytest.approx(0.0, rel=1e-12)
        assert empty_stats["failed_cases"] == []

        mixed = [
            {"success": True, "M": 10.0, "N": 20.0, "error": ""},
            {"success": False, "M": 30.0, "N": 40.0, "error": "boom"},
        ]
        stats = pbs.analyze_batch_results(mixed)
        assert stats["n_total"] == 2
        assert stats["n_success"] == 1
        assert stats["n_failed"] == 1
        assert stats["success_rate"] == pytest.approx(0.5, rel=1e-12)
        assert stats["failed_cases"] == [(30.0, 40.0, "boom")]

    def test_extract_strain_arrays_filters_failed_cases(self):
        """Test extract strain arrays filters failed cases."""
        results = [
            {"success": True, "eps_top": 0.001, "eps_bottom": -0.002},
            {"success": False, "eps_top": None, "eps_bottom": None},
            {"success": True, "eps_top": 0.003, "eps_bottom": -0.004},
        ]

        eps_top, eps_bottom = pbs.extract_strain_arrays(results)

        assert eps_top.shape == (2,)
        assert eps_bottom.shape == (2,)
        assert eps_top.tolist() == [0.001, 0.003]
        assert eps_bottom.tolist() == [-0.002, -0.004]


def test_module_main_block_prints_usage(capsys):
    """Test module main block prints usage."""
    import runpy

    runpy.run_module("section_design_checks.reinforced_concrete.analysis.parallel_batch_solver", run_name="__main__")
    out = capsys.readouterr().out
    assert "Parallel batch solver module for M-N interaction diagrams" in out
    assert "Example usage" in out
