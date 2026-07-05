"""
Pydantic models and enums for benchmark input/output files.
"""

from datetime import datetime
from enum import StrEnum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class AnalysisType(StrEnum):
    """Types of analysis that can be benchmarked."""
    INTERACTION_DIAGRAM = "interaction_diagram"
    BIAXIAL_INTERACTION = "biaxial_interaction"
    LOAD_CASE_UTILISATION = "load_case_utilisation"
    CRACKING = "cracking"
    SHEAR_CAPACITY = "shear_capacity"


class LimitState(StrEnum):
    """Limit state context for an analysis."""
    ULS_PT = "ULS_PT"       # Persistent / Transient
    ULS_ACC = "ULS_ACC"     # Accidental
    SLS_CHAR = "SLS_CHAR"   # Characteristic
    SLS_QP = "SLS_QP"       # Quasi-Permanent


# ---------------------------------------------------------------------------
# Input models (benchmark JSON file)
# ---------------------------------------------------------------------------

class ConcreteSpec(BaseModel):
    """Concrete material specification from benchmark file."""
    grade: str
    model_type: str = "parabola_rectangle"
    alpha_cc: float = 0.85
    alpha_ct: float = 1.0
    gamma_c: float = 1.5
    gamma_c_accidental: float = 1.2


class ReinforcementSpec(BaseModel):
    """Reinforcement specification from benchmark file."""
    grade: str = "B500B"
    model_type: str = "inclined"
    gamma_s: float = 1.15
    gamma_s_accidental: float = 1.0
    E_s: float = 200_000.0
    layout: list[list[float]] = Field(
        ...,
        description="List of [x, y, diameter] for each bar",
    )


class AnalysisSpec(BaseModel):
    """A single analysis to run and compare."""
    type: AnalysisType
    limit_state: LimitState = LimitState.ULS_PT
    n_points: Optional[int] = None
    tolerance: float = Field(
        default=10.0,
        description="Hausdorff distance tolerance for PASS/FAIL",
    )
    results: Any = Field(
        ...,
        description="External reference results (format depends on analysis type)",
    )


class BenchmarkFile(BaseModel):
    """Top-level schema for a benchmark JSON file."""
    name: str = ""
    source: str = ""
    description: str = ""
    outline_coords: list[list[float]]
    holes: Optional[list[list[list[float]]]] = None
    concrete: ConcreteSpec
    reinforcement: ReinforcementSpec
    analyses: list[AnalysisSpec]


# ---------------------------------------------------------------------------
# Output models (comparison results)
# ---------------------------------------------------------------------------

class PointError(BaseModel):
    """Per-point nearest-neighbour error between external and internal curves."""
    ext_M: float
    ext_N: float
    int_M: float
    int_N: float
    distance: float


class AnalysisResult(BaseModel):
    """Result of a single analysis comparison."""
    type: AnalysisType
    limit_state: LimitState
    status: str  # "PASS" or "FAIL"
    hausdorff_distance: float
    tolerance: float
    external_points: list[list[float]]
    internal_points: list[list[float]]
    per_point_errors: list[PointError] = Field(default_factory=list)


class BenchmarkResult(BaseModel):
    """Top-level result for an entire benchmark file."""
    benchmark_name: str
    source: str = ""
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    analyses: list[AnalysisResult] = Field(default_factory=list)
