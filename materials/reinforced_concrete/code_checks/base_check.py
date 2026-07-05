"""
Base class for structural code checks.

Provides common interface for all design checks (bending, shear, etc.).
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class CheckStatus(str, Enum):
    """Status of a design check."""
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    NOT_APPLICABLE = "not_applicable"


class CheckResult(BaseModel):
    """
    Result of a design check.

    Provides standardized output for all checks.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    check_name: str = Field(..., description="Name of the check performed")
    status: CheckStatus = Field(..., description="Pass/fail status")
    utilization: float | None = Field(
        None,
        description="Utilization ratio (demand/capacity), if applicable",
        ge=0,
    )

    # Scalar demand/capacity (kept for simple checks like shear VEd/VRd)
    demand: float | None = Field(None, description="Demand value")
    capacity: float | None = Field(None, description="Capacity value")
    units: str | None = Field(None, description="Units for demand/capacity")

    # Vector demand/capacity (for M-N, M-N-V, etc.)
    demand_components: dict[str, float] | None = Field(
        default=None, description="Component demands (e.g. {'N':..., 'M':...})"
    )
    capacity_components: dict[str, float] | None = Field(
        default=None, description="Component capacities at governing point"
    )
    units_components: dict[str, str] | None = Field(
        default=None, description="Units per component (e.g. {'N':'kN','M':'kN·m'})"
    )

    message: str = Field(default="", description="Descriptive message")
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional check-specific details"
    )
    code_reference: str | None = Field(
        None,
        description="Code clause reference (e.g., 'EC2 §6.1')"
    )

    def __str__(self) -> str:
        parts = [f"{self.check_name}: {self.status.value.upper()}"]

        if self.utilization is not None:
            parts.append(f"(utilization: {self.utilization:.1%})")

        # Prefer scalar display if present
        if self.demand is not None and self.capacity is not None:
            units_str = f" {self.units}" if self.units else ""
            parts.append(f"[{self.demand:.2f}/{self.capacity:.2f}{units_str}]")
        # Otherwise show vector summary if present
        elif self.demand_components and self.capacity_components:
            # small compact summary
            keys = [k for k in ("N", "M", "V") if k in self.demand_components or k in self.capacity_components]
            if not keys:
                keys = sorted(set(self.demand_components) | set(self.capacity_components))

            comp_bits = []
            for k in keys:
                d = self.demand_components.get(k)
                c = self.capacity_components.get(k)
                u = (self.units_components or {}).get(k, "")
                if d is None or c is None:
                    continue
                u = f" {u}" if u else ""
                comp_bits.append(f"{k}: {d:.2f}/{c:.2f}{u}")
            if comp_bits:
                parts.append("[" + ", ".join(comp_bits) + "]")

        if self.message:
            parts.append(f"- {self.message}")

        return " ".join(parts)


class BaseCodeCheck(BaseModel, ABC):
    """
    Abstract base class for code checks.

    All specific checks (bending, shear, etc.) inherit from this.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
    )

    @abstractmethod
    def perform_check(self, **kwargs) -> CheckResult:
        """
        Perform the design check.

        Args:
            **kwargs: Check-specific parameters

        Returns:
            CheckResult with pass/fail status and details
        """
        pass

    def _create_result(
        self,
        *,
        check_name: str,
        code_reference: str,
        warning_threshold: float = 0.95,
        message: str = "",
        details: dict[str, Any] | None = None,

        # scalar style (old)
        demand: float | None = None,
        capacity: float | None = None,
        units: str | None = None,

        # vector style (new)
        demand_components: dict[str, float] | None = None,
        capacity_components: dict[str, float] | None = None,
        units_components: dict[str, str] | None = None,

        # override (for interaction checks)
        utilization: float | None = None,
    ) -> CheckResult:
        """
        Helper to create standardized check results.

        Args:
            check_name: Name of check
            demand: Demand value
            capacity: Capacity value
            units: Units string
            code_reference: Code clause
            warning_threshold: Utilization threshold for warning (default 0.95)
            message: Custom message
            details: Additional details

        Returns:
            CheckResult
        """
        if utilization is None:
            if demand is None or capacity is None:
                raise ValueError("Provide either utilization=... or (demand and capacity).")
            utilization = demand / capacity if capacity > 0 else float("inf")

        if utilization <= 1.0:
            if utilization >= warning_threshold:
                status = CheckStatus.WARNING
                if not message:
                    message = f"High utilization ({utilization:.1%})"
            else:
                status = CheckStatus.PASS
                if not message:
                    message = "Check satisfied"
        else:
            status = CheckStatus.FAIL
            if not message:
                message = f"Capacity exceeded by {(utilization - 1.0) * 100:.1f}%"

        return CheckResult(
            check_name=check_name,
            status=status,
            utilization=float(utilization),
            demand=demand,
            capacity=capacity,
            units=units,
            demand_components=demand_components,
            capacity_components=capacity_components,
            units_components=units_components,
            message=message,
            details=details or {},
            code_reference=code_reference,
        )
