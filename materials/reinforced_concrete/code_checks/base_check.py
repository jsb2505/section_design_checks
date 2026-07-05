"""
Base class for structural code checks.

Provides common interface for all design checks (bending, shear, etc.).
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict


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
    utilization: Optional[float] = Field(
        None,
        description="Utilization ratio (demand/capacity), if applicable",
        ge=0,
    )
    demand: Optional[float] = Field(None, description="Demand value")
    capacity: Optional[float] = Field(None, description="Capacity value")
    units: Optional[str] = Field(None, description="Units for demand/capacity")
    message: str = Field(default="", description="Descriptive message")
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional check-specific details"
    )
    code_reference: Optional[str] = Field(
        None,
        description="Code clause reference (e.g., 'EC2 §6.1')"
    )

    def __repr__(self) -> str:
        util_str = f", {self.utilization:.2%}" if self.utilization is not None else ""
        return f"CheckResult({self.check_name}: {self.status.value}{util_str})"

    def __str__(self) -> str:
        parts = [f"{self.check_name}: {self.status.value.upper()}"]

        if self.utilization is not None:
            parts.append(f"(utilization: {self.utilization:.1%})")

        if self.demand is not None and self.capacity is not None:
            units_str = f" {self.units}" if self.units else ""
            parts.append(f"[{self.demand:.2f}/{self.capacity:.2f}{units_str}]")

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
        check_name: str,
        demand: float,
        capacity: float,
        units: str,
        code_reference: str,
        warning_threshold: float = 0.95,
        message: str = "",
        details: Optional[Dict[str, Any]] = None,
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
        utilization = demand / capacity if capacity > 0 else float('inf')

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
            utilization=utilization,
            demand=demand,
            capacity=capacity,
            units=units,
            message=message,
            details=details or {},
            code_reference=code_reference,
        )
