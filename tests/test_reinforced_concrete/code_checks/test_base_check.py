"""
Tests for reinforced_concrete.code_checks.base_check module.
"""

import pytest
from pydantic import ValidationError

from section_design_checks.reinforced_concrete.code_checks.base_check import (
    BaseCodeCheck,
    CheckResult,
    CheckStatus,
)


class TestCheckStatus:
    """Tests for CheckStatus enum."""

    def test_status_values(self):
        """Test that all status values are defined."""
        assert CheckStatus.PASS == "pass"
        assert CheckStatus.FAIL == "fail"
        assert CheckStatus.WARNING == "warning"
        assert CheckStatus.NOT_APPLICABLE == "not_applicable"


class TestCheckResult:
    """Tests for CheckResult class."""

    def test_create_result(self):
        """Test creating a check result."""
        result = CheckResult(
            check_name="Bending capacity",
            status=CheckStatus.PASS,
            utilization=0.75,
            demand=150.0,
            capacity=200.0,
            units="kN·m",
            message="Check passed",
            code_reference="EC2 §6.1",
        )

        assert result.check_name == "Bending capacity"
        assert result.status == CheckStatus.PASS
        assert result.utilization == 0.75
        assert result.demand == 150.0
        assert result.capacity == 200.0
        assert result.units == "kN·m"
        assert result.message == "Check passed"
        assert result.code_reference == "EC2 §6.1"

    def test_result_minimal(self):
        """Test creating result with minimal fields."""
        result = CheckResult(
            check_name="Test",
            status=CheckStatus.PASS,
        )

        assert result.check_name == "Test"
        assert result.status == CheckStatus.PASS
        assert result.utilization is None
        assert result.demand is None
        assert result.capacity is None

    def test_utilization_validation(self):
        """Test that utilization must be non-negative."""
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            CheckResult(
                check_name="Test",
                status=CheckStatus.PASS,
                utilization=-0.5,  # Negative
            )

    def test_repr(self):
        """Test __repr__ method."""
        result = CheckResult(
            check_name="Test Check",
            status=CheckStatus.PASS,
            utilization=0.75,
        )

        r = repr(result)
        assert "Test Check" in r
        assert "pass" in r
        assert ("75" in r and "%" in r) or "0.75" in r

    def test_str_with_utilization(self):
        """Test __str__ with utilization."""
        result = CheckResult(
            check_name="Bending",
            status=CheckStatus.PASS,
            utilization=0.75,
            demand=150.0,
            capacity=200.0,
            units="kN·m",
        )

        s = str(result)
        assert "Bending" in s
        assert "PASS" in s
        assert "75" in s  # Utilization percentage

    def test_str_with_message(self):
        """Test __str__ with message."""
        result = CheckResult(
            check_name="Shear",
            status=CheckStatus.WARNING,
            message="High utilization",
        )

        s = str(result)
        assert "Shear" in s
        assert "WARNING" in s
        assert "High utilization" in s

    def test_json_serialization(self):
        """Test JSON serialization."""
        result = CheckResult(
            check_name="Test",
            status=CheckStatus.PASS,
            utilization=0.75,
            demand=100.0,
            capacity=133.33,
        )

        json_data = result.model_dump()
        assert json_data["check_name"] == "Test"
        assert json_data["status"] == "pass"
        assert json_data["utilization"] == pytest.approx(0.75)

    def test_str_vector_summary_fallback_keys_and_missing_components(self):
        """Test str vector summary fallback keys and missing components."""
        result = CheckResult(
            check_name="Vector Check",
            status=CheckStatus.PASS,
            utilization=0.5,
            demand_components={"X": 10.0, "Y": 20.0},
            capacity_components={"X": 50.0},
            units_components={"X": "kN"},
        )
        s = str(result)
        assert "X: 10.00/50.00 kN" in s
        assert "Y:" not in s


class TestBaseCodeCheck:
    """Tests for BaseCodeCheck abstract class."""

    class ConcreteCheck(BaseCodeCheck):
        """Test implementation of BaseCodeCheck."""

        def perform_check(self, demand: float = 100.0, capacity: float = 200.0) -> CheckResult:
            return self._create_result(
                check_name="Test Check",
                demand=demand,
                capacity=capacity,
                units="kN",
                code_reference="Test §1.0",
            )

    def test_create_concrete_check(self):
        """Test creating concrete implementation."""
        check = self.ConcreteCheck()
        assert isinstance(check, BaseCodeCheck)

    def test_perform_check(self):
        """Test performing a check."""
        check = self.ConcreteCheck()
        result = check.perform_check(demand=100.0, capacity=200.0)

        assert isinstance(result, CheckResult)
        assert result.status == CheckStatus.PASS
        assert result.utilization == pytest.approx(0.5)

    def test_create_result_pass(self):
        """Test _create_result for passing check."""
        check = self.ConcreteCheck()
        result = check.perform_check(demand=80.0, capacity=100.0)

        assert result.status == CheckStatus.PASS
        assert result.utilization == pytest.approx(0.8)
        assert "satisfied" in result.message.lower()

    def test_create_result_warning(self):
        """Test _create_result for warning (high utilization)."""
        check = self.ConcreteCheck()
        result = check.perform_check(demand=96.0, capacity=100.0)

        # Default warning threshold is 0.95
        assert result.status == CheckStatus.WARNING
        assert result.utilization == pytest.approx(0.96)
        assert "utilization" in result.message.lower()

    def test_create_result_fail(self):
        """Test _create_result for failing check."""
        check = self.ConcreteCheck()
        result = check.perform_check(demand=120.0, capacity=100.0)

        assert result.status == CheckStatus.FAIL
        assert result.utilization == pytest.approx(1.2)
        assert "exceeded" in result.message.lower()

    def test_create_result_custom_threshold(self):
        """Test custom warning threshold."""
        self.ConcreteCheck()

        class CustomCheck(BaseCodeCheck):
            def perform_check(self) -> CheckResult:
                return self._create_result(
                    check_name="Custom",
                    demand=85.0,
                    capacity=100.0,
                    units="kN",
                    code_reference="Test",
                    warning_threshold=0.8,  # Custom threshold
                )

        custom = CustomCheck()
        result = custom.perform_check()

        # 0.85 > 0.8, so should be WARNING
        assert result.status == CheckStatus.WARNING

    def test_create_result_custom_message(self):
        """Test custom message."""
        self.ConcreteCheck()

        class MessageCheck(BaseCodeCheck):
            def perform_check(self) -> CheckResult:
                return self._create_result(
                    check_name="Message",
                    demand=50.0,
                    capacity=100.0,
                    units="kN",
                    code_reference="Test",
                    message="Custom message",
                )

        msg_check = MessageCheck()
        result = msg_check.perform_check()

        assert result.message == "Custom message"

    def test_create_result_with_details(self):
        """Test including additional details."""
        self.ConcreteCheck()

        class DetailCheck(BaseCodeCheck):
            def perform_check(self) -> CheckResult:
                return self._create_result(
                    check_name="Detail",
                    demand=50.0,
                    capacity=100.0,
                    units="kN",
                    code_reference="Test",
                    details={"method": "simplified", "factor": 1.5},
                )

        detail_check = DetailCheck()
        result = detail_check.perform_check()

        assert result.details["method"] == "simplified"
        assert result.details["factor"] == 1.5

    def test_zero_capacity(self):
        """Test handling of zero capacity."""
        check = self.ConcreteCheck()
        result = check.perform_check(demand=100.0, capacity=0.0)

        # Should handle gracefully (infinite utilization)
        assert result.status == CheckStatus.FAIL
        assert result.utilization == float('inf')

    def test_create_result_missing_inputs_raises(self):
        """Test create result missing inputs raises."""
        check = self.ConcreteCheck()
        with pytest.raises(ValueError, match="Provide either utilization"):
            check._create_result(
                check_name="Bad",
                code_reference="Test",
                demand=None,
                capacity=None,
            )


def test_abstract_perform_check_default_body_returns_none():
    # Call the abstract method body directly to exercise the default no-op branch.
    """Test abstract perform check default body returns none."""
    assert BaseCodeCheck.perform_check(object()) is None
