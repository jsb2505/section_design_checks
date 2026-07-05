"""
Tests for core.base_material module.
"""

import pytest
from pydantic import ValidationError
from materials.core.base_material import BaseMaterial


class ConcreteMaterialTest(BaseMaterial):
    """Test implementation of BaseMaterial."""

    elastic_modulus: float = 30000.0

    def get_elastic_modulus(self) -> float:
        return self.elastic_modulus


class TestBaseMaterial:
    """Tests for BaseMaterial abstract class."""

    def test_create_material(self):
        """Test creating a material with valid data."""
        mat = ConcreteMaterialTest(
            name="Concrete",
            density=2400.0,
            elastic_modulus=30000.0,
        )
        assert mat.name == "Concrete"
        assert mat.density == 2400.0
        assert mat.get_elastic_modulus() == 30000.0

    def test_material_name_required(self):
        """Test that name is required."""
        with pytest.raises(ValidationError):
            ConcreteMaterialTest(
                name="",  # Empty name should fail
                density=2400.0,
                elastic_modulus=30000.0,
            )

    def test_density_positive(self):
        """Test that density must be positive."""
        with pytest.raises(ValidationError):
            ConcreteMaterialTest(
                name="Test",
                density=-100.0,  # Negative density
                elastic_modulus=30000.0,
            )

    def test_density_optional(self):
        """Test that density is optional."""
        mat = ConcreteMaterialTest(
            name="Test",
            density=None,
            elastic_modulus=30000.0,
        )
        assert mat.density is None

    def test_material_repr(self):
        """Test __repr__ method."""
        mat = ConcreteMaterialTest(name="TestMat", elastic_modulus=30000.0)
        assert "TestMat" in repr(mat)

    def test_material_str(self):
        """Test __str__ method."""
        mat = ConcreteMaterialTest(name="TestMat", elastic_modulus=30000.0)
        assert str(mat) == "TestMat"

    def test_validate_assignment(self):
        """Test that validation occurs on assignment."""
        mat = ConcreteMaterialTest(name="Test", elastic_modulus=30000.0)

        # Valid assignment
        mat.density = 2500.0
        assert mat.density == 2500.0

        # Invalid assignment
        with pytest.raises(ValidationError):
            mat.density = -100.0

    def test_extra_fields_forbidden(self):
        """Test that extra fields are not allowed."""
        with pytest.raises(ValidationError):
            ConcreteMaterialTest(
                name="Test",
                elastic_modulus=30000.0,
                extra_field="not allowed",
            )
