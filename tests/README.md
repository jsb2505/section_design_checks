# Test Suite Documentation

## Overview

Tests are organized to mirror source domains so files are easy to find and maintain.

## Structure

```text
tests/
|-- conftest.py
|-- test_core/
|-- test_utils/
|-- test_api/
|-- test_reinforced_concrete/
|   |-- analysis/
|   |-- code_checks/
|   |-- constitutive/
|   |-- geometry/
|   |-- materials/
|   |-- ndp/
|   `-- thermal/
`-- integration/
```

## Naming Conventions

- File names: `test_<module_or_feature>.py`
- Test names: `test_<behavior>_<scenario>()`
- Module docstring: state what source module/feature is covered
- Assertion messages: include context when checking computed values or branch behavior

## Running Tests

```bash
# Run all tests
pytest

# Run all reinforced concrete tests
pytest tests/test_reinforced_concrete

# Run one grouped folder
pytest tests/test_reinforced_concrete/materials

# Run one test file
pytest tests/test_reinforced_concrete/materials/test_concrete_material.py

# Run one class
pytest tests/test_reinforced_concrete/materials/test_concrete_material.py::TestConcreteMaterial

# Run one test
pytest tests/test_reinforced_concrete/materials/test_concrete_material.py::TestConcreteMaterial::test_create_c30_concrete
```

## Coverage

```bash
pytest --cov=section_design_checks --cov-report=term-missing
```
