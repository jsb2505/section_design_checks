# Integration Tests

End-to-end integration tests for the section design checks library.

## Test Files

### Biaxial Interaction
- **[test_biaxial_utilization_vector.py](test_biaxial_utilization_vector.py)** - Tests the biaxial utilization vector method
  - 3D vector projection validation
  - Linear scaling verification
  - Multiple load case testing

- **[test_rugby_ball_surface.py](test_rugby_ball_surface.py)** - Validates biaxial surface generation
  - Proper ellipsoidal shape verification
  - Constant-N contour checking
  - Comprehensive visualization

## Running Tests

### Individual Test
```bash
python tests/integration/test_biaxial_utilization_vector.py
python tests/integration/test_rugby_ball_surface.py
```

### All Unit Tests
```bash
python -m pytest tests/unit/
```

### All Integration Tests
```bash
python -m pytest tests/integration/
```

### Full Test Suite
```bash
python run_tests.py
```

## Test Coverage

These integration tests complement the unit tests by verifying:
- End-to-end workflows
- Complex calculations
- Multiple component interactions
- Visual output validation
