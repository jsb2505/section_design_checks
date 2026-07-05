# Code Checks Implementation Guide

This document explains how to implement structural code checks using the materials library, covering both **first principles** and **codified** approaches.

## Overview

The materials library supports two fundamentally different approaches to code checks:

1. **First Principles** - Strain compatibility & force equilibrium (e.g., bending/M-N checks)
2. **Codified Approach** - Empirical formulas with business logic (e.g., shear checks)

Both approaches use the same underlying infrastructure:
- ✅ Geometry (`RCSection` with arbitrary shapes)
- ✅ Materials (`ConcreteMaterial`, `ReinforcingSteel`, `ConcreteAge`)
- ✅ Constitutive models (stress-strain relationships per EC2)
- ✅ Base check framework (`BaseCodeCheck`, `CheckResult`)

---

## Approach 1: First Principles (Bending/M-N Checks)

### Concept

Uses fundamental mechanics:
- **Strain compatibility**: Plane sections remain plane (ε varies linearly)
- **Force equilibrium**: ΣF = N_Ed, ΣM = M_Ed
- **Constitutive models**: Material stress-strain (with codified safety factors)

### Implementation

Already implemented via `BendingCheck` and M-N interaction diagrams:

```python
from materials.reinforced_concrete.code_checks.ec2 import BendingCheck
from materials.reinforced_concrete.geometry import create_rectangular_section
from materials.reinforced_concrete.materials import ConcreteMaterial

# Create section with reinforcement
section = create_rectangular_section(width=300, height=500)
# ... add reinforcement ...

# Create material (γ_c = 1.5 applied to get f_cd)
concrete = ConcreteMaterial(grade="C30/37", gamma_c=1.5)

# Create check
check = BendingCheck(
    section=section,
    concrete=concrete,
    concrete_model_type="parabola-rectangle",  # EC2 Fig 3.3
    steel_branch_type="inclined",              # EC2 Fig 3.8
)

# Perform check
result = check.perform_check(M_Ed=150, N_Ed=500)  # kN·m, kN

print(result)
# Bending check (EC2 §6.1): PASS (utilization: 68.5%)
```

### How It Works Internally

1. **Fiber mesh generation** - Section divided into concrete + steel fibers
2. **Assume neutral axis depth** - Defines strain distribution
3. **Calculate strains** - Linear distribution from neutral axis
4. **Get stresses** - Using constitutive models (with f_cd, f_yd)
5. **Integrate forces** - Sum fiber stresses × areas
6. **Check equilibrium** - ΣF = N, ΣM = M
7. **Generate M-N curve** - Repeat for many neutral axis depths
8. **Check capacity** - Is (N_Ed, M_Ed) inside interaction surface?

### Key Features

- ✅ **Exact** - No empirical simplifications
- ✅ **Handles complexity** - Arbitrary sections, multiple rebar layers
- ✅ **Flexible** - Different constitutive models (parabola-rectangle, bilinear, etc.)
- ✅ **Transparent** - All mechanics visible in code
- ⚠️ **Computationally intensive** - Fiber integration required

---

## Approach 2: Codified (Shear Checks)

### Concept

Uses code-prescribed empirical formulas:
- **Not first principles** - Based on experimental calibration
- **Business logic** - Different modes with specific formulas
- **Faster** - Direct calculation without iteration

### Implementation

Implemented via `ShearCheck`:

```python
from materials.reinforced_concrete.code_checks.ec2 import ShearCheck
from materials.reinforced_concrete.materials import ShearRebar

# Create shear reinforcement
shear_links = ShearRebar(
    diameter=10,
    spacing=200,
    n_legs=2,
    grade="B500B",
)

# Create check
check = ShearCheck(
    section=section,
    concrete=concrete,
    shear_reinforcement=shear_links,
    N_Ed=0,  # Axial force affects shear capacity
)

# Perform check
result = check.perform_check(V_Ed=100, cot_theta=2.5)  # kN

print(result)
# Shear check (EC2 §6.2): PASS (utilization: 72.3%)
```

### How It Works (Business Logic)

The check evaluates three different failure modes:

#### 1. **Concrete Shear Resistance** (§6.2.2, Eq. 6.2)

```
V_Rd,c = [C_Rd,c·k·(100·ρ_l·f_ck)^(1/3) + k_1·σ_cp]·b_w·d
```

Where:
- `C_Rd,c = 0.18 / γ_c`
- `k = 1 + √(200/d) ≤ 2.0` (size effect)
- `ρ_l = A_sl / (b_w·d) ≤ 0.02` (tension reinforcement ratio)
- `σ_cp` = compressive stress from axial force

**When used**: No shear reinforcement, or V_Ed < V_Rd,c

#### 2. **Shear Reinforcement Resistance** (§6.2.3, Eq. 6.8)

```
V_Rd,s = (A_sw/s)·z·f_ywd·cot(θ)
```

Where:
- `A_sw/s` = shear reinforcement ratio (area per unit length)
- `z = 0.9d` (lever arm, simplified)
- `f_ywd` = design yield strength of links
- `cot(θ)` = 1.0 to 2.5 (strut angle, user choice for economy)

**When used**: Shear reinforcement provided and V_Ed > V_Rd,c

#### 3. **Compression Strut Crushing** (§6.2.3, Eq. 6.9)

```
V_Rd,max = α_cw·b_w·z·ν·f_cd / (cot(θ) + tan(θ))
```

Where:
- `α_cw` = 1.0 to 2.5 (depends on stress level)
- `ν = 0.6·(1 - f_ck/250)` (strength reduction factor)

**When used**: Upper limit - concrete strut cannot carry more

### Business Logic Flow

```python
def perform_check(V_Ed, cot_theta):
    # Calculate all three capacities
    V_Rd_c = find_V_Rd_c()
    V_Rd_s = find_V_Rd_s(cot_theta)
    V_Rd_max = find_V_Rd_max(cot_theta)

    if shear_reinforcement is None:
        # No shear reinforcement
        V_Rd = V_Rd_c
        mode = "concrete only"
    else:
        if V_Ed > V_Rd_c:
            # Shear reinforcement engaged
            V_Rd = min(V_Rd_s, V_Rd_max)

            if V_Rd_s < V_Rd_max:
                mode = "shear reinforcement governs"
            else:
                mode = "strut crushing governs"
        else:
            # Concrete sufficient
            V_Rd = V_Rd_c
            mode = "concrete sufficient"

    return V_Ed <= V_Rd
```

### Key Features

- ✅ **Fast** - Direct calculation, no iteration
- ✅ **Code-compliant** - Follows EC2 exactly
- ✅ **Clear failure modes** - Identifies governing mechanism
- ✅ **Practical** - Used in real design practice
- ⚠️ **Empirical** - Not transparent mechanics
- ⚠️ **Limited scope** - Only covers cases code addresses

---

## Common Infrastructure

Both approaches share:

### 1. **Base Classes**

```python
class BaseCodeCheck(BaseModel, ABC):
    @abstractmethod
    def perform_check(self, **kwargs) -> CheckResult:
        pass

    def _create_result(self, check_name, demand, capacity, ...):
        # Standardized result creation
        utilization = demand / capacity
        status = PASS if utilization <= 1.0 else FAIL
        return CheckResult(...)
```

### 2. **CheckResult**

```python
@dataclass
class CheckResult:
    check_name: str
    status: CheckStatus  # PASS/FAIL/WARNING
    utilization: float   # demand/capacity
    demand: float
    capacity: float
    units: str
    message: str
    details: Dict[str, Any]
    code_reference: str
```

### 3. **Material Factors**

Both approaches use codified safety factors:

```python
# In ConcreteMaterial
f_cd = alpha_cc * f_ck / gamma_c  # γ_c = 1.5 for ULS

# In ReinforcingSteel
f_yd = f_yk / gamma_s  # γ_s = 1.15 for ULS
```

These are **built into the materials** and used by constitutive models.

---

## When to Use Each Approach

### Use First Principles When:

- ✅ Complex loading (biaxial bending, varying axial force)
- ✅ Non-standard sections (T-beams, L-sections, circular)
- ✅ Multiple reinforcement layers
- ✅ Need exact capacity curves
- ✅ Research or validation work

**Example checks**:
- Bending (M-N interaction) ✅ Implemented
- Biaxial bending (M-N-N) 🚧 Can be implemented
- Torsion (fiber-based) 🚧 Possible extension

### Use Codified Approach When:

- ✅ Code provides specific formulas
- ✅ Standard design cases
- ✅ Speed is important
- ✅ Following code exactly is required
- ✅ Empirical factors dominate (e.g., size effects)

**Example checks**:
- Shear (§6.2) ✅ Implemented
- Punching shear (§6.4) 🚧 Can be implemented
- Cracking (§7.3) 🚧 Can be implemented
- Deflection (§7.4) 🚧 Can be implemented

---

## Implementing New Checks

### Template: First Principles

```python
from materials.reinforced_concrete.code_checks.base_check import BaseCodeCheck, CheckResult

class MyFirstPrinciplesCheck(BaseCodeCheck):
    section: RCSection
    concrete: ConcreteMaterial

    def perform_check(self, applied_load: float) -> CheckResult:
        # 1. Set up fiber mesh
        mesh = create_fiber_mesh(self.section)

        # 2. Iterate to find capacity
        capacity = 0.0
        for trial_value in search_space:
            # Assume strain distribution
            strains = calculate_strains(trial_value)

            # Get stresses from constitutive models
            stresses = [
                concrete_model.get_stress(eps) for eps in strains
            ]

            # Integrate forces
            forces = sum(stress * area for stress, area in zip(stresses, areas))

            # Check equilibrium
            if equilibrium_satisfied(forces):
                capacity = calculate_capacity(forces)
                break

        # 3. Return result
        return self._create_result(
            check_name="My Check",
            demand=applied_load,
            capacity=capacity,
            units="kN",
            code_reference="EC2 §X.Y",
        )
```

### Template: Codified

```python
class MyCodeifiedCheck(BaseCodeCheck):
    section: RCSection
    concrete: ConcreteMaterial

    def find_capacity_mode_1(self) -> float:
        """EC2 §X.Y (Eq. X.Y) - Mode 1 capacity"""
        # Direct formula from code
        k = some_factor(self.section.get_property())
        return k * self.concrete.f_cd * self.section.area / 1000

    def find_capacity_mode_2(self) -> float:
        """EC2 §X.Y (Eq. X.Y) - Mode 2 capacity"""
        # Another formula
        return ...

    def perform_check(self, applied_load: float) -> CheckResult:
        # Business logic - which mode governs?
        cap1 = self.find_capacity_mode_1()
        cap2 = self.find_capacity_mode_2()

        if condition_A:
            capacity = cap1
            mode = "Mode 1"
        else:
            capacity = min(cap1, cap2)
            mode = "Governed by Mode 2"

        return self._create_result(
            check_name="My Codified Check",
            demand=applied_load,
            capacity=capacity,
            units="kN",
            code_reference="EC2 §X.Y",
            message=f"Governed by {mode}",
        )
```

---

## Complete Example

See [examples/code_checks_example.py](examples/code_checks_example.py) for a full working example showing:

1. Section definition (300×500 beam with reinforcement)
2. Material properties (C30/37 concrete, B500B steel)
3. Bending check (first principles, M-N diagram)
4. Shear check (codified, EC2 formulas)
5. Multiple load combinations

Run it:
```bash
cd materials
python examples/code_checks_example.py
```

---

## Summary

| Aspect | First Principles | Codified |
|--------|------------------|----------|
| **Basis** | Mechanics (strain compatibility, equilibrium) | Empirical formulas |
| **Transparency** | Full visibility of mechanics | Code equations only |
| **Accuracy** | Exact for given constitutive models | Calibrated to experiments |
| **Flexibility** | Handles any geometry/loading | Limited to code scope |
| **Speed** | Slower (iteration/integration) | Fast (direct calculation) |
| **Complexity** | High (implement mechanics) | Medium (implement formulas) |
| **Examples** | Bending, biaxial bending | Shear, punching, cracking |

**Both are valid and necessary** - choose based on the nature of the check and what the code prescribes.

---

## Next Steps

Potential checks to implement:

### First Principles
- [ ] Biaxial bending (M-N-N interaction surface)
- [ ] Fiber-based torsion
- [ ] Time-dependent effects (creep, shrinkage)

### Codified
- [ ] Punching shear (EC2 §6.4)
- [ ] Crack width (EC2 §7.3)
- [ ] Deflection (EC2 §7.4)
- [ ] Minimum reinforcement (EC2 §9.2)
- [ ] Detailing checks (anchorage, laps, etc.)

All can follow the patterns established in `BendingCheck` and `ShearCheck`.
