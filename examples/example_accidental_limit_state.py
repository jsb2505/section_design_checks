"""
Example demonstrating the use of accidental limit state in concrete stress-strain models.

This example shows how to create concrete stress-strain models for different limit states:
1. ULS Persistent/Transient (standard design) - uses gamma_c = 1.5
2. ULS Accidental (exceptional events) - uses gamma_c_accidental = 1.2

The accidental limit state provides higher design strength due to the reduced partial factor,
which is appropriate for exceptional loading conditions like impact or explosion.
"""

from materials.reinforced_concrete.materials import ConcreteMaterial
from materials.reinforced_concrete.constitutive import create_concrete_stress_strain


def main():
    # Create concrete material with both partial factors defined
    concrete = ConcreteMaterial(
        grade="C30/37",
        gamma_c=1.5,              # Standard partial factor for ULS persistent/transient
        gamma_c_accidental=1.2,   # Reduced partial factor for ULS accidental
        alpha_cc=0.85,            # Long-term strength reduction
    )

    print("=" * 70)
    print("Concrete Material Properties: C30/37")
    print("=" * 70)
    print(f"Characteristic strength (f_ck):           {concrete.f_ck:.2f} MPa")
    print(f"Design strength - persistent (f_cd):      {concrete.f_cd:.2f} MPa")
    print(f"Design strength - accidental (f_cd_acc):  {concrete.f_cd_accidental:.2f} MPa")
    print(f"\nStrength increase for accidental:         {(concrete.f_cd_accidental / concrete.f_cd - 1) * 100:.1f}%")
    print()

    # Create stress-strain models for different limit states
    print("=" * 70)
    print("Creating Stress-Strain Models")
    print("=" * 70)

    # 1. Standard design (ULS persistent/transient)
    model_persistent = create_concrete_stress_strain(
        concrete=concrete,
        model_type="parabola-rectangle",
        use_characteristic=False,  # Use design strength
        use_accidental=False,      # Standard partial factor
    )
    print(f"1. ULS Persistent/Transient model: f_c = {model_persistent.f_c:.2f} MPa")

    # 2. Accidental design (ULS accidental)
    model_accidental = create_concrete_stress_strain(
        concrete=concrete,
        model_type="parabola-rectangle",
        use_characteristic=False,  # Use design strength
        use_accidental=True,       # Accidental partial factor
    )
    print(f"2. ULS Accidental model:           f_c = {model_accidental.f_c:.2f} MPa")

    # 3. Characteristic (for comparison)
    model_characteristic = create_concrete_stress_strain(
        concrete=concrete,
        model_type="parabola-rectangle",
        use_characteristic=True,   # Use characteristic strength
        use_accidental=False,
    )
    print(f"3. Characteristic model:           f_c = {model_characteristic.f_c:.2f} MPa")
    print()

    # Demonstrate stress calculation at a typical strain
    test_strain = 0.002  # 2 per mille (typical design strain)
    print("=" * 70)
    print(f"Stress at strain = {test_strain:.4f} (2 per mille)")
    print("=" * 70)
    stress_persistent = model_persistent.get_stress(test_strain)
    stress_accidental = model_accidental.get_stress(test_strain)
    stress_characteristic = model_characteristic.get_stress(test_strain)

    print(f"ULS Persistent/Transient: {stress_persistent:.2f} MPa")
    print(f"ULS Accidental:           {stress_accidental:.2f} MPa")
    print(f"Characteristic:           {stress_characteristic:.2f} MPa")
    print()

    print("=" * 70)
    print("Usage Notes")
    print("=" * 70)
    print("- Use ULS Persistent/Transient for normal design situations")
    print("- Use ULS Accidental for exceptional events (impact, explosion, etc.)")
    print("- The accidental limit state uses reduced partial factors per EC2")
    print("- This can be applied in interaction_diagram.py, bending_check.py, etc.")
    print("=" * 70)


if __name__ == "__main__":
    main()
