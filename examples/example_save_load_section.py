"""
Example: Save and Load RC Section to/from JSON

Demonstrates how to serialize and deserialize:
- Section geometry (Shapely Polygon)
- Rebar groups with positions
- Material properties (Concrete, Steel)
- Optional: Applied forces/loads

Uses Pydantic's built-in serialization with custom handlers for Shapely.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from shapely.geometry import Polygon
from shapely import wkt

from materials.reinforced_concrete.geometry import (
    RCSection,
    RebarGroup,
    create_rectangular_section,
    create_circular_section,
)
from materials.reinforced_concrete.materials import ConcreteMaterial, Rebar
from materials.core.geometry import Point2D
from materials.reinforced_concrete.geometry.rebar_layer import create_linear_rebar_layer


class RCSectionSerializer:
    """
    Serializer for RC Section with all properties.

    Handles Shapely Polygon conversion to/from WKT (Well-Known Text) format.
    """

    @staticmethod
    def section_to_dict(
        section: RCSection,
        concrete: Optional[ConcreteMaterial] = None,
        applied_loads: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Convert RC Section to JSON-serializable dictionary.

        Args:
            section: RC Section to serialize
            concrete: Optional concrete material properties
            applied_loads: Optional dict of applied loads (N, Mx, My, V, etc.) in kN and kN·m

        Returns:
            Dictionary ready for JSON serialization
        """
        data: Dict[str, Any] = {
            "version": "1.0",
            "section": {
                # Convert Shapely Polygon to WKT (Well-Known Text)
                "outline_wkt": section.outline.wkt,
                "concrete_cover": section.concrete_cover,
                "section_name": section.section_name,

                # Serialize rebar groups
                "rebar_groups": [
                    {
                        # Pydantic models support model_dump()
                        # Exclude computed fields to allow reloading
                        "rebar": group.rebar.model_dump(exclude={'f_yk', 'f_yd', 'f_yd_accidental', 'E_s', 'f_t',
                                                                  'epsilon_yk', 'epsilon_yd', 'epsilon_uk', 'epsilon_ud',
                                                                  'k_ratio', 'ductility_class', 'area', 'perimeter'}),
                        "positions": [
                            {"x": pos.x, "y": pos.y} for pos in group.positions
                        ],
                        "layer_name": group.layer_name,
                    }
                    for group in section.rebar_groups
                ],
            },
        }

        # Add concrete material if provided
        if concrete is not None:
            # Exclude all computed fields from concrete
            data["concrete"] = concrete.model_dump(
                exclude={'f_ck', 'f_ck_cube', 'f_cm', 'f_cd', 'f_cd_accidental', 'f_ctm',
                        'f_ctk_005', 'f_ctk_095', 'f_ctd', 'E_cm', 'epsilon_c1', 'epsilon_cu1',
                        'epsilon_c2', 'epsilon_cu2', 'n', 'epsilon_c3', 'epsilon_cu3'}
            )

        # Add applied loads if provided
        if applied_loads is not None:
            data["applied_loads"] = applied_loads

        return data

    @staticmethod
    def dict_to_section(data: Dict[str, Any]) -> tuple[RCSection, Optional[ConcreteMaterial], Optional[Dict[str, float]]]:
        """
        Reconstruct RC Section from dictionary.

        Args:
            data: Dictionary from JSON

        Returns:
            Tuple of (section, concrete, applied_loads)
        """
        section_data = data["section"]

        # Reconstruct Shapely Polygon from WKT
        outline = wkt.loads(section_data["outline_wkt"])

        # Create base section (without rebars initially)
        section = RCSection(
            outline=outline,
            concrete_cover=section_data["concrete_cover"],
            section_name=section_data.get("section_name"),
            rebar_groups=[],  # Add later to avoid validation issues
        )

        # Reconstruct rebar groups
        for group_data in section_data["rebar_groups"]:
            # Reconstruct Rebar from dict
            rebar = Rebar.model_validate(group_data["rebar"])

            # Reconstruct positions
            positions = [
                Point2D(x=pos["x"], y=pos["y"])
                for pos in group_data["positions"]
            ]

            # Create rebar group
            rebar_group = RebarGroup(
                rebar=rebar,
                positions=positions,
                layer_name=group_data.get("layer_name"),
            )

            section.add_rebar_group(rebar_group)

        # Reconstruct concrete material if present
        concrete = None
        if "concrete" in data:
            concrete = ConcreteMaterial.model_validate(data["concrete"])

        # Get applied loads if present
        applied_loads = data.get("applied_loads")

        return section, concrete, applied_loads

    @staticmethod
    def save_to_json(
        file_path: str | Path,
        section: RCSection,
        concrete: Optional[ConcreteMaterial] = None,
        applied_loads: Optional[Dict[str, float]] = None,
        indent: int = 2,
    ) -> None:
        """
        Save RC Section to JSON file.

        Args:
            file_path: Output file path
            section: RC Section to save
            concrete: Optional concrete material
            applied_loads: Optional applied loads dict
            indent: JSON indentation
        """
        data = RCSectionSerializer.section_to_dict(section, concrete, applied_loads)

        file_path = Path(file_path)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent)

        print(f"[OK] Saved section to: {file_path}")

    @staticmethod
    def load_from_json(
        file_path: str | Path,
    ) -> tuple[RCSection, Optional[ConcreteMaterial], Optional[Dict[str, float]]]:
        """
        Load RC Section from JSON file.

        Args:
            file_path: Input file path

        Returns:
            Tuple of (section, concrete, applied_loads)
        """
        file_path = Path(file_path)
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        section, concrete, applied_loads = RCSectionSerializer.dict_to_section(data)

        print(f"[OK] Loaded section from: {file_path}")
        return section, concrete, applied_loads


def example_save_load_rectangular_beam():
    """Example 1: Rectangular beam with bottom and top reinforcement."""
    print("\n" + "=" * 60)
    print("Example 1: Rectangular Beam")
    print("=" * 60)

    # Create rectangular section
    section = create_rectangular_section(300, 500, section_name="Beam B1")

    # Add bottom reinforcement (tension zone)
    bottom_rebar = Rebar(diameter=20, grade="B500B")
    bottom_layer = create_linear_rebar_layer(
        rebar=bottom_rebar,
        n_bars=3,
        start_point=(50, 50),
        end_point=(250, 50),
        layer_name="bottom",
    )
    section.add_rebar_group(bottom_layer)

    # Add top reinforcement (compression zone)
    top_rebar = Rebar(diameter=16, grade="B500B")
    top_layer = create_linear_rebar_layer(
        rebar=top_rebar,
        n_bars=2,
        start_point=(100, 450),
        end_point=(200, 450),
        layer_name="top",
    )
    section.add_rebar_group(top_layer)

    # Define concrete material
    concrete = ConcreteMaterial(grade="C30/37")

    # Define applied loads (example)
    applied_loads = {
        "N_Ed_kN": 100.0,  # Axial compression
        "M_Ed_kNm": 150.0,  # Bending moment
        "V_Ed_kN": 80.0,    # Shear force
    }

    # Save to JSON
    RCSectionSerializer.save_to_json(
        file_path="beam_b1.json",
        section=section,
        concrete=concrete,
        applied_loads=applied_loads,
        indent=2,
    )

    # Load back from JSON
    print("\nLoading back from JSON...")
    loaded_section, loaded_concrete, loaded_loads = RCSectionSerializer.load_from_json("beam_b1.json")

    # Verify
    print(f"\nOriginal section: {section}")
    print(f"Loaded section:   {loaded_section}")
    print(f"\nOriginal concrete: {concrete}")
    print(f"Loaded concrete:   {loaded_concrete}")
    print(f"\nApplied loads: {loaded_loads}")

    # Check properties match
    assert abs(section.get_area() - loaded_section.get_area()) < 1e-6
    assert abs(section.total_steel_area - loaded_section.total_steel_area) < 1e-6
    assert len(section.rebar_groups) == len(loaded_section.rebar_groups)
    print("\n[OK] All properties match!")


def example_save_load_circular_column():
    """Example 2: Circular column with spiral reinforcement."""
    print("\n" + "=" * 60)
    print("Example 2: Circular Column")
    print("=" * 60)

    # Create circular section
    section = create_circular_section(
        diameter=400,
        n_points=64,
        section_name="Column C1"
    )

    # Add perimeter bars (8 bars around circumference)
    rebar = Rebar(diameter=20, grade="B500B")

    # Calculate positions for 8 bars evenly spaced on circumference
    import numpy as np
    radius = 400 / 2 - 40  # 40mm cover
    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    positions = [
        Point2D(x=radius * np.cos(a), y=radius * np.sin(a))
        for a in angles
    ]

    perimeter_group = RebarGroup(
        rebar=rebar,
        positions=positions,
        layer_name="perimeter",
    )
    section.add_rebar_group(perimeter_group)

    # Define high-strength concrete
    concrete = ConcreteMaterial(
        grade="C40/50",
        alpha_cc=0.85,  # German NA value
    )

    # Define applied loads (biaxial)
    applied_loads = {
        "N_Ed_kN": 2000.0,   # High axial load
        "Mx_Ed_kNm": 120.0,  # Moment about x-axis
        "My_Ed_kNm": 80.0,   # Moment about y-axis
    }

    # Save to JSON
    RCSectionSerializer.save_to_json(
        file_path="column_c1.json",
        section=section,
        concrete=concrete,
        applied_loads=applied_loads,
    )

    # Load back
    loaded_section, loaded_concrete, loaded_loads = RCSectionSerializer.load_from_json("column_c1.json")

    print(f"\nOriginal: {section}")
    print(f"Loaded:   {loaded_section}")
    print(f"\nConcrete: {loaded_concrete}")
    print(f"Applied loads: {loaded_loads}")
    print("\n[OK] Circular section saved and loaded successfully!")


def example_save_load_minimal():
    """Example 3: Minimal section without materials or loads."""
    print("\n" + "=" * 60)
    print("Example 3: Minimal Section (geometry only)")
    print("=" * 60)

    # Create section with rebars only
    section = create_rectangular_section(250, 400, section_name="Minimal")

    rebar = Rebar(diameter=16, grade="B500B")
    positions = [
        Point2D(x=40, y=40),
        Point2D(x=210, y=40),
    ]
    group = RebarGroup(rebar=rebar, positions=positions, layer_name="bottom")
    section.add_rebar_group(group)

    # Save without concrete or loads
    RCSectionSerializer.save_to_json(
        file_path="section_minimal.json",
        section=section,
    )

    # Load back
    loaded_section, loaded_concrete, loaded_loads = RCSectionSerializer.load_from_json("section_minimal.json")

    print(f"\nLoaded: {loaded_section}")
    print(f"Concrete: {loaded_concrete}")  # Should be None
    print(f"Loads: {loaded_loads}")  # Should be None
    print("\n[OK] Minimal section works!")


def example_inspect_json():
    """Example 4: Show what the JSON looks like."""
    print("\n" + "=" * 60)
    print("Example 4: JSON Structure")
    print("=" * 60)

    # Create simple section
    section = create_rectangular_section(300, 500, section_name="Example")
    rebar = Rebar(diameter=20, grade="B500B")
    positions = [Point2D(x=50, y=50), Point2D(x=250, y=50)]
    group = RebarGroup(rebar=rebar, positions=positions, layer_name="bottom")
    section.add_rebar_group(group)

    concrete = ConcreteMaterial(grade="C30/37")
    applied_loads = {"N_Ed_kN": 500.0, "M_Ed_kNm": 200.0}

    # Convert to dict
    data = RCSectionSerializer.section_to_dict(section, concrete, applied_loads)

    print("\nJSON structure:")
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("RC SECTION SAVE/LOAD EXAMPLES")
    print("=" * 60)

    # Run examples
    example_save_load_rectangular_beam()
    example_save_load_circular_column()
    example_save_load_minimal()
    example_inspect_json()

    print("\n" + "=" * 60)
    print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - beam_b1.json (rectangular beam with materials and loads)")
    print("  - column_c1.json (circular column with biaxial loads)")
    print("  - section_minimal.json (geometry only)")
    print("\nYou can now:")
    print("  1. Save any RC section with materials and loads")
    print("  2. Load them back later for analysis")
    print("  3. Share section definitions between projects")
    print("  4. Archive section designs with full metadata")
