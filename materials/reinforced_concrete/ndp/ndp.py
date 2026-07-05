"""
Nationally Determined Parameters (NDP) data for Eurocode 2.

Structure:
- _NDP_METADATA: Parameter descriptions and code references (shared across all)
- EN1992_X_X: Dict with "EU" as base values, other country codes as overrides only

The lookup logic in __init__.py merges: EU base + country overrides + metadata
"""

from math import cos, radians, sqrt, tan


def _max_link_spacing_ec2(
    *,
    effective_depth: float,
    section_depth: float,
    f_ck: float,
    V_Ed: float,
    V_Rd_max: float,
    V_Rd_c: float | None,
    link_angle_degrees: float,
) -> float:
    """Base EC2 §9.2.2(6): s_l,max = 0.75 d (1 + cot α)."""
    if effective_depth <= 0:
        raise ValueError(f"effective_depth must be > 0, got {effective_depth}")

    if abs(link_angle_degrees - 90.0) < 1e-9:
        cot_alpha = 0.0
    else:
        cot_alpha = 1.0 / tan(radians(link_angle_degrees))

    return 0.75 * effective_depth * (1.0 + cot_alpha)


def _max_link_spacing_eu_de(
    *,
    effective_depth: float,
    section_depth: float,
    f_ck: float,
    V_Ed: float,
    V_Rd_max: float,
    V_Rd_c: float | None,
    link_angle_degrees: float,
) -> float:
    """
    German NA spacing limits (piecewise in V_Ed / V_Rd,max).

    Bands:
    - V_Ed <= 0.3*V_Rd,max
    - 0.3*V_Rd,max < V_Ed <= 0.6*V_Rd,max
    - V_Ed > 0.6*V_Rd,max

    Note b:
    - If h < 200 mm and V_Ed <= V_Rd,c, spacing need not be < 150 mm.
    """
    h = float(section_depth)
    if h <= 0:
        raise ValueError(f"section_depth must be > 0, got {section_depth}")

    ved = abs(float(V_Ed))
    vrd_max = max(float(V_Rd_max), 1e-12)
    ratio = ved / vrd_max

    f_ck_lim = 300.0 if float(f_ck) <= 50.0 else 200.0

    if ratio <= 0.3:
        s_l_max = min(0.7 * h, f_ck_lim)
        if h < 200.0 and V_Rd_c is not None and ved <= abs(float(V_Rd_c)):
            s_l_max = max(s_l_max, 150.0)
        return s_l_max

    if ratio <= 0.6:
        return min(0.5 * h, f_ck_lim)

    return min(0.25 * h, 200.0)


def _max_leg_spacing_ec2(
    *,
    effective_depth: float,
    section_depth: float,
    f_ck: float,
    V_Ed: float,
    V_Rd_max: float,
    V_Rd_c: float | None,
    link_angle_degrees: float,
) -> float:
    """Base EC2 §9.2.2(8): s_t,max = min(600 mm, 0.75 d)."""
    if effective_depth <= 0:
        raise ValueError(f"effective_depth must be > 0, got {effective_depth}")
    return min(600.0, 0.75 * float(effective_depth))


def _max_leg_spacing_eu_de(
    *,
    effective_depth: float,
    section_depth: float,
    f_ck: float,
    V_Ed: float,
    V_Rd_max: float,
    V_Rd_c: float | None,
    link_angle_degrees: float,
) -> float:
    """
    German NA maximum leg spacing limits (piecewise in V_Ed / V_Rd,max).

    Bands:
    - V_Ed <= 0.3*V_Rd,max
    - 0.3*V_Rd,max < V_Ed <= V_Rd,max
    """
    h = float(section_depth)
    if h <= 0:
        raise ValueError(f"section_depth must be > 0, got {section_depth}")

    ved = abs(float(V_Ed))
    vrd_max = max(float(V_Rd_max), 1e-12)
    ratio = ved / vrd_max

    if ratio <= 0.3:
        return min(h, 800.0) if float(f_ck) <= 50.0 else min(h, 600.0)

    # Include ratio > 1.0 in the same branch (check fails anyway on utilization).
    return min(h, 600.0) if float(f_ck) <= 50.0 else min(h, 400.0)


def _h_c_ef_multiplier_de(h: float, d: float) -> float:
    """German NA bilinear h_c,ef multiplier (NCI Re 7.3.2(3), Figure 7.1d).

    Returns the multiplier on d_1 = (h - d) for h_c,ef.
    Bilinear graph: (h/d_1=0, mult=0) -> (5, 2.5) -> (30, 5.0).

    For h/d_1 <= 5 (thin sections), gives h/2 (same as standard cap).
    For h/d_1 > 5 (thick sections), multiplier increases linearly to 5.0.
    """
    d_1 = h - d
    if d_1 <= 0:
        return 2.5
    ratio = h / d_1
    if ratio <= 5.0:
        return 0.5 * ratio
    elif ratio <= 30.0:
        return 2.5 + 0.1 * (ratio - 5.0)
    else:
        return 5.0


# =============================================================================
# METADATA: Descriptions and code references for all NDP parameters
# =============================================================================

_NDP_METADATA = {
    "gamma_c": {
        "description": "Concrete partial safety factor for ULS Persistent-Transient",
        "ref": "2.4.2.4(1)",
    },
    "gamma_s": {
        "description": "Reinforcing steel partial safety factor for ULS Persistent-Transient",
        "ref": "2.4.2.4(1)",
    },
    "gamma_c_accidental": {
        "description": "Concrete partial safety factor for ULS Accidental",
        "ref": "2.4.2.4(1)",
    },
    "gamma_s_accidental": {
        "description": "Reinforcing steel partial safety factor for ULS Accidental",
        "ref": "2.4.2.4(1)",
    },
    "k_f": {
        "description": "Foundation factor for cast-in place piles without permanent casing",
        "ref": "2.4.2.5(2)",
    },
    "f_ck_min": {
        "description": "Minimum cylinder strength supported (inclusive)",
        "ref": "3.1.2(2)",
    },
    "f_ck_cube_min": {
        "description": "Minimum cube strength supported (inclusive)",
        "ref": "3.1.2(2)",
    },
    "f_ck_max": {
        "description": "Maximum cylinder strength supported (inclusive)",
        "ref": "3.1.2(2)",
    },
    "f_ck_cube_max": {
        "description": "Maximum cube strength supported (inclusive)",
        "ref": "3.1.2(2)",
    },
    "alpha_cc": {
        "description": "ULS concrete compressive strength coefficient for long-term effects",
        "ref": "3.1.6(1)",
    },
    "alpha_ct": {
        "description": "ULS concrete tensile strength coefficient for long-term effects",
        "ref": "3.1.6(2)",
    },
    "alpha_cc_shear": {
        "description": "ULS concrete compressive strength coefficient for shear (§3.1.6(2))",
        "ref": "3.1.6(2)",
    },
    "k_strain": {
        "description": "Design strain ratio limit between yield strain and ultimate strain in reinforcement",
        "ref": "3.2.7(2)",
    },
    "c_rd_c_coefficient": {
        "description": "Concrete (unreinforced) shear capacity coefficient",
        "ref": "6.2.2(1)",
    },
    "k_1_shear": {
        "description": "Concrete (unreinforced) shear capacity axial stress coefficient",
        "ref": "6.2.2(1)",
    },
    "v_min_coefficient": {
        "description": "Concrete (unreinforced) minimum shear capacity coefficient.",
        "ref": "6.2.2(1), Eq. (6.3N)",
    },
    "cot_theta_lower_lim": {
        "description": "Lower limit of cot(theta)",
        "ref": "6.2.3(2)",
    },
    "cot_theta_upper_lim": {
        "description": "Upper limit of cot(theta)",
        "ref": "6.2.3(2)",
    },
    "nu_shear": {
        "description": "Strength reduction factor for concrete cracked in shear",
        "ref": "6.2.2(6)",
    },
    "nu_1": {
        "description": "Strength reduction factor for V_Rd,max (Note 1)",
        "ref": "6.2.3(3) Note 1",
    },
    "nu_1_note_2": {
        "description": "Increased nu_1 for V_Rd,max when sigma_s < 0.8*f_yk (Note 2)",
        "ref": "6.2.3(3) Note 2",
    },
    "alpha_cw": {
        "description": "Coefficient for stress state in compression chord",
        "ref": "6.2.3(3)",
    },
    "nu_torsion": {
        "description": "Strength reduction factor for torsion",
        "ref": "6.3.2(4)",
    },
    "z_cap": {
        "description": "Upper limit on lever arm z for shear: z_cap = max(d - 2·d_2, d - d_2 - 30)",
        "ref": "NA 6.2.3(1)",
    },
    "k_1_stress": {
        "description": "SLS Characteristic concrete stress limit factor",
        "ref": "7.2(2)",
    },
    "k_2_stress": {
        "description": "SLS Quasi-Permanent concrete stress limit factor",
        "ref": "7.2(3)",
    },
    "k_3_stress": {
        "description": "SLS Characteristic reinforcement stress limit factor",
        "ref": "7.2(5)",
    },
    "k_4_stress": {
        "description": "SLS Characteristic imposed deformation stress limit factor",
        "ref": "7.2(5)",
    },
    "f_ct_eff_min": {
        "description": "The minimum value to take for f_ct,eff for minimum crack reinforcement",
        "ref": "7.3.2(2)",
    },
    "k_1_crack": {
        "description": "Crack calculation rebar bond factor",
        "ref": "7.3.4(3)",
    },
    "k_3_crack": {
        "description": "Crack calculation cover factor",
        "ref": "7.3.4(3)",
    },
    "k_4_crack": {
        "description": "Crack calculation diameter factor",
        "ref": "7.3.4(3)",
    },
    "s_r_max_lim": {
        "description": "Upper limit for max crack spacing",
        "ref": "7.3.4(3)",
    },
    "h_c_ef_multiplier": {
        "description": "Multiplier function for h_c,ef = f(h, d) * (h - d). "
                       "None uses default 2.5.",
        "ref": "7.3.2(3)",
    },
    "h_c_ef_relaxed_na_factor": {
        "description": "When bars are not within (h-x)/3, use (h-x_I) * factor "
                       "with x in State I. None uses default relaxation (drop NA term).",
        "ref": "7.3.2(3)",
    },
    "rho_w_min": {
        "description": "Minimum shear reinforcement ratio",
        "ref": "9.2.2(5) (9.5N)",
    },
    "max_link_spacing": {
        "description": "Maximum allowable longitudinal spacing of shear links/stirrups",
        "ref": "9.2.2(6)",
    },
    "max_leg_spacing": {
        "description": "Maximum allowable transverse spacing between shear reinforcement legs",
        "ref": "9.2.2(8)",
    },
    "as_min_flexural_ratio": {
        "description": "Minimum flexural longitudinal reinforcement ratio for ULS/SLS detailing",
        "ref": "9.2.1.1(1) (9.1N)",
    },
    "as_max_flexural_ratio": {
        "description": "Maximum flexural longitudinal reinforcement ratio",
        "ref": "9.2.1.1(3)",
    },
}


# =============================================================================
# EN 1992-1-1:2004 (Buildings)
# =============================================================================

EN1992_1_1_2004 = {
    # -------------------------------------------------------------------------
    # EU (Base Eurocode) - All parameters must be defined here
    # -------------------------------------------------------------------------
    "EU": {
        "gamma_c": 1.5,
        "gamma_s": 1.15,
        "gamma_c_accidental": 1.2,
        "gamma_s_accidental": 1.0,
        "k_f": 1.1,
        "f_ck_min": 12,
        "f_ck_cube_min": 15,
        "f_ck_max": 90,
        "f_ck_cube_max": 105,
        "alpha_cc": 1.0,
        "alpha_ct": 1.0,
        "alpha_cc_shear": 1.0,
        "k_strain": 0.9,
        "c_rd_c_coefficient": 0.18,
        "k_1_shear": 0.15,
        "v_min_coefficient": 0.035,
        "cot_theta_lower_lim": 1.0,
        "cot_theta_upper_lim": 2.5,
        "nu_shear": lambda f_ck: 0.6 * (1 - f_ck / 250),
        "nu_1": lambda f_ck, angle_deg: 0.6 * (1 - f_ck / 250),
        "nu_1_note_2": lambda f_ck, angle_deg: (
            0.6 if f_ck <= 60
            else max(0.9 - f_ck / 200, 0.5)
        ),
        "alpha_cw": lambda f_cd, sigma_cp: (
            1.0 if sigma_cp == 0
            else (1.0 + sigma_cp / f_cd) if sigma_cp <= 0.25 * f_cd
            else 1.25 if sigma_cp <= 0.5 * f_cd
            else 2.5 * (1 - sigma_cp / f_cd)
        ),
        "nu_torsion": lambda f_ck: 0.6 * (1 - f_ck / 250),
        "z_cap": None,  # No additional z cap in base EC2
        "k_1_stress": 0.6,
        "k_2_stress": 0.45,
        "k_3_stress": 0.8,
        "k_4_stress": 1.0,
        "f_ct_eff_min": None,  # No lower bound in base EC2, depends on f_ck
        "k_1_crack": lambda is_high_bond_bar, k_2: 0.8 if is_high_bond_bar else 1.6,
        "k_3_crack": 3.4,
        "k_4_crack": 0.425,
        "s_r_max_lim": None,  # No additional limit in base EC2
        "h_c_ef_multiplier": None,  # Standard 2.5 factor
        "h_c_ef_relaxed_na_factor": None,  # Drop (h-x)/3 term on relaxation
        "rho_w_min": lambda f_ck, f_yk, f_ctm: 0.08 * sqrt(f_ck) / f_yk,
        "max_link_spacing": _max_link_spacing_ec2,
        "max_leg_spacing": _max_leg_spacing_ec2,
        "as_min_flexural_ratio": lambda f_ctm, f_yk: max(0.0013, 0.26 * (f_ctm / f_yk)),
        "as_max_flexural_ratio": 0.04,
    },

    # -------------------------------------------------------------------------
    # EU_UK (UK National Annex) - Only parameters that differ from EU
    # -------------------------------------------------------------------------
    "EU_UK": {
        "alpha_cc": 0.85,
        "alpha_cc_shear": 1.0,
        "nu_1": lambda f_ck, angle_deg: (
            (0.6 * (1 - f_ck / 250)) * (1 - 0.5 * cos(radians(angle_deg)))
        ),
        "nu_1_note_2": lambda f_ck, angle_deg: (
            0.54 * (1 - 0.5 * cos(radians(angle_deg))) if f_ck <= 60
            else max(
                (0.84 - f_ck / 200) * (1 - 0.5 * cos(radians(angle_deg))),
                0.5
            )
        ),
    },

    # -------------------------------------------------------------------------
    # EU_DE (German National Annex) - Only parameters that differ from EU
    # -------------------------------------------------------------------------
    "EU_DE": {
        "gamma_c_accidental": 1.3,
        "f_ck_max": 100,
        "f_ck_cube_max": 115,
        "alpha_cc": 0.85,
        "alpha_ct": 0.85,
        "alpha_cc_shear": 0.85,
        "c_rd_c_coefficient": 0.15,
        "k_1_shear": 0.12,
        "v_min_coefficient": lambda d, gamma_c: (
            0.0525 if d <= 600
            else 0.0375 if d >= 800
            else 0.0525 + (d - 600) * (0.0375 - 0.0525) / (800 - 600)
        ) / gamma_c,
        "cot_theta_upper_lim": lambda f_ck, f_cd, sigma_cp, b_w, z, V_Ed: (
            min(
                3.0,
                (1.2 + 1.4 * sigma_cp / f_cd)
                / (1 - (0.24 * f_ck**(1/3) * (1 - 1.2 * sigma_cp / f_cd) * b_w * z) / V_Ed)
            )
        ),
        "nu_shear": 0.675,
        "nu_1": lambda f_ck, angle_deg: 0.75 * max(1.1 - f_ck / 500, 1.0),
        "nu_1_note_2": lambda f_ck, angle_deg: 0.75 * max(1.1 - f_ck / 500, 1.0),  # Note 2 not allowed
        "alpha_cw": 1.0,
        "nu_torsion": 0.525,
        "z_cap": lambda d, d_2: max(d - 2 * d_2, d - d_2 - 30),  # German NA lever arm cap
        "f_ct_eff_min": 3.0,
        "k_1_crack": lambda is_high_bond_bar, k_2: 1 / k_2,
        "k_3_crack": 0.0,
        "k_4_crack": 1.0 / 3.6,
        "s_r_max_lim": lambda sigma_s, diameter, f_ct_eff: (sigma_s * diameter) / (3.6 * f_ct_eff),
        "h_c_ef_multiplier": _h_c_ef_multiplier_de,
        "h_c_ef_relaxed_na_factor": 0.5,  # (h - x_I) / 2
        "rho_w_min": lambda f_ck, f_yk, f_ctm: 0.16 * f_ctm / f_yk,  # (9.5aDE)
        "max_link_spacing": _max_link_spacing_eu_de,
        "max_leg_spacing": _max_leg_spacing_eu_de,
        "as_min_flexural_ratio": lambda f_ctm, f_yk: 0.0,
    },
}


# =============================================================================
# EN 1992-2:2005 (Bridges)
# =============================================================================

EN1992_2_2005 = {
    # -------------------------------------------------------------------------
    # EU (Base Eurocode) - All parameters must be defined here
    # -------------------------------------------------------------------------
    "EU": {
        "gamma_c": 1.5,
        "gamma_s": 1.15,
        "gamma_c_accidental": 1.2,
        "gamma_s_accidental": 1.0,
        "k_f": 1.1,
        "f_ck_min": 30,
        "f_ck_cube_min": 37,
        "f_ck_max": 70,
        "f_ck_cube_max": 85,
        "alpha_cc": 1.0,
        "alpha_ct": 1.0,
        "alpha_cc_shear": 1.0,
        "k_strain": 0.9,
        "c_rd_c_coefficient": 0.18,
        "k_1_shear": 0.15,
        "v_min_coefficient": 0.035,
        "cot_theta_lower_lim": 1.0,
        "cot_theta_upper_lim": 2.5,
        "nu_shear": lambda f_ck: 0.6 * (1 - f_ck / 250),
        "nu_1": lambda f_ck, angle_deg: 0.6 * (1 - f_ck / 250),
        "nu_1_note_2": lambda f_ck, angle_deg: (
            0.6 if f_ck <= 60
            else max(0.9 - f_ck / 200, 0.5)
        ),
        "alpha_cw": lambda f_cd, sigma_cp: (
            1.0 if sigma_cp == 0
            else (1.0 + sigma_cp / f_cd) if sigma_cp <= 0.25 * f_cd
            else 1.25 if sigma_cp <= 0.5 * f_cd
            else 2.5 * (1 - sigma_cp / f_cd)
        ),
        "nu_torsion": lambda f_ck: 0.6 * (1 - f_ck / 250),
        "z_cap": None,  # No additional z cap in base EC2
        "k_1_stress": 0.6,
        "k_2_stress": 0.45,
        "k_3_stress": 0.8,
        "k_4_stress": 1.0,
        "f_ct_eff_min": 2.9, # (7.1)
        "k_1_crack": lambda is_high_bond_bar, k_2: 0.8 if is_high_bond_bar else 1.6,
        "k_3_crack": 3.4,
        "k_4_crack": 0.425,
        "s_r_max_lim": None,  # No additional limit in base EC2
        "h_c_ef_multiplier": None,  # Standard 2.5 factor
        "h_c_ef_relaxed_na_factor": None,  # Drop (h-x)/3 term on relaxation
        "max_link_spacing": _max_link_spacing_ec2,
        "max_leg_spacing": _max_leg_spacing_ec2,
        "as_min_flexural_ratio": lambda f_ctm, f_yk: max(0.0013, 0.26 * (f_ctm / f_yk)),
        "as_max_flexural_ratio": 0.04,
    },

    # -------------------------------------------------------------------------
    # EU_UK (UK National Annex) - Only parameters that differ from EU
    # -------------------------------------------------------------------------
    "EU_UK": {
        "f_ck_min": 25,
        "f_ck_cube_min": 30,
        "alpha_cc": 0.85,
        "alpha_cc_shear": 1.0,
        "nu_1": lambda f_ck, angle_deg: (
            (0.6 * (1 - f_ck / 250)) * (1 - 0.5 * cos(radians(angle_deg)))
        ),
        "nu_1_note_2": lambda f_ck, angle_deg: (
            0.54 * (1 - 0.5 * cos(radians(angle_deg))) if f_ck <= 60
            else max(
                (0.84 - f_ck / 200) * (1 - 0.5 * cos(radians(angle_deg))),
                0.5
            )
        ),
    },

    # -------------------------------------------------------------------------
    # EU_DE (German National Annex) - Only parameters that differ from EU
    # -------------------------------------------------------------------------
    "EU_DE": {
        "gamma_c_accidental": 1.3,
        "f_ck_min": 12,
        "f_ck_cube_min": 15,
        "f_ck_max": 50,
        "f_ck_cube_max": 60,
        "alpha_cc": 0.85,
        "alpha_ct": 0.85,
        "alpha_cc_shear": 0.85,
        "c_rd_c_coefficient": 0.15,
        "k_1_shear": 0.12,
        "v_min_coefficient": lambda d, gamma_c: (
            0.0525 if d <= 600
            else 0.0375 if d >= 800
            else 0.0525 + (d - 600) * (0.0375 - 0.0525) / (800 - 600)
        ) / gamma_c,
        "cot_theta_upper_lim": lambda f_ck, f_cd, sigma_cp, b_w, z, V_Ed: (
            min(
                3.0,
                (1.2 + 1.4 * sigma_cp / f_cd)
                / (1 - (0.24 * f_ck**(1/3) * (1 - 1.2 * sigma_cp / f_cd) * b_w * z) / V_Ed)
            )
        ),
        "nu_shear": 0.75,
        "nu_1": lambda f_ck, angle_deg: 0.75,
        "nu_1_note_2": lambda f_ck, angle_deg: 0.75 * max(1.1 - f_ck / 500, 1.0),  # Note 2 not allowed
        "alpha_cw": 1.0,
        "nu_torsion": 0.525,
        "z_cap": lambda d, d_2: max(d - 2 * d_2, d - d_2 - 30),  # German NA lever arm cap
        "f_ct_eff_min": 3.0,
        "k_1_crack": lambda is_high_bond_bar, k_2: 1 / k_2,
        "k_3_crack": 0.0,
        "k_4_crack": 1.0 / 3.6,
        "s_r_max_lim": lambda sigma_s, diameter, f_ct_eff: (sigma_s * diameter) / (3.6 * f_ct_eff),
        "h_c_ef_multiplier": _h_c_ef_multiplier_de,
        "h_c_ef_relaxed_na_factor": 0.5,  # (h - x_I) / 2
        "max_link_spacing": _max_link_spacing_eu_de,
        "max_leg_spacing": _max_leg_spacing_eu_de,
        "as_min_flexural_ratio": lambda f_ctm, f_yk: 0.0,
    },
}
