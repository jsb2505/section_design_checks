"""
Nationally Determined Parameters (NDP) data for Eurocode 2.

Structure:
- _NDP_METADATA: Parameter descriptions and code references (shared across all)
- EN1992_X_X: Dict with "EU" as base values, other country codes as overrides only

The lookup logic in __init__.py merges: EU base + country overrides + metadata
"""

from math import cos, radians, sqrt

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
    "k_1_stress": {
        "description": "SLS Characteristic concrete stress limit factor",
        "ref": "7.2(2)",
    },
    "k_2_stress": {
        "description": "SLS Quasi-Permanent concrete stress limit factor",
        "ref": "7.2(3)",
    },
    "k_3_stress": {
        "description": "SLS reinforcement stress limit factor",
        "ref": "7.2(5)",
    },
    "k_3_crack": {
        "description": "Crack calculation cover factor",
        "ref": "7.3.4(3)",
    },
    "k_4_crack": {
        "description": "Crack calculation diameter factor",
        "ref": "7.3.4(3)",
    },
    "z_cap": {
        "description": "Upper limit on lever arm z for shear: z_cap = max(d - 2·d_2, d - d_2 - 30)",
        "ref": "NA 6.2.3(1)",
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
        "f_ck_max": 90,
        "f_ck_cube_max": 105,
        "alpha_cc": 1.0,
        "alpha_ct": 1.0,
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
        "k_1_stress": 0.6,
        "k_2_stress": 0.45,
        "k_3_stress": 0.8,
        "k_3_crack": 3.4,
        "k_4_crack": 0.425,
        "z_cap": None,  # No additional z cap in base Eurocode
    },

    # -------------------------------------------------------------------------
    # EU_UK (UK National Annex) - Only parameters that differ from EU
    # -------------------------------------------------------------------------
    "EU_UK": {
        "alpha_cc": 0.85,
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
        "k_3_crack": 0.0,
        "k_4_crack": 1.0 / 3.6,
        "z_cap": lambda d, d_2: max(d - 2 * d_2, d - d_2 - 30),  # German NA lever arm cap
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
        "f_ck_max": 90,
        "f_ck_cube_max": 105,
        "alpha_cc": 1.0,
        "alpha_ct": 1.0,
        "k_strain": 0.9,
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
        "k_1_stress": 0.6,
        "k_2_stress": 0.45,
        "k_3_stress": 0.8,
        "k_3_crack": 3.4,
        "k_4_crack": 0.425,
        "z_cap": None,  # No additional z cap in base Eurocode
    },

    # -------------------------------------------------------------------------
    # EU_UK (UK National Annex) - Only parameters that differ from EU
    # -------------------------------------------------------------------------
    "EU_UK": {
        "alpha_cc": 0.85,
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
        "cot_theta_upper_lim": 3.0,  # TODO: intermediate cap formula exists
        "nu_shear": 0.675,
        "nu_1": lambda f_ck, angle_deg: 0.75 * max(1.1 - f_ck / 500, 1.0),
        "nu_1_note_2": lambda f_ck, angle_deg: 0.75 * max(1.1 - f_ck / 500, 1.0),  # Note 2 not allowed
        "alpha_cw": 1.0,
        "nu_torsion": 0.525,
        "k_3_crack": 0.0,
        "k_4_crack": 1.0 / 3.6,
        "z_cap": lambda d, d_2: max(d - 2 * d_2, d - d_2 - 30),  # German NA lever arm cap
    },
}
