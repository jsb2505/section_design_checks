EN1992_1_1_2004 = {
    "EU": {
        "gamma_c": {
            "description": "Concrete partial safety factor for ULS Persistent-Transient",
            "ref": "2.4.2.4(1)",
            "value": 1.5,
        },
        "gamma_s": {
            "description": "Reinforcing steel partial safety factor for ULS Persistent-Transient",
            "ref": "2.4.2.4(1)",
            "value": 1.15,
        },
        "gamma_c_accidental": {
            "description": "Concrete partial safety factor for ULS Accidental",
            "ref": "2.4.2.4(1)",
            "value": 1.2,
        },
        "gamma_s_accidental": {
            "description": "Reinforcing steel partial safety factor for ULS Accidental",
            "ref": "2.4.2.4(1)",
            "value": 1.0,
        },
        "f_ck_max": {
            "description": "Maximum cylinder strength supported (inclusive)",
            "ref": "3.1.2(2)",
            "value": 90,
        },
        "f_ck_cube_max": {
            "description": "Maximum cube strength supported (inclusive)",
            "ref": "3.1.2(2)",
            "value": 105,
        },
        "alpha_cc": {
            "description": "ULS concrete compressive strength coefficient for long-term effects",
            "ref": "3.1.6(1)",
            "value": 0.85,
        },
        "alpha_ct": {
            "description": "ULS concrete compressive strength coefficient for long-term effects",
            "ref": "3.1.6(2)",
            "value": 1.0,
        },
        "k_strain": {
            "description": "Design strain ratio limit between yield strain and ultimate strain in reinforcement",
            "ref": "3.2.7(2)",
            "value": 0.9,
        },
        "cot_theta_lower_lim": {
            "description": "Lower limit of cot(θ)",
            "ref": "6.2.3(2)",
            "value": 1,
        },
        "cot_theta_upper_lim": {
            "description": "Upper limit of cot(θ)",
            "ref": "6.2.3(2)",
            "value": 2.5,
        },
        "nu_shear": {
            "description": "Strength reduction factor for concrete cracked in shear ν = 0.6·(1 - f_ck/250)",
            "ref": "6.2.2(6)",
            "value": lambda f_ck: 0.6 * (1 - f_ck / 250),
        },
        "nu_1": {
            "description": "Strength reduction factor for V_Rd,max. Base EC2: ν₁ = ν (Note 1)",
            "ref": "6.2.3(3) Note 1",
            "value": lambda f_ck, angle_deg: 0.6 * (1 - f_ck / 250),
        },
        "nu_1_note_2": {
            "description": "Increased ν₁ for V_Rd,max when σ_s < 0.8·f_yk (Note 2)",
            "ref": "6.2.3(3) Note 2",
            "value": lambda f_ck, angle_deg: (
                0.6 if f_ck <= 60
                else max(0.9 - f_ck / 200, 0.5)
            ),
        },
        "alpha_cw": {
            "description": "Coefficient for stress state in compression chord (Base EC2: piecewise formula)",
            "ref": "6.2.3(3)",
            "value": lambda f_cd, sigma_cp: (
                1.0 if sigma_cp == 0
                else (1.0 + sigma_cp / f_cd) if sigma_cp <= 0.25 * f_cd
                else 1.25 if sigma_cp <= 0.5 * f_cd
                else 2.5 * (1 - sigma_cp / f_cd)
            ),
        },
        "nu_torsion": {
            "description": "Strength reduction factor for torsion ν = 0.6·(1 - f_ck/250)",
            "ref": "6.3.2(4)",
            "value": lambda f_ck: 0.6 * (1 - f_ck / 250),
        },
        "k_1_stress": {
            "description": "SLS Characteristic concrete stress limit factor",
            "ref": "7.2(2)",
            "value": 0.6,
        },
        "k_2_stress": {
            "description": "SLS Quasi-Permanent concrete stress limit factor",
            "ref": "7.2(3)",
            "value": 0.45,
        },
        "k_3_stress": {
            "description": "SLS reinforcement stress limit factor",
            "ref": "7.2(5)",
            "value": 0.8,
        },
        "k_3_crack": {
            "description": "Crack calculation cover factor",
            "ref": "7.3.4(3)",
            "value": 3.4,
        },
        "k_4_crack": {
            "description": "Crack calculation diameter factor",
            "ref": "7.3.4(3)",
            "value": 0.425,
        },
    },
    "EU_UK": {
        "gamma_c": {
            "description": "Concrete partial safety factor for ULS Persistent-Transient",
            "ref": "2.4.2.4(1)",
            "value": 1.5,
        },
        "gamma_s": {
            "description": "Reinforcing steel partial safety factor for ULS Persistent-Transient",
            "ref": "2.4.2.4(1)",
            "value": 1.15,
        },
        "gamma_c_accidental": {
            "description": "Concrete partial safety factor for ULS Accidental",
            "ref": "2.4.2.4(1)",
            "value": 1.2,
        },
        "gamma_s_accidental": {
            "description": "Reinforcing steel partial safety factor for ULS Accidental",
            "ref": "2.4.2.4(1)",
            "value": 1.0,
        },
        "f_ck_max": {
            "description": "Maximum cylinder strength supported (inclusive)",
            "ref": "3.1.2(2)",
            "value": 90,
        },
        "f_ck_cube_max": {
            "description": "Maximum cube strength supported (inclusive)",
            "ref": "3.1.2(2)",
            "value": 105,
        },
        "alpha_cc": {
            "description": "ULS concrete compressive strength coefficient for long-term effects",
            "ref": "3.1.6(1)",
            "value": 1.0,
        },
        "alpha_ct": {
            "description": "ULS concrete compressive strength coefficient for long-term effects",
            "ref": "3.1.6(2)",
            "value": 1.0,
        },
        "k_strain": {
            "description": "Design strain ratio limit between yield strain and ultimate strain in reinforcement",
            "ref": "3.2.7(2)",
            "value": 0.9,
        },
        "nu_shear": {
            "description": "Strength reduction factor for concrete cracked in shear ν = 0.6·(1 - f_ck/250)",
            "ref": "6.2.2(6)",
            "value": lambda f_ck: 0.6 * (1 - f_ck / 250),
        },
        "nu_torsion": {
            "description": "Strength reduction factor for torsion ν = 0.6·(1 - f_ck/250)",
            "ref": "6.3.2(4)",
            "value": lambda f_ck: 0.6 * (1 - f_ck / 250),
        },
        "nu_1": {
            "description": "Strength reduction factor for V_Rd,max. UK NA: ν₁ = ν·(1 - 0.5·cos(α)) (Note 1)",
            "ref": "6.2.3(3) Note 1",
            "value": lambda f_ck, angle_deg: (0.6 * (1 - f_ck / 250)) * (1 - 0.5 * __import__('math').cos(__import__('math').radians(angle_deg))),
        },
        "nu_1_note_2": {
            "description": "Increased ν₁ for V_Rd,max when σ_s < 0.8·f_yk (Note 2)",
            "ref": "6.2.3(3) Note 2",
            "value": lambda f_ck, angle_deg: (
                0.54 * (1 - 0.5 * __import__('math').cos(__import__('math').radians(angle_deg))) if f_ck <= 60
                else max((0.84 - f_ck / 200) * (1 - 0.5 * __import__('math').cos(__import__('math').radians(angle_deg))), 0.5)
            ),
        },
        "alpha_cw": {
            "description": "Coefficient for stress state in compression chord (UK NA: piecewise formula)",
            "ref": "6.2.3(3)",
            "value": lambda f_cd, sigma_cp: (
                1.0 if sigma_cp == 0
                else (1.0 + sigma_cp / f_cd) if sigma_cp <= 0.25 * f_cd
                else 1.25 if sigma_cp <= 0.5 * f_cd
                else 2.5 * (1 - sigma_cp / f_cd)
            ),
        },
        "cot_theta_lower_lim": {
            "description": "Lower limit of cot(θ)",
            "ref": "6.2.3(2)",
            "value": 1,
        },
        "cot_theta_upper_lim": {
            "description": "Upper limit of cot(θ)",
            "ref": "6.2.3(2)",
            "value": 2.5,
        },
        "k_1_stress": {
            "description": "SLS Characteristic concrete stress limit factor",
            "ref": "7.2(2)",
            "value": 0.6,
        },
        "k_2_stress": {
            "description": "SLS Quasi-Permanent concrete stress limit factor",
            "ref": "7.2(3)",
            "value": 0.45,
        },
        "k_3_stress": {
            "description": "SLS reinforcement stress limit factor",
            "ref": "7.2(5)",
            "value": 0.8,
        },
        "k_3_crack": {
            "description": "Crack calculation cover factor",
            "ref": "7.3.4(3)",
            "value": 3.4,
        },
        "k_4_crack": {
            "description": "Crack calculation diameter factor",
            "ref": "7.3.4(3)",
            "value": 0.425,
        },
    },
    "EU_DE": {
        "gamma_c": {
            "description": "Concrete partial safety factor for ULS Persistent-Transient",
            "ref": "2.4.2.4(1)",
            "value": 1.5,
        },
        "gamma_s": {
            "description": "Reinforcing steel partial safety factor for ULS Persistent-Transient",
            "ref": "2.4.2.4(1)",
            "value": 1.15,
        },
        "gamma_c_accidental": {
            "description": "Concrete partial safety factor for ULS Accidental",
            "ref": "2.4.2.4(1)",
            "value": 1.3,
        },
        "gamma_s_accidental": {
            "description": "Reinforcing steel partial safety factor for ULS Accidental",
            "ref": "2.4.2.4(1)",
            "value": 1.0,
        },
        "f_ck_max": {
            "description": "Maximum cylinder strength supported (inclusive)",
            "ref": "3.1.2(2)",
            "value": 100,
        },
        "f_ck_cube_max": {
            "description": "Maximum cube strength supported (inclusive)",
            "ref": "3.1.2(2)",
            "value": 115,
        },
        "alpha_cc": {
            "description": "ULS concrete compressive strength coefficient for long-term effects",
            "ref": "3.1.6(1)",
            "value": 0.85,
        },
        "alpha_ct": {
            "description": "ULS concrete compressive strength coefficient for long-term effects",
            "ref": "3.1.6(2)",
            "value": 0.85,
        },
        "k_strain": {
            "description": "Design strain ratio limit between yield strain and ultimate strain in reinforcement",
            "ref": "3.2.7(2)",
            "value": 0.9,
        },
        "nu_shear": {
            "description": "Strength reduction factor for concrete cracked in shear (German NA: constant)",
            "ref": "6.2.2(6)",
            "value": 0.675,
        },
        "nu_torsion": {
            "description": "Strength reduction factor for torsion (German NA: constant)",
            "ref": "6.3.2(4)",
            "value": 0.525,
        },
        "nu_1": {
            "description": "Strength reduction factor for V_Rd,max. German NA: ν₁ = 0.75·ν₂ where ν₂ = max(1.1 - f_ck/500, 1.0) (Note 1)",
            "ref": "6.2.3(3) Note 1",
            "value": lambda f_ck, angle_deg: 0.75 * max(1.1 - f_ck / 500, 1.0),
        },
        "nu_1_note_2": {
            "description": "Increased ν₁ for V_Rd,max when σ_s < 0.8·f_yk (Note 2, German NA does not allow this, returns Note 1 value)",
            "ref": "6.2.3(3) Note 2",
            "value": lambda f_ck, angle_deg: 0.75 * max(1.1 - f_ck / 500, 1.0),
        },
        "alpha_cw": {
            "description": "Coefficient for stress state in compression chord (German NA: constant)",
            "ref": "6.2.3(3)",
            "value": 1.0,
        },
        "cot_theta_lower_lim": {
            "description": "Lower limit of cot(θ)",
            "ref": "6.2.3(2)",
            "value": 1,
        },
        "cot_theta_upper_lim": {
            "description": "Upper limit of cot(θ)",
            "ref": "6.2.3(2)", # TODO  there is an intermediate cap: (1.2 + 1.4*sigma_cp/f_cd)/(1-(V_Rd_cc/V_Ed)) where: V_Rd_cc = c*0.48*f_ck^(1/3)*(1-1.2*sigma_cp/f_cd)*b_w*z and c=0.5
            "value": 3.0,
        },
        "k_1_stress": {
            "description": "SLS Characteristic concrete stress limit factor",
            "ref": "7.2(2)",
            "value": 0.6,
        },
        "k_2_stress": {
            "description": "SLS Quasi-Permanent concrete stress limit factor",
            "ref": "7.2(3)",
            "value": 0.45,
        },
        "k_3_stress": {
            "description": "SLS reinforcement stress limit factor",
            "ref": "7.2(5)",
            "value": 0.8,
        },
        "k_3_crack": {
            "description": "Crack calculation cover factor",
            "ref": "7.3.4(3)",
            "value": 0.0,
        },
        "k_4_crack": {
            "description": "Crack calculation diameter factor",
            "ref": "7.3.4(3)",
            "value": 1.0 / 3.6,
        },
    },
}