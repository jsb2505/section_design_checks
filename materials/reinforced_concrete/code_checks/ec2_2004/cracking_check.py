"""
Cracking (flexure) check using M-N interaction diagrams and serviceability loads and
stress-strain relationships.

This is a FIRST PRINCIPLES check based on strain compatibility and force equilibrium.
Uses the fibre-based M-N interaction diagram infrastructure.
"""

from math import sqrt
from typing import Optional
from scipy import interpolate

from pydantic import Field, PrivateAttr

from materials.reinforced_concrete.code_checks.base_check import (
    BaseCodeCheck,
    CheckResult,
)
from materials.reinforced_concrete.constitutive import ConcreteModelType, SteelModelType
from materials.reinforced_concrete.geometry import RCSection
from materials.reinforced_concrete.materials import ConcreteMaterial
from materials.reinforced_concrete.analysis import create_interaction_diagram
from materials.reinforced_concrete.analysis.interaction_diagram import MNInteractionDiagram


# TO DO:
# Finish find_adjusted_ratio_of_bond_strengths method
# rename methods
# remove tension and compression rebar these should be calculated from the interaction diagram
# will need to update the method calls so the new rebar set-up works
# i.e. area of steel and effective depth

class CrackingCheck(BaseCodeCheck):
    '''
    EC2 2004 cracking check for reinforced concrete sections in flexure.
    Ref: EC2 §7.3
    '''
    section: RCSection = Field(
        ...,
        description="RC section with reinforcement",
    )

    concrete: ConcreteMaterial = Field(
        ...,
        description="Concrete material properties",
    )

    concrete_model_type: ConcreteModelType = Field(
        default=ConcreteModelType.PARABOLA_RECTANGLE,
        description="EC2 concrete stress-strain model (Fig 3.3, 3.4, 3.2)",
    )

    steel_model_type: SteelModelType = Field(
        default=SteelModelType.INCLINED,
        description="Steel post-yield behaviour (Fig 3.8)",
    )

    n_fibres_width: int = Field(
        default=20,
        description="Number of concrete fibres across width",
        ge=10,
        le=500,
    )

    n_fibres_height: int = Field(
        default=30,
        description="Number of concrete fibres across height",
        ge=10,
        le=500,
    )

    
    # ===========================
    # Internal state (private)
    # ===========================

    _diagram: MNInteractionDiagram = PrivateAttr()
    _diagram_no_comp_steel: Optional[MNInteractionDiagram] = PrivateAttr(default=None)

    def model_post_init(self, __context):
        """
        Create M-N interaction diagram on initialization for reuse across multiple checks.

        This significantly improves performance when checking multiple load cases against
        the same section, as the diagram (mesh + material models) only needs to be created once.
        """
        super().model_post_init(__context)

        # Create and cache the diagram for reuse (with compression steel)
        self._diagram = create_interaction_diagram(
            section=self.section,
            concrete=self.concrete,
            concrete_model_type=self.concrete_model_type,
            steel_model_type=self.steel_model_type,
            n_fibres_width=self.n_fibres_width,
            n_fibres_height=self.n_fibres_height,
            use_characteristic=True,
            ignore_compression_steel=False,
        )
        # Diagram without compression steel is created lazily on first use
        self._diagram_no_comp_steel = None

    def _get_diagram(self, ignore_compression_steel: bool = False) -> MNInteractionDiagram:
        """Get the appropriate cached diagram based on ignore_compression_steel flag."""
        if not ignore_compression_steel:
            return self._diagram

        # Lazily create the diagram without compression steel
        if self._diagram_no_comp_steel is None:
            self._diagram_no_comp_steel = create_interaction_diagram(
                section=self.section,
                concrete=self.concrete,
                concrete_model_type=self.concrete_model_type,
                steel_model_type=self.steel_model_type,
                n_fibres_width=self.n_fibres_width,
                n_fibres_height=self.n_fibres_height,
                use_characteristic=True,
                ignore_compression_steel=True,
            )
        return self._diagram_no_comp_steel


    def find_rho_p_eff(self,
                       area_of_steel_tension: float,
                       h_c_ef: float,
                       adjusted_ratio_of_bond_strengths: Optional[float] = None,
                       area_of_prestressing_steel: float = 0) -> float:
        '''Ratio of area of tension steel reinforcement to effective
        area of concrete surrounding tension reinforcement.
        
        Ref: EC2 §7.3.4(2) (7.10)
        '''
        xi_1 = adjusted_ratio_of_bond_strengths  # ξ_1
        if xi_1 is None:
            xi_1 = 0  # ξ_1
        # Effective area of concrete surrounding reinforcement
        a_c_eff = h_c_ef * self.breadth
        rho_p_eff = (area_of_steel_tension + xi_1 * area_of_prestressing_steel) / a_c_eff
        return rho_p_eff

    @staticmethod
    def find_adjusted_ratio_of_bond_strengths(
            largest_diameter_of_rebar: int,
            equivalent_diameter_of_tendon: float,
            ratio_of_bond_strength_between_bonded_tendons_and_ribbed_steel: float
            ) -> float:
        '''ξ_1
        
        Ref: EC2 §7.3.2(3) (7.5) & §6.8.2 & Table 6.2
        '''
        xi = ratio_of_bond_strength_between_bonded_tendons_and_ribbed_steel  # ξ, §6.8.2 & Table 6.2
        xi_1 = sqrt(xi * largest_diameter_of_rebar / equivalent_diameter_of_tendon)
        return xi_1

    def find_h_c_ef(self) -> float:
        '''Effective height of concrete surrounding tension reinforcement, in [mm].

        Ref: EC2 §7.3.2(3) & Fig 7.1
        '''
        E_cm = self.concrete.get_elastic_modulus()
        h = self.height
        d_mean = self.mean_effective_depth
        x_u = self.find_neutral_axis_depth_uncracked(E_cm)
        h_c_ef = min(
            2.5 * (h - d_mean),
            (h - x_u) / 3,
            h / 2
        )
        return h_c_ef

    def find_area_of_steel_minimum(self,
                                   f_ctm: float,
                                   k_c: float = 1,
                                   steel_stress: float = 500) -> float:
        '''Minimum area of reinforcement to control cracking, in [mm^2].

        f_ctm may be given at time, t, i.e. f_ctm(t), if cracking expected earlier than 28days.
        steel_stress should be positive in [MPa]
        Ref: EC2 §7.3.2(2) (7.1)
        '''
        b = self.breadth
        h = self.height
        min_dimension = min(h, b)
        # k = coefficient for non-uniform, self-equilibrium stress
        k_max = 1
        k_min = 0.65
        if min_dimension <= 300:
            k = k_max
        elif min_dimension >= 800:
            k = k_min
        else:
            k_interpolator = interpolate.interp1d([300, 800], [k_max, k_min])
            k = k_interpolator(min_dimension)
        f_ctm = self.concrete.f_ctm
        h_c_ef = self.find_h_c_ef()
        a_ct = h_c_ef * b
        a_s_min_crack = k_c * k * f_ctm * a_ct / abs(steel_stress)
        return a_s_min_crack

    def find_k_c(
            self,
            axial_force_sls: float = 0,
            is_in_bending: bool = True,
            use_transformed_area: bool = True
        ) -> float:
        '''Factor for minimum crack reinforcement taking into 
        account stress distribution within the section.

        axial_force may be tensile (negative) or compressive (positive), must be in [kN].
        The axial_force should be calculated for the relevant SLS.
        f_ctm may be given at time, t, i.e. f_ctm(t)
        Ref: EC2 §7.3.2(2) (7.1)
        '''
        if is_in_bending:
            h = self.height
            if h < 1000:
                h_star = h
            else:
                h_star = 1000

            if axial_force_sls >= 0:
                k_1 = 1.5 # EC2 clause doesn't say what to use if N_Ed = 0. This is conservative.
            else:
                k_1 = (2 * h_star) / (3 * h)
            
            f_ctm = self.concrete.f_ctm

            if use_transformed_area:
                A_eff = self.section.get_transformed_area(self.concrete.get_elastic_modulus())
            else:
                A_eff = self.section.get_area()
            
            concrete_stress = axial_force_sls * 10**3 / A_eff  # convert kN to N
            k_c = min(1, 0.4 * (1 -  concrete_stress / (k_1 * (h / h_star) * f_ctm)))
        else:
            k_c = 1  # pure tension
        return k_c

    # TODO NEED TO FINISH
    def find_maximum_crack_spacing(self,
                                   cover_to_flexural_tension_rebar: float,
                                   elastic_modulus_of_concrete: float,
                                   rho_p_eff: float,
                                   is_in_bending: bool = True,
                                   is_high_bond_bar: bool = True) -> float:
        '''Maximum crack spacing, in [mm].

        Ref: EC2 §7.3.4(3) (7.11)
        '''
        # See PD 6687-1:2020 has more guidance on crack check
        # cycle through tension rebar only...
        for rebar_group in self.section.rebar_groups:
            bar_diameter_tens = rebar_group.rebar.diameter
            bar_spacing_tens = self.section.find_bar_spacing(rebar_group)

        if is_in_bending:
            x_c = self.section.find_neutral_axis_depth_cracked(elastic_modulus_of_concrete)
        else:
            x_c = 0

        # Assumed that this applies to tension only rebar bars
        #ø_eq = (n_1 * ø_1**2 + n_2 * ø_2**2) / (n_1 * ø_1 + n_2 * ø_2)

        #! Using the same approach used for equivalent diameter
        # to find an equivalent spacing of multiple bars
        #max_spacing = (n_1**2 * ø_1 + n_2**2 * ø_2) / (n_1 * ø_1 + n_2 * ø_2)

        if mixed_sizes:
            eqv_dia = find_equivalent_diameter()
        else:
            eqv_dia = bar_diameter_tens

        if is_high_bond_bar:
            k_1 = 0.8
        else:  # plain bar
            k_1 = 1.6

        if is_in_bending:
            k_2 = 0.5
        else:  # pure tension
            k_2 = 1

        k_3 = 3.4
        k_4 = 0.425
        c = cover_to_flexural_tension_rebar
        s_r_max = k_3*c + (k_1 * k_2 * k_4 * eqv_dia / rho_p_eff)

        if s_r_max > 5*(c + eqv_dia/2) or not is_high_bond_bar:
            h = self.height
            s_r_max = 1.3 * (h - x_c)
        return s_r_max

    @staticmethod
    def find_crack_width(maximum_crack_spacing: float, difference_in_mean_strains: float) -> float:
        '''Crack width, in [mm].

        Ref: EC2 §7.3.4(1) (7.8)
        '''
        w_k = maximum_crack_spacing * difference_in_mean_strains
        return w_k

    def find_cracking_moment(self, elastic_modulus_of_concrete: float) -> float:
        '''Return the cracking moment in [kNm]. 
        
        The cracking moment is the bending moment at which a section cracks in flexure.
        '''
        f_ctm_fl = self.concrete.find_mean_flexural_tensile_strength(self.height)
        elastic_section_modulus = self.section.find_elastic_section_modulus_uncracked(elastic_modulus_of_concrete)
        m_cr = f_ctm_fl * elastic_section_modulus
        return m_cr / 10**6

    def perform_check(
        self,
        *,
        M_Ed: float,
        N_Ed: float = 0.0,
        warning_threshold: float = 0.95,
        ignore_compression_steel: bool = False,
        **kwargs,
    ) -> CheckResult:
        '''
        Docstring for perform_check
        
        :param self: Description
        :param M_Ed: Description
        :type M_Ed: float
        :param N_Ed: Description
        :type N_Ed: float
        :param warning_threshold: Description
        :type warning_threshold: float
        :param ignore_compression_steel: Description
        :type ignore_compression_steel: bool
        :param kwargs: Description
        :return: Description
        :rtype: CheckResult
        '''
        return self._check_single_case(
            M_Ed=M_Ed,
            N_Ed=N_Ed,
            warning_threshold=warning_threshold,
            ignore_compression_steel=ignore_compression_steel,
        )


    def _check_single_case(
        self,