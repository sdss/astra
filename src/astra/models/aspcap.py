import datetime
import numpy as np
from astra.fields import (
    BitField, PixelArray, BasePixelArrayAccessor, LogLambdaArrayAccessor,
    AutoField,
    FloatField,
    TextField,
    ForeignKeyField,
    IntegerField,
    DateTimeField,
    BooleanField,    
)
from astra.models.pipeline import PipelineOutputModel
from astra.models.ferre import FerreCoarse, FerreStellarParameters, FerreChemicalAbundances
from astra.models.source import Source
from astra.models.spectrum import Spectrum
from astra.glossary import Glossary
from playhouse.hybrid import hybrid_property



class StellarParameterPixelAccessor(BasePixelArrayAccessor):
    
    def __get__(self, instance, instance_type=None):
        if instance is not None:
            try:
                return instance.__pixel_data__[self.name]
            except (AttributeError, KeyError):
                # Load them all.
                instance.__pixel_data__ = {}

                upstream = FerreStellarParameters.get(instance.stellar_parameters_task_pk)
                continuum = upstream.unmask(
                    (upstream.rectified_model_flux/upstream.model_flux)
                /   (upstream.rectified_flux/upstream.ferre_flux)
                )

                instance.__pixel_data__.setdefault("continuum", continuum)
                instance.__pixel_data__.setdefault("model_flux", upstream.unmask(upstream.model_flux))

                return instance.__pixel_data__[self.name]

        return self.field

        

class ChemicalAbundancePixelAccessor(BasePixelArrayAccessor):

    def __get__(self, instance, instance_type=None):
        if instance is not None:
            try:
                return instance.__pixel_data__[self.name]
            except (AttributeError, KeyError):
                instance.__pixel_data__ = {}

                x_h = self.name[len("model_flux_"):]
                upstream = FerreChemicalAbundances.get(getattr(instance, f"{x_h}_task_pk"))

                try:
                    instance.__pixel_data__.setdefault(self.name, upstream.unmask(upstream.rectified_model_flux))
                except:
                    instance.__pixel_data__[self.name] = np.nan * np.ones(8575)
                
            finally:
                return instance.__pixel_data__[self.name]
        
        return self.field


class ChemicalAbundanceModelFluxArray(PixelArray):
    
    def __init__(self, ext=None, column_name=None, transform=None, accessor_class=ChemicalAbundancePixelAccessor, help_text=None, **kwargs):
        super(ChemicalAbundanceModelFluxArray, self).__init__(
            ext=ext,
            column_name=column_name,
            transform=transform,
            accessor_class=accessor_class,
            help_text=help_text,
            **kwargs
        )
        

class ASPCAP(PipelineOutputModel):

    """ APOGEE Stellar Parameter and Chemical Abundances Pipeline (ASPCAP) """
    
    #> Spectral Data
    wavelength = PixelArray(
        accessor_class=LogLambdaArrayAccessor,
        accessor_kwargs=dict(
            crval=4.179,
            cdelt=6e-6,
            naxis=8575,
        ),
    )    
    model_flux = PixelArray(
        accessor_class=StellarParameterPixelAccessor, 
        help_text="Model flux at optimized stellar parameters"
    )
    continuum = PixelArray(
        accessor_class=StellarParameterPixelAccessor,
        help_text="Continuum"
    )

    #> Model Fluxes from Chemical Abundance Fits
    model_flux_al_h = ChemicalAbundanceModelFluxArray(help_text="Model flux at optimized [Al/H]")
    model_flux_c_12_13 = ChemicalAbundanceModelFluxArray(help_text="Model flux at optimized C12/13")
    model_flux_ca_h = ChemicalAbundanceModelFluxArray(help_text="Model flux at optimized [Ca/H]")
    model_flux_ce_h = ChemicalAbundanceModelFluxArray(help_text="Model flux at optimized [Ce/H]")
    model_flux_c_1_h = ChemicalAbundanceModelFluxArray(help_text="Model flux at optimized [C 1/H]")
    model_flux_c_h = ChemicalAbundanceModelFluxArray(help_text="Model flux at optimized [C/H]")
    model_flux_co_h = ChemicalAbundanceModelFluxArray(help_text="Model flux at optimized [Co/H]")
    model_flux_cr_h = ChemicalAbundanceModelFluxArray(help_text="Model flux at optimized [Cr/H]")
    model_flux_cu_h = ChemicalAbundanceModelFluxArray(help_text="Model flux at optimized [Cu/H]")
    model_flux_fe_h = ChemicalAbundanceModelFluxArray(help_text="Model flux at optimized [Fe/H]")
    model_flux_k_h = ChemicalAbundanceModelFluxArray(help_text="Model flux at optimized [K/H]")
    model_flux_mg_h = ChemicalAbundanceModelFluxArray(help_text="Model flux at optimized [Mg/H]")
    model_flux_mn_h = ChemicalAbundanceModelFluxArray(help_text="Model flux at optimized [Mn/H]")
    model_flux_na_h = ChemicalAbundanceModelFluxArray(help_text="Model flux at optimized [Na/H]")
    model_flux_nd_h = ChemicalAbundanceModelFluxArray(help_text="Model flux at optimized [Nd/H]")
    model_flux_ni_h = ChemicalAbundanceModelFluxArray(help_text="Model flux at optimized [Ni/H]")
    model_flux_n_h = ChemicalAbundanceModelFluxArray(help_text="Model flux at optimized [N/H]")
    model_flux_o_h = ChemicalAbundanceModelFluxArray(help_text="Model flux at optimized [O/H]")
    model_flux_p_h = ChemicalAbundanceModelFluxArray(help_text="Model flux at optimized [P/H]")
    model_flux_si_h = ChemicalAbundanceModelFluxArray(help_text="Model flux at optimized [Si/H]")
    model_flux_s_h = ChemicalAbundanceModelFluxArray(help_text="Model flux at optimized [S/H]")
    model_flux_ti_h = ChemicalAbundanceModelFluxArray(help_text="Model flux at optimized [Ti/H]")
    model_flux_ti_2_h = ChemicalAbundanceModelFluxArray(help_text="Model flux at optimized [Ti 2/H]")
    model_flux_v_h = ChemicalAbundanceModelFluxArray(help_text="Model flux at optimized [V/H]")

    #> IRFM Effective Temperatures from V-Ks (Gonzalez Hernandez and Bonifacio 2009)
    irfm_teff = FloatField(null=True, help_text=Glossary.teff)
    irfm_teff_flags = BitField(default=0, help_text="IRFM temperature flags")
    flag_out_of_v_k_bounds = irfm_teff_flags.flag(2**0, "Out of V-Ks bounds")
    flag_out_of_fe_h_bounds = irfm_teff_flags.flag(2**1, "Out of [Fe/H] bounds")
    flag_extrapolated_v_mag = irfm_teff_flags.flag(2**2, "Synthetic V magnitude is extrapolated")
    flag_poor_quality_k_mag = irfm_teff_flags.flag(2**3, "Poor quality Ks magnitude")
    flag_ebv_used_is_upper_limit = irfm_teff_flags.flag(2**4, "E(B-V) used is an upper limit")
    flag_as_dwarf_for_irfm_teff = irfm_teff_flags.flag(2**5, "Flagged as dwarf for IRFM temperature")
    flag_as_giant_for_irfm_teff = irfm_teff_flags.flag(2**6, "Flagged as giant for IRFM temperature")

    #> Stellar Parameters
    teff = FloatField(null=True, help_text=Glossary.teff)
    e_teff = FloatField(null=True, help_text=Glossary.e_teff)
    logg = FloatField(null=True, help_text=Glossary.logg)
    e_logg = FloatField(null=True, help_text=Glossary.e_logg)
    v_micro = FloatField(null=True, help_text=Glossary.v_micro)
    e_v_micro = FloatField(null=True, help_text=Glossary.e_v_micro)
    v_sini = FloatField(null=True, help_text=Glossary.v_sini)
    e_v_sini = FloatField(null=True, help_text=Glossary.e_v_sini)
    m_h_atm = FloatField(null=True, help_text=Glossary.m_h_atm)
    e_m_h_atm = FloatField(null=True, help_text=Glossary.e_m_h_atm)
    alpha_m_atm = FloatField(null=True, help_text=Glossary.alpha_m_atm)
    e_alpha_m_atm = FloatField(null=True, help_text=Glossary.e_alpha_m_atm)
    c_m_atm = FloatField(null=True, help_text=Glossary.c_m_atm)
    e_c_m_atm = FloatField(null=True, help_text=Glossary.e_c_m_atm)
    n_m_atm = FloatField(null=True, help_text=Glossary.n_m_atm)
    e_n_m_atm = FloatField(null=True, help_text=Glossary.e_n_m_atm)

    #> Chemical Abundances
    al_h = FloatField(null=True, help_text=Glossary.al_h)
    e_al_h = FloatField(null=True, help_text=Glossary.e_al_h)
    al_h_flags = BitField(default=0, help_text=Glossary.al_h_flags)
    al_h_rchi2 = FloatField(null=True, help_text=Glossary.al_h_rchi2)
    flag_al_h_censored_high_teff = al_h_flags.flag(2**5, "Censored value because abundances known to be wrong for this Teff")
    flag_al_h_censored_low_teff_vmicro = al_h_flags.flag(2**6, "Censored value because it has low Teff and v_micro")
    flag_al_h_censored_unphysical = al_h_flags.flag(2**7, "Censored value because FERRE returned unphysical value")
    flag_al_h_bad_grid_edge = al_h_flags.flag(2**8, "Grid edge bad")
    flag_al_h_warn_grid_edge = al_h_flags.flag(2**9, "Grid edge warning")
    flag_al_h_warn_teff = al_h_flags.flag(2**10, "These abundances are known to be unreliable for this Teff")
    flag_al_h_warn_m_h = al_h_flags.flag(2**11, "These abundances are known to be unreliable for this [M/H]")
    
    c_12_13 = FloatField(null=True, help_text=Glossary.c_12_13)
    e_c_12_13 = FloatField(null=True, help_text=Glossary.e_c_12_13)
    c_12_13_flags = BitField(default=0, help_text=Glossary.c_12_13_flags)
    c_12_13_rchi2 = FloatField(null=True, help_text=Glossary.c_12_13_rchi2)
    flag_c_12_13_censored_high_teff = c_12_13_flags.flag(2**5, "Censored value because abundances known to be wrong for this Teff")
    flag_c_12_13_censored_low_teff_vmicro = c_12_13_flags.flag(2**6, "Censored value because it has low Teff and v_micro")
    flag_c_12_13_censored_unphysical = c_12_13_flags.flag(2**7, "Censored value because FERRE returned unphysical value")
    flag_c_12_13_bad_grid_edge = c_12_13_flags.flag(2**8, "Grid edge bad")
    flag_c_12_13_warn_grid_edge = c_12_13_flags.flag(2**9, "Grid edge warning")
    flag_c_12_13_warn_teff = c_12_13_flags.flag(2**10, "These abundances are known to be unreliable for this Teff")
    flag_c_12_13_warn_m_h = c_12_13_flags.flag(2**11, "These abundances are known to be unreliable for this [M/H]")


    ca_h = FloatField(null=True, help_text=Glossary.ca_h)
    e_ca_h = FloatField(null=True, help_text=Glossary.e_ca_h)
    ca_h_flags = BitField(default=0, help_text=Glossary.ca_h_flags)
    ca_h_rchi2 = FloatField(null=True, help_text=Glossary.ca_h_rchi2)
    flag_ca_h_censored_high_teff = ca_h_flags.flag(2**5, "Censored value because abundances known to be wrong for this Teff")
    flag_ca_h_censored_low_teff_vmicro = ca_h_flags.flag(2**6, "Censored value because it has low Teff and v_micro")
    flag_ca_h_censored_unphysical = ca_h_flags.flag(2**7, "Censored value because FERRE returned unphysical value")
    flag_ca_h_bad_grid_edge = ca_h_flags.flag(2**8, "Grid edge bad")
    flag_ca_h_warn_grid_edge = ca_h_flags.flag(2**9, "Grid edge warning")
    flag_ca_h_warn_teff = ca_h_flags.flag(2**10, "These abundances are known to be unreliable for this Teff")
    flag_ca_h_warn_m_h = ca_h_flags.flag(2**11, "These abundances are known to be unreliable for this [M/H]")



    ce_h = FloatField(null=True, help_text=Glossary.ce_h)
    e_ce_h = FloatField(null=True, help_text=Glossary.e_ce_h)
    ce_h_flags = BitField(default=0, help_text=Glossary.ce_h_flags)
    ce_h_rchi2 = FloatField(null=True, help_text=Glossary.ce_h_rchi2)
    flag_ce_h_upper_limit_t1 = ce_h_flags.flag(2**0, "At least one line is an upper limit by the 1% threshold in Hayes et al. (2022, ApJ, 262, 34)")
    flag_ce_h_upper_limit_t2 = ce_h_flags.flag(2**1, "At least one line is an upper limit by the 2% threshold in Hayes et al. (2022, ApJ, 262, 34)")
    flag_ce_h_upper_limit_t3 = ce_h_flags.flag(2**2, "At least one line is an upper limit by the 3% threshold in Hayes et al. (2022, ApJ, 262, 34)")
    flag_ce_h_upper_limit_t4 = ce_h_flags.flag(2**3, "At least one line is an upper limit by the 4% threshold in Hayes et al. (2022, ApJ, 262, 34)")
    flag_ce_h_upper_limit_t5 = ce_h_flags.flag(2**4, "At least one line is an upper limit by the 5% threshold in Hayes et al. (2022, ApJ, 262, 34)")
    flag_ce_h_censored_high_teff = ce_h_flags.flag(2**5, "Censored value because abundances known to be wrong for this Teff")
    flag_ce_h_censored_low_teff_vmicro = ce_h_flags.flag(2**6, "Censored value because it has low Teff and v_micro")
    flag_ce_h_censored_unphysical = ce_h_flags.flag(2**7, "Censored value because FERRE returned unphysical value")
    flag_ce_h_bad_grid_edge = ce_h_flags.flag(2**8, "Grid edge bad")
    flag_ce_h_warn_grid_edge = ce_h_flags.flag(2**9, "Grid edge warning")
    flag_ce_h_warn_teff = ce_h_flags.flag(2**10, "These abundances are known to be unreliable for this Teff")
    flag_ce_h_warn_m_h = ce_h_flags.flag(2**11, "These abundances are known to be unreliable for this [M/H]")
    
    c_1_h = FloatField(null=True, help_text=Glossary.c_1_h)
    e_c_1_h = FloatField(null=True, help_text=Glossary.e_c_1_h)
    c_1_h_flags = BitField(default=0, help_text=Glossary.c_1_h_flags)
    c_1_h_rchi2 = FloatField(null=True, help_text=Glossary.c_1_h_rchi2)
    flag_c_1_h_upper_limit_t1 = c_1_h_flags.flag(2**0, "At least one line is an upper limit by the 1% threshold in Hayes et al. (2022, ApJ, 262, 34)")
    flag_c_1_h_upper_limit_t2 = c_1_h_flags.flag(2**1, "At least one line is an upper limit by the 2% threshold in Hayes et al. (2022, ApJ, 262, 34)")
    flag_c_1_h_upper_limit_t3 = c_1_h_flags.flag(2**2, "At least one line is an upper limit by the 3% threshold in Hayes et al. (2022, ApJ, 262, 34)")
    flag_c_1_h_upper_limit_t4 = c_1_h_flags.flag(2**3, "At least one line is an upper limit by the 4% threshold in Hayes et al. (2022, ApJ, 262, 34)")
    flag_c_1_h_upper_limit_t5 = c_1_h_flags.flag(2**4, "At least one line is an upper limit by the 5% threshold in Hayes et al. (2022, ApJ, 262, 34)")
    flag_c_1_h_censored_high_teff = c_1_h_flags.flag(2**5, "Censored value because abundances known to be wrong for this Teff")
    flag_c_1_h_censored_low_teff_vmicro = c_1_h_flags.flag(2**6, "Censored value because it has low Teff and v_micro")
    flag_c_1_h_censored_unphysical = c_1_h_flags.flag(2**7, "Censored value because FERRE returned unphysical value")
    flag_c_1_h_bad_grid_edge = c_1_h_flags.flag(2**8, "Grid edge bad")
    flag_c_1_h_warn_grid_edge = c_1_h_flags.flag(2**9, "Grid edge warning")
    flag_c_1_h_warn_teff = c_1_h_flags.flag(2**10, "These abundances are known to be unreliable for this Teff")
    flag_c_1_h_warn_m_h = c_1_h_flags.flag(2**11, "These abundances are known to be unreliable for this [M/H]")
    
    c_h = FloatField(null=True, help_text=Glossary.c_h)
    e_c_h = FloatField(null=True, help_text=Glossary.e_c_h)
    c_h_flags = BitField(default=0, help_text=Glossary.c_h_flags)
    c_h_rchi2 = FloatField(null=True, help_text=Glossary.c_h_rchi2)
    flag_c_h_upper_limit_t1 = c_h_flags.flag(2**0, "At least one line is an upper limit by the 1% threshold in Hayes et al. (2022, ApJ, 262, 34)")
    flag_c_h_upper_limit_t2 = c_h_flags.flag(2**1, "At least one line is an upper limit by the 2% threshold in Hayes et al. (2022, ApJ, 262, 34)")
    flag_c_h_upper_limit_t3 = c_h_flags.flag(2**2, "At least one line is an upper limit by the 3% threshold in Hayes et al. (2022, ApJ, 262, 34)")
    flag_c_h_upper_limit_t4 = c_h_flags.flag(2**3, "At least one line is an upper limit by the 4% threshold in Hayes et al. (2022, ApJ, 262, 34)")
    flag_c_h_upper_limit_t5 = c_h_flags.flag(2**4, "At least one line is an upper limit by the 5% threshold in Hayes et al. (2022, ApJ, 262, 34)")
    flag_c_h_censored_high_teff = c_h_flags.flag(2**5, "Censored value because abundances known to be wrong for this Teff")
    flag_c_h_censored_low_teff_vmicro = c_h_flags.flag(2**6, "Censored value because it has low Teff and v_micro")
    flag_c_h_censored_unphysical = c_h_flags.flag(2**7, "Censored value because FERRE returned unphysical value")
    flag_c_h_bad_grid_edge = c_h_flags.flag(2**8, "Grid edge bad")
    flag_c_h_warn_grid_edge = c_h_flags.flag(2**9, "Grid edge warning")
    flag_c_h_warn_teff = c_h_flags.flag(2**10, "These abundances are known to be unreliable for this Teff")
    flag_c_h_warn_m_h = c_h_flags.flag(2**11, "These abundances are known to be unreliable for this [M/H]")
    
    co_h = FloatField(null=True, help_text=Glossary.co_h)
    e_co_h = FloatField(null=True, help_text=Glossary.e_co_h)
    co_h_flags = BitField(default=0, help_text=Glossary.co_h_flags)
    co_h_rchi2 = FloatField(null=True, help_text=Glossary.co_h_rchi2)
    flag_co_h_censored_high_teff = co_h_flags.flag(2**5, "Censored value because abundances known to be wrong for this Teff")
    flag_co_h_censored_low_teff_vmicro = co_h_flags.flag(2**6, "Censored value because it has low Teff and v_micro")
    flag_co_h_censored_unphysical = co_h_flags.flag(2**7, "Censored value because FERRE returned unphysical value")
    flag_co_h_bad_grid_edge = co_h_flags.flag(2**8, "Grid edge bad")
    flag_co_h_warn_grid_edge = co_h_flags.flag(2**9, "Grid edge warning")
    flag_co_h_warn_teff = co_h_flags.flag(2**10, "These abundances are known to be unreliable for this Teff")
    flag_co_h_warn_m_h = co_h_flags.flag(2**11, "These abundances are known to be unreliable for this [M/H]")
    
    cr_h = FloatField(null=True, help_text=Glossary.cr_h)
    e_cr_h = FloatField(null=True, help_text=Glossary.e_cr_h)
    cr_h_flags = BitField(default=0, help_text=Glossary.cr_h_flags)
    cr_h_rchi2 = FloatField(null=True, help_text=Glossary.cr_h_rchi2)
    flag_cr_h_censored_high_teff = cr_h_flags.flag(2**5, "Censored value because abundances known to be wrong for this Teff")
    flag_cr_h_censored_low_teff_vmicro = cr_h_flags.flag(2**6, "Censored value because it has low Teff and v_micro")
    flag_cr_h_censored_unphysical = cr_h_flags.flag(2**7, "Censored value because FERRE returned unphysical value")
    flag_cr_h_bad_grid_edge = cr_h_flags.flag(2**8, "Grid edge bad")
    flag_cr_h_warn_grid_edge = cr_h_flags.flag(2**9, "Grid edge warning")
    flag_cr_h_warn_teff = cr_h_flags.flag(2**10, "These abundances are known to be unreliable for this Teff")
    flag_cr_h_warn_m_h = cr_h_flags.flag(2**11, "These abundances are known to be unreliable for this [M/H]")
    
    cu_h = FloatField(null=True, help_text=Glossary.cu_h)
    e_cu_h = FloatField(null=True, help_text=Glossary.e_cu_h)
    cu_h_flags = BitField(default=0, help_text=Glossary.cu_h_flags)
    cu_h_rchi2 = FloatField(null=True, help_text=Glossary.cu_h_rchi2)
    flag_cu_h_upper_limit_t1 = cu_h_flags.flag(2**0, "At least one line is an upper limit by the 1% threshold in Hayes et al. (2022, ApJ, 262, 34)")
    flag_cu_h_upper_limit_t2 = cu_h_flags.flag(2**1, "At least one line is an upper limit by the 2% threshold in Hayes et al. (2022, ApJ, 262, 34)")
    flag_cu_h_upper_limit_t3 = cu_h_flags.flag(2**2, "At least one line is an upper limit by the 3% threshold in Hayes et al. (2022, ApJ, 262, 34)")
    flag_cu_h_upper_limit_t4 = cu_h_flags.flag(2**3, "At least one line is an upper limit by the 4% threshold in Hayes et al. (2022, ApJ, 262, 34)")
    flag_cu_h_upper_limit_t5 = cu_h_flags.flag(2**4, "At least one line is an upper limit by the 5% threshold in Hayes et al. (2022, ApJ, 262, 34)")
    flag_cu_h_censored_high_teff = cu_h_flags.flag(2**5, "Censored value because abundances known to be wrong for this Teff")
    flag_cu_h_censored_low_teff_vmicro = cu_h_flags.flag(2**6, "Censored value because it has low Teff and v_micro")
    flag_cu_h_censored_unphysical = cu_h_flags.flag(2**7, "Censored value because FERRE returned unphysical value")
    flag_cu_h_bad_grid_edge = cu_h_flags.flag(2**8, "Grid edge bad")
    flag_cu_h_warn_grid_edge = cu_h_flags.flag(2**9, "Grid edge warning")
    flag_cu_h_warn_teff = cu_h_flags.flag(2**10, "These abundances are known to be unreliable for this Teff")
    flag_cu_h_warn_m_h = cu_h_flags.flag(2**11, "These abundances are known to be unreliable for this [M/H]")
    
    fe_h = FloatField(null=True, help_text=Glossary.fe_h)
    e_fe_h = FloatField(null=True, help_text=Glossary.e_fe_h)
    fe_h_flags = BitField(default=0, help_text=Glossary.fe_h_flags)
    fe_h_rchi2 = FloatField(null=True, help_text=Glossary.fe_h_rchi2)    
    flag_fe_h_censored_high_teff = fe_h_flags.flag(2**5, "Censored value because abundances known to be wrong for this Teff")
    flag_fe_h_censored_low_teff_vmicro = fe_h_flags.flag(2**6, "Censored value because it has low Teff and v_micro")
    flag_fe_h_censored_unphysical = fe_h_flags.flag(2**7, "Censored value because FERRE returned unphysical value")
    flag_fe_h_bad_grid_edge = fe_h_flags.flag(2**8, "Grid edge bad")
    flag_fe_h_warn_grid_edge = fe_h_flags.flag(2**9, "Grid edge warning")
    flag_fe_h_warn_teff = fe_h_flags.flag(2**10, "These abundances are known to be unreliable for this Teff")
    flag_fe_h_warn_m_h = fe_h_flags.flag(2**11, "These abundances are known to be unreliable for this [M/H]")

    k_h = FloatField(null=True, help_text=Glossary.k_h)
    e_k_h = FloatField(null=True, help_text=Glossary.e_k_h)
    k_h_flags = BitField(default=0, help_text=Glossary.k_h_flags)
    k_h_rchi2 = FloatField(null=True, help_text=Glossary.k_h_rchi2)
    flag_k_h_censored_high_teff = k_h_flags.flag(2**5, "Censored value because abundances known to be wrong for this Teff")
    flag_k_h_censored_low_teff_vmicro = k_h_flags.flag(2**6, "Censored value because it has low Teff and v_micro")
    flag_k_h_censored_unphysical = k_h_flags.flag(2**7, "Censored value because FERRE returned unphysical value")
    flag_k_h_bad_grid_edge = k_h_flags.flag(2**8, "Grid edge bad")
    flag_k_h_warn_grid_edge = k_h_flags.flag(2**9, "Grid edge warning")
    flag_k_h_warn_teff = k_h_flags.flag(2**10, "These abundances are known to be unreliable for this Teff")
    flag_k_h_warn_m_h = k_h_flags.flag(2**11, "These abundances are known to be unreliable for this [M/H]")

    mg_h = FloatField(null=True, help_text=Glossary.mg_h)
    e_mg_h = FloatField(null=True, help_text=Glossary.e_mg_h)
    mg_h_flags = BitField(default=0, help_text=Glossary.mg_h_flags)
    mg_h_rchi2 = FloatField(null=True, help_text=Glossary.mg_h_rchi2)
    flag_mg_h_censored_high_teff = mg_h_flags.flag(2**5, "Censored value because abundances known to be wrong for this Teff")
    flag_mg_h_censored_low_teff_vmicro = mg_h_flags.flag(2**6, "Censored value because it has low Teff and v_micro")
    flag_mg_h_censored_unphysical = mg_h_flags.flag(2**7, "Censored value because FERRE returned unphysical value")
    flag_mg_h_bad_grid_edge = mg_h_flags.flag(2**8, "Grid edge bad")
    flag_mg_h_warn_grid_edge = mg_h_flags.flag(2**9, "Grid edge warning")
    flag_mg_h_warn_teff = mg_h_flags.flag(2**10, "These abundances are known to be unreliable for this Teff")
    flag_mg_h_warn_m_h = mg_h_flags.flag(2**11, "These abundances are known to be unreliable for this [M/H]")

    mn_h = FloatField(null=True, help_text=Glossary.mn_h)
    e_mn_h = FloatField(null=True, help_text=Glossary.e_mn_h)
    mn_h_flags = BitField(default=0, help_text=Glossary.mn_h_flags)
    mn_h_rchi2 = FloatField(null=True, help_text=Glossary.mn_h_rchi2)
    flag_mn_h_censored_high_teff = mn_h_flags.flag(2**5, "Censored value because abundances known to be wrong for this Teff")
    flag_mn_h_censored_low_teff_vmicro = mn_h_flags.flag(2**6, "Censored value because it has low Teff and v_micro")
    flag_mn_h_censored_unphysical = mn_h_flags.flag(2**7, "Censored value because FERRE returned unphysical value")
    flag_mn_h_bad_grid_edge = mn_h_flags.flag(2**8, "Grid edge bad")
    flag_mn_h_warn_grid_edge = mn_h_flags.flag(2**9, "Grid edge warning")
    flag_mn_h_warn_teff = mn_h_flags.flag(2**10, "These abundances are known to be unreliable for this Teff")
    flag_mn_h_warn_m_h = mn_h_flags.flag(2**11, "These abundances are known to be unreliable for this [M/H]")

    na_h = FloatField(null=True, help_text=Glossary.na_h)
    e_na_h = FloatField(null=True, help_text=Glossary.e_na_h)
    na_h_flags = BitField(default=0, help_text=Glossary.na_h_flags)
    na_h_rchi2 = FloatField(null=True, help_text=Glossary.na_h_rchi2)
    flag_na_h_upper_limit_t1 = na_h_flags.flag(2**0, "At least one line is an upper limit by the 1% threshold in Hayes et al. (2022, ApJ, 262, 34)")
    flag_na_h_upper_limit_t2 = na_h_flags.flag(2**1, "At least one line is an upper limit by the 2% threshold in Hayes et al. (2022, ApJ, 262, 34)")
    flag_na_h_upper_limit_t3 = na_h_flags.flag(2**2, "At least one line is an upper limit by the 3% threshold in Hayes et al. (2022, ApJ, 262, 34)")
    flag_na_h_upper_limit_t4 = na_h_flags.flag(2**3, "At least one line is an upper limit by the 4% threshold in Hayes et al. (2022, ApJ, 262, 34)")
    flag_na_h_upper_limit_t5 = na_h_flags.flag(2**4, "At least one line is an upper limit by the 5% threshold in Hayes et al. (2022, ApJ, 262, 34)")
    flag_na_h_censored_high_teff = na_h_flags.flag(2**5, "Censored value because abundances known to be wrong for this Teff")
    flag_na_h_censored_low_teff_vmicro = na_h_flags.flag(2**6, "Censored value because it has low Teff and v_micro")
    flag_na_h_censored_unphysical = na_h_flags.flag(2**7, "Censored value because FERRE returned unphysical value")
    flag_na_h_bad_grid_edge = na_h_flags.flag(2**8, "Grid edge bad")
    flag_na_h_warn_grid_edge = na_h_flags.flag(2**9, "Grid edge warning")
    flag_na_h_warn_teff = na_h_flags.flag(2**10, "These abundances are known to be unreliable for this Teff")
    flag_na_h_warn_m_h = na_h_flags.flag(2**11, "These abundances are known to be unreliable for this [M/H]")

    nd_h = FloatField(null=True, help_text=Glossary.nd_h)
    e_nd_h = FloatField(null=True, help_text=Glossary.e_nd_h)
    nd_h_flags = BitField(default=0, help_text=Glossary.nd_h_flags)
    nd_h_rchi2 = FloatField(null=True, help_text=Glossary.nd_h_rchi2)
    flag_nd_h_upper_limit_t1 = nd_h_flags.flag(2**0, "At least one line is an upper limit by the 1% threshold in Hayes et al. (2022, ApJ, 262, 34)")
    flag_nd_h_upper_limit_t2 = nd_h_flags.flag(2**1, "At least one line is an upper limit by the 2% threshold in Hayes et al. (2022, ApJ, 262, 34)")
    flag_nd_h_upper_limit_t3 = nd_h_flags.flag(2**2, "At least one line is an upper limit by the 3% threshold in Hayes et al. (2022, ApJ, 262, 34)")
    flag_nd_h_upper_limit_t4 = nd_h_flags.flag(2**3, "At least one line is an upper limit by the 4% threshold in Hayes et al. (2022, ApJ, 262, 34)")
    flag_nd_h_upper_limit_t5 = nd_h_flags.flag(2**4, "At least one line is an upper limit by the 5% threshold in Hayes et al. (2022, ApJ, 262, 34)")
    flag_nd_h_censored_high_teff = nd_h_flags.flag(2**5, "Censored value because abundances known to be wrong for this Teff")
    flag_nd_h_censored_low_teff_vmicro = nd_h_flags.flag(2**6, "Censored value because it has low Teff and v_micro")
    flag_nd_h_censored_unphysical = nd_h_flags.flag(2**7, "Censored value because FERRE returned unphysical value")
    flag_nd_h_bad_grid_edge = nd_h_flags.flag(2**8, "Grid edge bad")
    flag_nd_h_warn_grid_edge = nd_h_flags.flag(2**9, "Grid edge warning")
    flag_nd_h_warn_teff = nd_h_flags.flag(2**10, "These abundances are known to be unreliable for this Teff")
    flag_nd_h_warn_m_h = nd_h_flags.flag(2**11, "These abundances are known to be unreliable for this [M/H]")

    ni_h = FloatField(null=True, help_text=Glossary.ni_h)
    e_ni_h = FloatField(null=True, help_text=Glossary.e_ni_h)
    ni_h_flags = BitField(default=0, help_text=Glossary.ni_h_flags)
    ni_h_rchi2 = FloatField(null=True, help_text=Glossary.ni_h_rchi2)    
    flag_ni_h_censored_high_teff = ni_h_flags.flag(2**5, "Censored value because abundances known to be wrong for this Teff")
    flag_ni_h_censored_low_teff_vmicro = ni_h_flags.flag(2**6, "Censored value because it has low Teff and v_micro")
    flag_ni_h_censored_unphysical = ni_h_flags.flag(2**7, "Censored value because FERRE returned unphysical value")
    flag_ni_h_bad_grid_edge = ni_h_flags.flag(2**8, "Grid edge bad")
    flag_ni_h_warn_grid_edge = ni_h_flags.flag(2**9, "Grid edge warning")
    flag_ni_h_warn_teff = ni_h_flags.flag(2**10, "These abundances are known to be unreliable for this Teff")
    flag_ni_h_warn_m_h = ni_h_flags.flag(2**11, "These abundances are known to be unreliable for this [M/H]")

    n_h = FloatField(null=True, help_text=Glossary.n_h)
    e_n_h = FloatField(null=True, help_text=Glossary.e_n_h)
    n_h_flags = BitField(default=0, help_text=Glossary.n_h_flags)
    n_h_rchi2 = FloatField(null=True, help_text=Glossary.n_h_rchi2)
    flag_n_h_upper_limit_t1 = n_h_flags.flag(2**0, "At least one line is an upper limit by the 1% threshold in Hayes et al. (2022, ApJ, 262, 34)")
    flag_n_h_upper_limit_t2 = n_h_flags.flag(2**1, "At least one line is an upper limit by the 2% threshold in Hayes et al. (2022, ApJ, 262, 34)")
    flag_n_h_upper_limit_t3 = n_h_flags.flag(2**2, "At least one line is an upper limit by the 3% threshold in Hayes et al. (2022, ApJ, 262, 34)")
    flag_n_h_upper_limit_t4 = n_h_flags.flag(2**3, "At least one line is an upper limit by the 4% threshold in Hayes et al. (2022, ApJ, 262, 34)")
    flag_n_h_upper_limit_t5 = n_h_flags.flag(2**4, "At least one line is an upper limit by the 5% threshold in Hayes et al. (2022, ApJ, 262, 34)")
    flag_n_h_censored_high_teff = n_h_flags.flag(2**5, "Censored value because abundances known to be wrong for this Teff")
    flag_n_h_censored_low_teff_vmicro = n_h_flags.flag(2**6, "Censored value because it has low Teff and v_micro")
    flag_n_h_censored_unphysical = n_h_flags.flag(2**7, "Censored value because FERRE returned unphysical value")
    flag_n_h_bad_grid_edge = n_h_flags.flag(2**8, "Grid edge bad")
    flag_n_h_warn_grid_edge = n_h_flags.flag(2**9, "Grid edge warning")
    flag_n_h_warn_teff = n_h_flags.flag(2**10, "These abundances are known to be unreliable for this Teff")
    flag_n_h_warn_m_h = n_h_flags.flag(2**11, "These abundances are known to be unreliable for this [M/H]")

    o_h = FloatField(null=True, help_text=Glossary.o_h)
    e_o_h = FloatField(null=True, help_text=Glossary.e_o_h)
    o_h_flags = BitField(default=0, help_text=Glossary.o_h_flags)
    o_h_rchi2 = FloatField(null=True, help_text=Glossary.o_h_rchi2)
    flag_o_h_upper_limit_t1 = o_h_flags.flag(2**0, "At least one line is an upper limit by the 1% threshold in Hayes et al. (2022, ApJ, 262, 34)")
    flag_o_h_upper_limit_t2 = o_h_flags.flag(2**1, "At least one line is an upper limit by the 2% threshold in Hayes et al. (2022, ApJ, 262, 34)")
    flag_o_h_upper_limit_t3 = o_h_flags.flag(2**2, "At least one line is an upper limit by the 3% threshold in Hayes et al. (2022, ApJ, 262, 34)")
    flag_o_h_upper_limit_t4 = o_h_flags.flag(2**3, "At least one line is an upper limit by the 4% threshold in Hayes et al. (2022, ApJ, 262, 34)")
    flag_o_h_upper_limit_t5 = o_h_flags.flag(2**4, "At least one line is an upper limit by the 5% threshold in Hayes et al. (2022, ApJ, 262, 34)")
    flag_o_h_censored_high_teff = o_h_flags.flag(2**5, "Censored value because abundances known to be wrong for this Teff")
    flag_o_h_censored_low_teff_vmicro = o_h_flags.flag(2**6, "Censored value because it has low Teff and v_micro")
    flag_o_h_censored_unphysical = o_h_flags.flag(2**7, "Censored value because FERRE returned unphysical value")
    flag_o_h_bad_grid_edge = o_h_flags.flag(2**8, "Grid edge bad")
    flag_o_h_warn_grid_edge = o_h_flags.flag(2**9, "Grid edge warning")
    flag_o_h_warn_teff = o_h_flags.flag(2**10, "These abundances are known to be unreliable for this Teff")
    flag_o_h_warn_m_h = o_h_flags.flag(2**11, "These abundances are known to be unreliable for this [M/H]")

    p_h = FloatField(null=True, help_text=Glossary.p_h)
    e_p_h = FloatField(null=True, help_text=Glossary.e_p_h)
    p_h_flags = BitField(default=0, help_text=Glossary.p_h_flags)
    p_h_rchi2 = FloatField(null=True, help_text=Glossary.p_h_rchi2)
    flag_p_h_upper_limit_t1 = p_h_flags.flag(2**0, "At least one line is an upper limit by the 1% threshold in Hayes et al. (2022, ApJ, 262, 34)")
    flag_p_h_upper_limit_t2 = p_h_flags.flag(2**1, "At least one line is an upper limit by the 2% threshold in Hayes et al. (2022, ApJ, 262, 34)")
    flag_p_h_upper_limit_t3 = p_h_flags.flag(2**2, "At least one line is an upper limit by the 3% threshold in Hayes et al. (2022, ApJ, 262, 34)")
    flag_p_h_upper_limit_t4 = p_h_flags.flag(2**3, "At least one line is an upper limit by the 4% threshold in Hayes et al. (2022, ApJ, 262, 34)")
    flag_p_h_upper_limit_t5 = p_h_flags.flag(2**4, "At least one line is an upper limit by the 5% threshold in Hayes et al. (2022, ApJ, 262, 34)")
    flag_p_h_censored_high_teff = p_h_flags.flag(2**5, "Censored value because abundances known to be wrong for this Teff")
    flag_p_h_censored_low_teff_vmicro = p_h_flags.flag(2**6, "Censored value because it has low Teff and v_micro")
    flag_p_h_censored_unphysical = p_h_flags.flag(2**7, "Censored value because FERRE returned unphysical value")
    flag_p_h_bad_grid_edge = p_h_flags.flag(2**8, "Grid edge bad")
    flag_p_h_warn_grid_edge = p_h_flags.flag(2**9, "Grid edge warning")
    flag_p_h_warn_teff = p_h_flags.flag(2**10, "These abundances are known to be unreliable for this Teff")
    flag_p_h_warn_m_h = p_h_flags.flag(2**11, "These abundances are known to be unreliable for this [M/H]")

    si_h = FloatField(null=True, help_text=Glossary.si_h)
    e_si_h = FloatField(null=True, help_text=Glossary.e_si_h)
    si_h_flags = BitField(default=0, help_text=Glossary.si_h_flags)
    si_h_rchi2 = FloatField(null=True, help_text=Glossary.si_h_rchi2)
    flag_si_h_censored_high_teff = si_h_flags.flag(2**5, "Censored value because abundances known to be wrong for this Teff")
    flag_si_h_censored_low_teff_vmicro = si_h_flags.flag(2**6, "Censored value because it has low Teff and v_micro")
    flag_si_h_censored_unphysical = si_h_flags.flag(2**7, "Censored value because FERRE returned unphysical value")
    flag_si_h_bad_grid_edge = si_h_flags.flag(2**8, "Grid edge bad")
    flag_si_h_warn_grid_edge = si_h_flags.flag(2**9, "Grid edge warning")
    flag_si_h_warn_teff = si_h_flags.flag(2**10, "These abundances are known to be unreliable for this Teff")
    flag_si_h_warn_m_h = si_h_flags.flag(2**11, "These abundances are known to be unreliable for this [M/H]")

    s_h = FloatField(null=True, help_text=Glossary.s_h)
    e_s_h = FloatField(null=True, help_text=Glossary.e_s_h)
    s_h_flags = BitField(default=0, help_text=Glossary.s_h_flags)
    s_h_rchi2 = FloatField(null=True, help_text=Glossary.s_h_rchi2)
    flag_s_h_upper_limit_t1 = s_h_flags.flag(2**0, "At least one line is an upper limit by the 1% threshold in Hayes et al. (2022, ApJ, 262, 34)")
    flag_s_h_upper_limit_t2 = s_h_flags.flag(2**1, "At least one line is an upper limit by the 2% threshold in Hayes et al. (2022, ApJ, 262, 34)")
    flag_s_h_upper_limit_t3 = s_h_flags.flag(2**2, "At least one line is an upper limit by the 3% threshold in Hayes et al. (2022, ApJ, 262, 34)")
    flag_s_h_upper_limit_t4 = s_h_flags.flag(2**3, "At least one line is an upper limit by the 4% threshold in Hayes et al. (2022, ApJ, 262, 34)")
    flag_s_h_upper_limit_t5 = s_h_flags.flag(2**4, "At least one line is an upper limit by the 5% threshold in Hayes et al. (2022, ApJ, 262, 34)")
    flag_s_h_censored_high_teff = s_h_flags.flag(2**5, "Censored value because abundances known to be wrong for this Teff")
    flag_s_h_censored_low_teff_vmicro = s_h_flags.flag(2**6, "Censored value because it has low Teff and v_micro")
    flag_s_h_censored_unphysical = s_h_flags.flag(2**7, "Censored value because FERRE returned unphysical value")
    flag_s_h_bad_grid_edge = s_h_flags.flag(2**8, "Grid edge bad")
    flag_s_h_warn_grid_edge = s_h_flags.flag(2**9, "Grid edge warning")
    flag_s_h_warn_teff = s_h_flags.flag(2**10, "These abundances are known to be unreliable for this Teff")
    flag_s_h_warn_m_h = s_h_flags.flag(2**11, "These abundances are known to be unreliable for this [M/H]")

    ti_h = FloatField(null=True, help_text=Glossary.ti_h)
    e_ti_h = FloatField(null=True, help_text=Glossary.e_ti_h)
    ti_h_flags = BitField(default=0, help_text=Glossary.ti_h_flags)
    ti_h_rchi2 = FloatField(null=True, help_text=Glossary.ti_h_rchi2)
    flag_ti_h_censored_high_teff = ti_h_flags.flag(2**5, "Censored value because abundances known to be wrong for this Teff")
    flag_ti_h_censored_low_teff_vmicro = ti_h_flags.flag(2**6, "Censored value because it has low Teff and v_micro")
    flag_ti_h_censored_unphysical = ti_h_flags.flag(2**7, "Censored value because FERRE returned unphysical value")
    flag_ti_h_bad_grid_edge = ti_h_flags.flag(2**8, "Grid edge bad")
    flag_ti_h_warn_grid_edge = ti_h_flags.flag(2**9, "Grid edge warning")
    flag_ti_h_warn_teff = ti_h_flags.flag(2**10, "These abundances are known to be unreliable for this Teff")
    flag_ti_h_warn_m_h = ti_h_flags.flag(2**11, "These abundances are known to be unreliable for this [M/H]")

    ti_2_h = FloatField(null=True, help_text=Glossary.ti_2_h)
    e_ti_2_h = FloatField(null=True, help_text=Glossary.e_ti_2_h)
    ti_2_h_flags = BitField(default=0, help_text=Glossary.ti_2_h_flags)
    ti_2_h_rchi2 = FloatField(null=True, help_text=Glossary.ti_2_h_rchi2)
    flag_ti_2_h_censored_high_teff = ti_2_h_flags.flag(2**5, "Censored value because abundances known to be wrong for this Teff")
    flag_ti_2_h_censored_low_teff_vmicro = ti_2_h_flags.flag(2**6, "Censored value because it has low Teff and v_micro")
    flag_ti_2_h_censored_unphysical = ti_2_h_flags.flag(2**7, "Censored value because FERRE returned unphysical value")
    flag_ti_2_h_bad_grid_edge = ti_2_h_flags.flag(2**8, "Grid edge bad")
    flag_ti_2_h_warn_grid_edge = ti_2_h_flags.flag(2**9, "Grid edge warning")
    flag_ti_2_h_warn_teff = ti_2_h_flags.flag(2**10, "These abundances are known to be unreliable for this Teff")
    flag_ti_2_h_warn_m_h = ti_2_h_flags.flag(2**11, "These abundances are known to be unreliable for this [M/H]")

    v_h = FloatField(null=True, help_text=Glossary.v_h)
    e_v_h = FloatField(null=True, help_text=Glossary.e_v_h)
    v_h_flags = BitField(default=0, help_text=Glossary.v_h_flags)
    v_h_rchi2 = FloatField(null=True, help_text=Glossary.v_h_rchi2)
    flag_v_h_upper_limit_t1 = v_h_flags.flag(2**0, "At least one line is an upper limit by the 1% threshold in Hayes et al. (2022, ApJ, 262, 34)")
    flag_v_h_upper_limit_t2 = v_h_flags.flag(2**1, "At least one line is an upper limit by the 2% threshold in Hayes et al. (2022, ApJ, 262, 34)")
    flag_v_h_upper_limit_t3 = v_h_flags.flag(2**2, "At least one line is an upper limit by the 3% threshold in Hayes et al. (2022, ApJ, 262, 34)")
    flag_v_h_upper_limit_t4 = v_h_flags.flag(2**3, "At least one line is an upper limit by the 4% threshold in Hayes et al. (2022, ApJ, 262, 34)")
    flag_v_h_upper_limit_t5 = v_h_flags.flag(2**4, "At least one line is an upper limit by the 5% threshold in Hayes et al. (2022, ApJ, 262, 34)")
    flag_v_h_censored_high_teff = v_h_flags.flag(2**5, "Censored value because abundances known to be wrong for this Teff")
    flag_v_h_censored_low_teff_vmicro = v_h_flags.flag(2**6, "Censored value because it has low Teff and v_micro")
    flag_v_h_censored_unphysical = v_h_flags.flag(2**7, "Censored value because FERRE returned unphysical value")
    flag_v_h_bad_grid_edge = v_h_flags.flag(2**8, "Grid edge bad")
    flag_v_h_warn_grid_edge = v_h_flags.flag(2**9, "Grid edge warning")
    flag_v_h_warn_teff = v_h_flags.flag(2**10, "These abundances are known to be unreliable for this Teff")
    flag_v_h_warn_m_h = v_h_flags.flag(2**11, "These abundances are known to be unreliable for this [M/H]")

    #> FERRE Settings
    short_grid_name = TextField(default="", help_text="Short name describing the FERRE grid used")
    continuum_order = IntegerField(default=-1, help_text="Continuum order used in FERRE")
    continuum_reject = FloatField(null=True, help_text="Tolerance for FERRE to reject continuum points")
    interpolation_order = IntegerField(default=-1, help_text="Interpolation order used by FERRE")
    initial_flags = BitField(default=0, help_text=Glossary.initial_flags)
    flag_initial_guess_from_apogeenet = initial_flags.flag(2**0, help_text="Initial guess from APOGEENet")
    flag_initial_guess_from_doppler = initial_flags.flag(2**1, help_text="Initial guess from Doppler (SDSS-V)")
    flag_initial_guess_from_doppler_sdss4 = initial_flags.flag(2**1, help_text="Initial guess from Doppler (SDSS-IV)")
    flag_initial_guess_from_gaia_xp_andrae23 = initial_flags.flag(2**3, help_text="Initial guess from Andrae et al. (2023)")
    flag_initial_guess_from_user = initial_flags.flag(2**2, help_text="Initial guess specified by user")

    #> Summary Statistics
    snr = FloatField(null=True, help_text=Glossary.snr)
    rchi2 = FloatField(null=True, help_text=Glossary.rchi2)
    ferre_log_snr_sq = FloatField(null=True, help_text="FERRE-reported log10(snr**2)")
    ferre_time_elapsed = FloatField(null=True, help_text="Total core-second use reported by FERRE [s]")
    result_flags = BitField(default=0, help_text="Flags indicating FERRE issues")

    flag_ferre_fail = result_flags.flag(2**0, "FERRE failed")
    flag_missing_model_flux = result_flags.flag(2**1, "Missing model fluxes from FERRE")
    flag_potential_ferre_timeout = result_flags.flag(2**2, "Potentially impacted by FERRE timeout")
    flag_no_suitable_initial_guess = result_flags.flag(2**3, help_text="FERRE not executed because there's no suitable initial guess")
    flag_spectrum_io_error = result_flags.flag(2**4, help_text="Error accessing spectrum pixel data")
    flag_teff_grid_edge_warn = result_flags.flag(2**5, help_text="Teff is within one step from the grid edge")
    flag_teff_grid_edge_bad = result_flags.flag(2**6, help_text="Teff is within 1/8th of a step from the grid edge")
    flag_logg_grid_edge_warn = result_flags.flag(2**7, help_text="logg is within one step from the grid edge")
    flag_logg_grid_edge_bad = result_flags.flag(2**8, help_text="logg is within 1/8th of a step from the grid edge")
    flag_v_micro_grid_edge_warn = result_flags.flag(2**9, help_text="v_micro is within one step from the grid edge")
    flag_v_micro_grid_edge_bad = result_flags.flag(2**10, help_text="v_micro is within 1/8th of a step from the grid edge")
    flag_v_sini_grid_edge_warn = result_flags.flag(2**11, help_text="v_sini is within one step from the highest grid edge")
    flag_v_sini_grid_edge_bad = result_flags.flag(2**12, help_text="v_sini is within 1/8th of a step from the highest grid edge")
    flag_m_h_atm_grid_edge_warn = result_flags.flag(2**13, help_text="[M/H] is within one step from the grid edge")
    flag_m_h_atm_grid_edge_bad = result_flags.flag(2**14, help_text="[M/H] is within 1/8th of a step from the grid edge")
    flag_alpha_m_grid_edge_warn = result_flags.flag(2**15, help_text="[alpha/M] is within one step from the grid edge")
    flag_alpha_m_grid_edge_bad = result_flags.flag(2**16, help_text="[alpha/M] is within 1/8th of a step from the grid edge")
    flag_c_m_atm_grid_edge_warn = result_flags.flag(2**17, help_text="[C/M] is within one step from the grid edge")
    flag_c_m_atm_grid_edge_bad = result_flags.flag(2**18, help_text="[C/M] is within 1/8th of a step from the grid edge")
    flag_n_m_atm_grid_edge_warn = result_flags.flag(2**19, help_text="[N/M] is within one step from the grid edge")
    flag_n_m_atm_grid_edge_bad = result_flags.flag(2**20, help_text="[N/M] is within 1/8th of a step from the grid edge")    
    flag_suspicious_parameters = result_flags.flag(2**21, help_text="Stellar parameters are in a suspicious and low-density region")
    flag_high_v_sini = result_flags.flag(2**22, help_text="High rotational velocity")
    flag_high_v_micro = result_flags.flag(2**23, help_text="v_micro exceeds 3 km/s")
    flag_unphysical_parameters = result_flags.flag(2**24, help_text="FERRE returned unphysical stellar parameters")
    flag_high_rchi2 = result_flags.flag(2**25, help_text="Reduced chi-squared is greater than 1000")
    flag_low_snr = result_flags.flag(2**26, help_text="S/N is less than 20")
    flag_high_std_v_rad = result_flags.flag(2**27, help_text="Standard deviation of v_rad is greater than 1 km/s")
    
    @hybrid_property
    def flag_warn(self):
        return (self.result_flags > 0)

    @flag_warn.expression
    def flag_warn(self):
        return (self.result_flags > 0)

    @hybrid_property
    def flag_bad(self):
        return (
            self.flag_suspicious_parameters
        |   self.flag_high_v_sini
        |   self.flag_high_v_micro
        |   self.flag_unphysical_parameters
        |   self.flag_high_rchi2
        |   self.flag_low_snr
        |   self.flag_high_std_v_rad
        |   self.flag_teff_grid_edge_bad
        |   self.flag_logg_grid_edge_bad
        |   self.flag_ferre_fail
        |   self.flag_missing_model_flux
        |   self.flag_potential_ferre_timeout
        |   self.flag_no_suitable_initial_guess
        |   self.flag_spectrum_io_error
    )

    @flag_bad.expression
    def flag_bad(self):
        return (
            self.flag_suspicious_parameters
        |   self.flag_high_v_sini
        |   self.flag_high_v_micro
        |   self.flag_unphysical_parameters
        |   self.flag_high_rchi2
        |   self.flag_low_snr
        |   self.flag_high_std_v_rad
        |   self.flag_teff_grid_edge_bad
        |   self.flag_logg_grid_edge_bad
        |   self.flag_ferre_fail
        |   self.flag_missing_model_flux
        |   self.flag_potential_ferre_timeout
        |   self.flag_no_suitable_initial_guess
        |   self.flag_spectrum_io_error
    )

        
    
    
    
    #> Task Primary Keys
    stellar_parameters_task_pk = ForeignKeyField(FerreStellarParameters, unique=True, null=True, lazy_load=False, help_text="Task primary key for stellar parameters")
    al_h_task_pk = ForeignKeyField(FerreChemicalAbundances, unique=True, null=True, lazy_load=False, help_text="Task primary key for [Al/H]")
    c_12_13_task_pk = ForeignKeyField(FerreChemicalAbundances, unique=True, null=True, lazy_load=False, help_text="Task primary key for C12/C13")
    ca_h_task_pk = ForeignKeyField(FerreChemicalAbundances, unique=True, null=True, lazy_load=False, help_text="Task primary key for [Ca/H]")
    ce_h_task_pk = ForeignKeyField(FerreChemicalAbundances, unique=True, null=True, lazy_load=False, help_text="Task primary key for [Ce/H]")
    c_1_h_task_pk = ForeignKeyField(FerreChemicalAbundances, unique=True, null=True, lazy_load=False, help_text="Task primary key for [C 1/H]")
    c_h_task_pk = ForeignKeyField(FerreChemicalAbundances, unique=True, null=True, lazy_load=False, help_text="Task primary key for [C/H]")
    co_h_task_pk = ForeignKeyField(FerreChemicalAbundances, unique=True, null=True, lazy_load=False, help_text="Task primary key for [Co/H]")
    cr_h_task_pk = ForeignKeyField(FerreChemicalAbundances, unique=True, null=True, lazy_load=False, help_text="Task primary key for [Cr/H]")
    cu_h_task_pk = ForeignKeyField(FerreChemicalAbundances, unique=True, null=True, lazy_load=False, help_text="Task primary key for [Cu/H]")
    fe_h_task_pk = ForeignKeyField(FerreChemicalAbundances, unique=True, null=True, lazy_load=False, help_text="Task primary key for [Fe/H]")
    k_h_task_pk = ForeignKeyField(FerreChemicalAbundances, unique=True, null=True, lazy_load=False, help_text="Task primary key for [K/H]")
    mg_h_task_pk = ForeignKeyField(FerreChemicalAbundances, unique=True, null=True, lazy_load=False, help_text="Task primary key for [Mg/H]")
    mn_h_task_pk = ForeignKeyField(FerreChemicalAbundances, unique=True, null=True, lazy_load=False, help_text="Task primary key for [Mn/H]")
    na_h_task_pk = ForeignKeyField(FerreChemicalAbundances, unique=True, null=True, lazy_load=False, help_text="Task primary key for [Na/H]")
    nd_h_task_pk = ForeignKeyField(FerreChemicalAbundances, unique=True, null=True, lazy_load=False, help_text="Task primary key for [Nd/H]")
    ni_h_task_pk = ForeignKeyField(FerreChemicalAbundances, unique=True, null=True, lazy_load=False, help_text="Task primary key for [Ni/H]")
    n_h_task_pk = ForeignKeyField(FerreChemicalAbundances, unique=True, null=True, lazy_load=False, help_text="Task primary key for [N/H]")
    o_h_task_pk = ForeignKeyField(FerreChemicalAbundances, unique=True, null=True, lazy_load=False, help_text="Task primary key for [O/H]")
    p_h_task_pk = ForeignKeyField(FerreChemicalAbundances, unique=True, null=True, lazy_load=False, help_text="Task primary key for [P/H]")
    si_h_task_pk = ForeignKeyField(FerreChemicalAbundances, unique=True, null=True, lazy_load=False, help_text="Task primary key for [Si/H]")
    s_h_task_pk = ForeignKeyField(FerreChemicalAbundances, unique=True, null=True, lazy_load=False, help_text="Task primary key for [S/H]")
    ti_h_task_pk = ForeignKeyField(FerreChemicalAbundances, unique=True, null=True, lazy_load=False, help_text="Task primary key for [Ti/H]")
    ti_2_h_task_pk = ForeignKeyField(FerreChemicalAbundances, unique=True, null=True, lazy_load=False, help_text="Task primary key for [Ti 2/H]")
    v_h_task_pk = ForeignKeyField(FerreChemicalAbundances, unique=True, null=True, lazy_load=False, help_text="Task primary key for [V/H]")

    #> Calibration flags
    calibrated_flags = BitField(null=True, help_text="Calibration flags")
    flag_as_dwarf_for_calibration = calibrated_flags.flag(2**0, "Classified as main-sequence star for logg calibration")
    flag_as_giant_for_calibration = calibrated_flags.flag(2**1, "Classified as red giant branch star for logg calibration")
    flag_as_red_clump_for_calibration = calibrated_flags.flag(2**2, "Classified as red clump star for logg calibration")
    flag_as_m_dwarf_for_calibration = calibrated_flags.flag(2**3, "Classifed as M-dwarf for teff and logg calibration")
    flag_censored_logg_for_metal_poor_m_dwarf = calibrated_flags.flag(2**4, "Censored logg for metal-poor ([M/H] < -0.6) M-dwarf")
    mass = FloatField(null=True, help_text="Mass inferred from isochrones [M_sun]")
    radius = FloatField(null=True, help_text="Radius inferred from isochrones [R_sun]")
    
    #> Raw (Uncalibrated) Quantities
    raw_teff = FloatField(null=True, help_text=Glossary.raw_teff)
    raw_e_teff = FloatField(null=True, help_text=Glossary.raw_e_teff)
    raw_logg = FloatField(null=True, help_text=Glossary.raw_logg)
    raw_e_logg = FloatField(null=True, help_text=Glossary.raw_e_logg)
    raw_v_micro = FloatField(null=True, help_text=Glossary.raw_v_micro)
    raw_e_v_micro = FloatField(null=True, help_text=Glossary.raw_e_v_micro)
    raw_v_sini = FloatField(null=True, help_text=Glossary.raw_v_sini)
    raw_e_v_sini = FloatField(null=True, help_text=Glossary.raw_e_v_sini)
    raw_m_h_atm = FloatField(null=True, help_text=Glossary.raw_m_h_atm)
    raw_e_m_h_atm = FloatField(null=True, help_text=Glossary.raw_e_m_h_atm)
    raw_alpha_m_atm = FloatField(null=True, help_text=Glossary.raw_alpha_m_atm)
    raw_e_alpha_m_atm = FloatField(null=True, help_text=Glossary.raw_e_alpha_m_atm)
    raw_c_m_atm = FloatField(null=True, help_text=Glossary.raw_c_m_atm)
    raw_e_c_m_atm = FloatField(null=True, help_text=Glossary.raw_e_c_m_atm)
    raw_n_m_atm = FloatField(null=True, help_text=Glossary.raw_n_m_atm)
    raw_e_n_m_atm = FloatField(null=True, help_text=Glossary.raw_e_n_m_atm)
    raw_al_h = FloatField(null=True, help_text=Glossary.raw_al_h)
    raw_e_al_h = FloatField(null=True, help_text=Glossary.raw_e_al_h)
    raw_c_12_13 = FloatField(null=True, help_text=Glossary.raw_c_12_13)
    raw_e_c_12_13 = FloatField(null=True, help_text=Glossary.raw_e_c_12_13)
    raw_ca_h = FloatField(null=True, help_text=Glossary.raw_ca_h)
    raw_e_ca_h = FloatField(null=True, help_text=Glossary.raw_e_ca_h)
    raw_ce_h = FloatField(null=True, help_text=Glossary.raw_ce_h)
    raw_e_ce_h = FloatField(null=True, help_text=Glossary.raw_e_ce_h)
    raw_c_1_h = FloatField(null=True, help_text=Glossary.raw_c_1_h)
    raw_e_c_1_h = FloatField(null=True, help_text=Glossary.raw_e_c_1_h)
    raw_c_h = FloatField(null=True, help_text=Glossary.raw_c_h)
    raw_e_c_h = FloatField(null=True, help_text=Glossary.raw_e_c_h)
    raw_co_h = FloatField(null=True, help_text=Glossary.raw_co_h)
    raw_e_co_h = FloatField(null=True, help_text=Glossary.raw_e_co_h)
    raw_cr_h = FloatField(null=True, help_text=Glossary.raw_cr_h)
    raw_e_cr_h = FloatField(null=True, help_text=Glossary.raw_e_cr_h)
    raw_cu_h = FloatField(null=True, help_text=Glossary.raw_cu_h)
    raw_e_cu_h = FloatField(null=True, help_text=Glossary.raw_e_cu_h) 
    raw_fe_h = FloatField(null=True, help_text=Glossary.raw_fe_h)
    raw_e_fe_h = FloatField(null=True, help_text=Glossary.raw_e_fe_h)
    raw_k_h = FloatField(null=True, help_text=Glossary.raw_k_h)
    raw_e_k_h = FloatField(null=True, help_text=Glossary.raw_e_k_h)
    raw_mg_h = FloatField(null=True, help_text=Glossary.raw_mg_h)
    raw_e_mg_h = FloatField(null=True, help_text=Glossary.raw_e_mg_h)
    raw_mn_h = FloatField(null=True, help_text=Glossary.raw_mn_h)
    raw_e_mn_h = FloatField(null=True, help_text=Glossary.raw_e_mn_h)
    raw_na_h = FloatField(null=True, help_text=Glossary.raw_na_h)
    raw_e_na_h = FloatField(null=True, help_text=Glossary.raw_e_na_h)
    raw_nd_h = FloatField(null=True, help_text=Glossary.raw_nd_h)
    raw_e_nd_h = FloatField(null=True, help_text=Glossary.raw_e_nd_h)
    raw_ni_h = FloatField(null=True, help_text=Glossary.raw_ni_h)
    raw_e_ni_h = FloatField(null=True, help_text=Glossary.raw_e_ni_h)
    raw_n_h = FloatField(null=True, help_text=Glossary.raw_n_h)
    raw_e_n_h = FloatField(null=True, help_text=Glossary.raw_e_n_h)
    raw_o_h = FloatField(null=True, help_text=Glossary.raw_o_h)
    raw_e_o_h = FloatField(null=True, help_text=Glossary.raw_e_o_h)
    raw_p_h = FloatField(null=True, help_text=Glossary.raw_p_h)
    raw_e_p_h = FloatField(null=True, help_text=Glossary.raw_e_p_h)
    raw_si_h = FloatField(null=True, help_text=Glossary.raw_si_h)
    raw_e_si_h = FloatField(null=True, help_text=Glossary.raw_e_si_h)
    raw_s_h = FloatField(null=True, help_text=Glossary.raw_s_h)
    raw_e_s_h = FloatField(null=True, help_text=Glossary.raw_e_s_h)
    raw_ti_h = FloatField(null=True, help_text=Glossary.raw_ti_h)
    raw_e_ti_h = FloatField(null=True, help_text=Glossary.raw_e_ti_h)
    raw_ti_2_h = FloatField(null=True, help_text=Glossary.raw_ti_2_h)
    raw_e_ti_2_h = FloatField(null=True, help_text=Glossary.raw_e_ti_2_h)
    raw_v_h = FloatField(null=True, help_text=Glossary.raw_v_h)
    raw_e_v_h = FloatField(null=True, help_text=Glossary.raw_e_v_h)
    



def apply_noise_model():
    import pickle
    from astra import __version__
    from astra.utils import expand_path
    
    with open(expand_path(f"$MWM_ASTRA/{__version__}/aux/ASPCAP_corrections.pkl"), "rb") as fp:
        corrections, reference = pickle.load(fp)

    update_kwds = {}
    for label_name, kwds in corrections.items():
        offset, scale = kwds["offset"], kwds["scale"]
        update_kwds[f"e_{label_name}"] = scale * getattr(ASPCAP, f"raw_e_{label_name}") + offset
        
    (
        ASPCAP
        .update(**update_kwds)
        .where(ASPCAP.v_astra == __version__)
        .execute()
    )
    
        
