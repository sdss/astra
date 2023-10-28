from peewee import (
    AutoField,
    FloatField,
    TextField,
    ForeignKeyField,
    IntegerField,
    BitField,
    DateTimeField,
    BooleanField,
)

import datetime
import numpy as np

from astra import __version__
from astra.models.base import BaseModel
from astra.models.fields import BitField, PixelArray, BasePixelArrayAccessor, LogLambdaArrayAccessor
from astra.models.ferre import FerreCoarse, FerreStellarParameters, FerreChemicalAbundances
from astra.models.source import Source
from astra.models.spectrum import Spectrum
from astra.models.pipeline import PipelineOutputMixin
from astra.glossary import Glossary


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
        

class ASPCAP(BaseModel, PipelineOutputMixin):

    """ APOGEE Stellar Parameter and Chemical Abundances Pipeline (ASPCAP) """
    
    #> Identifiers
    spectrum_pk = ForeignKeyField(Spectrum, index=True, lazy_load=False, help_text=Glossary.spectrum_pk)    
    source_pk = ForeignKeyField(Source, index=True, lazy_load=False)
    
    #> Astra Metadata
    task_pk = AutoField(help_text=Glossary.task_pk)
    v_astra = TextField(default=__version__, help_text=Glossary.v_astra)
    created = DateTimeField(default=datetime.datetime.now, help_text=Glossary.created) 
    t_elapsed = FloatField(null=True, help_text=Glossary.t_elapsed)
    t_overhead = FloatField(null=True, help_text=Glossary.t_overhead)
    tag = TextField(default="", index=True, help_text=Glossary.tag)
    
    #> Spectral Data
    wavelength = PixelArray(
        accessor_class=LogLambdaArrayAccessor,
        accessor_kwargs=dict(
            crval=4.179,
            cdelt=6e-6,
            naxis=8575,
        ),
        help_text=Glossary.wavelength
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

    #> IRFM Temperatures (Casagrande et al. 2010)
    irfm_teff = FloatField(null=True, help_text=Glossary.teff)
    e_irfm_teff = FloatField(null=True, help_text=Glossary.e_teff)
    irfm_teff_flags = BitField(default=0, help_text="IRFM temperature flags")
    flag_out_of_v_k_bounds = irfm_teff_flags.flag(2**0, "Out of V-Ks bounds (0.78, 3.15)")
    flag_extrapolated_v_mag = irfm_teff_flags.flag(2**1, "Synthetic V magnitude is extrapolated")
    flag_poor_quality_k_mag = irfm_teff_flags.flag(2**2, "Poor quality Ks magnitude")
    flag_ebv_used_is_upper_limit = irfm_teff_flags.flag(2**3, "E(B-V) used is an upper limit")

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

    c_12_13 = FloatField(null=True, help_text=Glossary.c_12_13)
    e_c_12_13 = FloatField(null=True, help_text=Glossary.e_c_12_13)
    c_12_13_flags = BitField(default=0, help_text=Glossary.c_12_13_flags)
    c_12_13_rchi2 = FloatField(null=True, help_text=Glossary.c_12_13_rchi2)

    ca_h = FloatField(null=True, help_text=Glossary.ca_h)
    e_ca_h = FloatField(null=True, help_text=Glossary.e_ca_h)
    ca_h_flags = BitField(default=0, help_text=Glossary.ca_h_flags)
    ca_h_rchi2 = FloatField(null=True, help_text=Glossary.ca_h_rchi2)
    
    ce_h = FloatField(null=True, help_text=Glossary.ce_h)
    e_ce_h = FloatField(null=True, help_text=Glossary.e_ce_h)
    ce_h_flags = BitField(default=0, help_text=Glossary.ce_h_flags)
    ce_h_rchi2 = FloatField(null=True, help_text=Glossary.ce_h_rchi2)
    
    c_1_h = FloatField(null=True, help_text=Glossary.c_1_h)
    e_c_1_h = FloatField(null=True, help_text=Glossary.e_c_1_h)
    c_1_h_flags = BitField(default=0, help_text=Glossary.c_1_h_flags)
    c_1_h_rchi2 = FloatField(null=True, help_text=Glossary.c_1_h_rchi2)
    
    c_h = FloatField(null=True, help_text=Glossary.c_h)
    e_c_h = FloatField(null=True, help_text=Glossary.e_c_h)
    c_h_flags = BitField(default=0, help_text=Glossary.c_h_flags)
    c_h_rchi2 = FloatField(null=True, help_text=Glossary.c_h_rchi2)
    
    co_h = FloatField(null=True, help_text=Glossary.co_h)
    e_co_h = FloatField(null=True, help_text=Glossary.e_co_h)
    co_h_flags = BitField(default=0, help_text=Glossary.co_h_flags)
    co_h_rchi2 = FloatField(null=True, help_text=Glossary.co_h_rchi2)
    
    cr_h = FloatField(null=True, help_text=Glossary.cr_h)
    e_cr_h = FloatField(null=True, help_text=Glossary.e_cr_h)
    cr_h_flags = BitField(default=0, help_text=Glossary.cr_h_flags)
    cr_h_rchi2 = FloatField(null=True, help_text=Glossary.cr_h_rchi2)
    
    cu_h = FloatField(null=True, help_text=Glossary.cu_h)
    e_cu_h = FloatField(null=True, help_text=Glossary.e_cu_h)
    cu_h_flags = BitField(default=0, help_text=Glossary.cu_h_flags)
    cu_h_rchi2 = FloatField(null=True, help_text=Glossary.cu_h_rchi2)
    
    fe_h = FloatField(null=True, help_text=Glossary.fe_h)
    e_fe_h = FloatField(null=True, help_text=Glossary.e_fe_h)
    fe_h_flags = BitField(default=0, help_text=Glossary.fe_h_flags)
    fe_h_rchi2 = FloatField(null=True, help_text=Glossary.fe_h_rchi2)

    k_h = FloatField(null=True, help_text=Glossary.k_h)
    e_k_h = FloatField(null=True, help_text=Glossary.e_k_h)
    k_h_flags = BitField(default=0, help_text=Glossary.k_h_flags)
    k_h_rchi2 = FloatField(null=True, help_text=Glossary.k_h_rchi2)

    mg_h = FloatField(null=True, help_text=Glossary.mg_h)
    e_mg_h = FloatField(null=True, help_text=Glossary.e_mg_h)
    mg_h_flags = BitField(default=0, help_text=Glossary.mg_h_flags)
    mg_h_rchi2 = FloatField(null=True, help_text=Glossary.mg_h_rchi2)

    mn_h = FloatField(null=True, help_text=Glossary.mn_h)
    e_mn_h = FloatField(null=True, help_text=Glossary.e_mn_h)
    mn_h_flags = BitField(default=0, help_text=Glossary.mn_h_flags)
    mn_h_rchi2 = FloatField(null=True, help_text=Glossary.mn_h_rchi2)

    na_h = FloatField(null=True, help_text=Glossary.na_h)
    e_na_h = FloatField(null=True, help_text=Glossary.e_na_h)
    na_h_flags = BitField(default=0, help_text=Glossary.na_h_flags)
    na_h_rchi2 = FloatField(null=True, help_text=Glossary.na_h_rchi2)

    nd_h = FloatField(null=True, help_text=Glossary.nd_h)
    e_nd_h = FloatField(null=True, help_text=Glossary.e_nd_h)
    nd_h_flags = BitField(default=0, help_text=Glossary.nd_h_flags)
    nd_h_rchi2 = FloatField(null=True, help_text=Glossary.nd_h_rchi2)

    ni_h = FloatField(null=True, help_text=Glossary.ni_h)
    e_ni_h = FloatField(null=True, help_text=Glossary.e_ni_h)
    ni_h_flags = BitField(default=0, help_text=Glossary.ni_h_flags)
    ni_h_rchi2 = FloatField(null=True, help_text=Glossary.ni_h_rchi2)

    n_h = FloatField(null=True, help_text=Glossary.n_h)
    e_n_h = FloatField(null=True, help_text=Glossary.e_n_h)
    n_h_flags = BitField(default=0, help_text=Glossary.n_h_flags)
    n_h_rchi2 = FloatField(null=True, help_text=Glossary.n_h_rchi2)

    o_h = FloatField(null=True, help_text=Glossary.o_h)
    e_o_h = FloatField(null=True, help_text=Glossary.e_o_h)
    o_h_flags = BitField(default=0, help_text=Glossary.o_h_flags)
    o_h_rchi2 = FloatField(null=True, help_text=Glossary.o_h_rchi2)

    p_h = FloatField(null=True, help_text=Glossary.p_h)
    e_p_h = FloatField(null=True, help_text=Glossary.e_p_h)
    p_h_flags = BitField(default=0, help_text=Glossary.p_h_flags)
    p_h_rchi2 = FloatField(null=True, help_text=Glossary.p_h_rchi2)

    si_h = FloatField(null=True, help_text=Glossary.si_h)
    e_si_h = FloatField(null=True, help_text=Glossary.e_si_h)
    si_h_flags = BitField(default=0, help_text=Glossary.si_h_flags)
    si_h_rchi2 = FloatField(null=True, help_text=Glossary.si_h_rchi2)

    s_h = FloatField(null=True, help_text=Glossary.s_h)
    e_s_h = FloatField(null=True, help_text=Glossary.e_s_h)
    s_h_flags = BitField(default=0, help_text=Glossary.s_h_flags)
    s_h_rchi2 = FloatField(null=True, help_text=Glossary.s_h_rchi2)

    ti_h = FloatField(null=True, help_text=Glossary.ti_h)
    e_ti_h = FloatField(null=True, help_text=Glossary.e_ti_h)
    ti_h_flags = BitField(default=0, help_text=Glossary.ti_h_flags)
    ti_h_rchi2 = FloatField(null=True, help_text=Glossary.ti_h_rchi2)

    ti_2_h = FloatField(null=True, help_text=Glossary.ti_2_h)
    e_ti_2_h = FloatField(null=True, help_text=Glossary.e_ti_2_h)
    ti_2_h_flags = BitField(default=0, help_text=Glossary.ti_2_h_flags)
    ti_2_h_rchi2 = FloatField(null=True, help_text=Glossary.ti_2_h_rchi2)

    v_h = FloatField(null=True, help_text=Glossary.v_h)
    e_v_h = FloatField(null=True, help_text=Glossary.e_v_h)
    v_h_flags = BitField(default=0, help_text=Glossary.v_h_flags)
    v_h_rchi2 = FloatField(null=True, help_text=Glossary.v_h_rchi2)

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
    ferre_flags = BitField(default=0, help_text="Flags indicating FERRE issues")

    flag_ferre_fail = ferre_flags.flag(2**0, "FERRE failed")
    flag_missing_model_flux = ferre_flags.flag(2**1, "Missing model fluxes from FERRE")
    flag_potential_ferre_timeout = ferre_flags.flag(2**2, "Potentially impacted by FERRE timeout")
    flag_no_suitable_initial_guess = ferre_flags.flag(2**3, help_text="FERRE not executed because there's no suitable initial guess")
    flag_spectrum_io_error = ferre_flags.flag(2**4, help_text="Error accessing spectrum pixel data")
    flag_teff_grid_edge_warn = ferre_flags.flag(2**5)
    flag_teff_grid_edge_bad = ferre_flags.flag(2**6)
    flag_logg_grid_edge_warn = ferre_flags.flag(2**7)
    flag_logg_grid_edge_bad = ferre_flags.flag(2**8)
    flag_v_micro_grid_edge_warn = ferre_flags.flag(2**9)
    flag_v_micro_grid_edge_bad = ferre_flags.flag(2**10)
    flag_v_sini_grid_edge_warn = ferre_flags.flag(2**11)
    flag_v_sini_grid_edge_bad = ferre_flags.flag(2**12)
    flag_m_h_atm_grid_edge_warn = ferre_flags.flag(2**13)
    flag_m_h_atm_grid_edge_bad = ferre_flags.flag(2**14)
    flag_alpha_m_grid_edge_warn = ferre_flags.flag(2**15)
    flag_alpha_m_grid_edge_bad = ferre_flags.flag(2**16)
    flag_c_m_atm_grid_edge_warn = ferre_flags.flag(2**17)
    flag_c_m_atm_grid_edge_bad = ferre_flags.flag(2**18)
    flag_n_m_atm_grid_edge_warn = ferre_flags.flag(2**19)
    flag_n_m_atm_grid_edge_bad = ferre_flags.flag(2**20)    
    
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

    #> Raw (Uncalibrated) Quantities
    calibrated_flags = BitField(null=True, help_text="Calibration flags")
    flag_main_sequence = calibrated_flags.flag(2**0, "Classified as main-sequence star for calibration")
    flag_red_giant_branch = calibrated_flags.flag(2**1, "Classified as red giant branch star for calibration")
    flag_red_clump = calibrated_flags.flag(2**2, "Classified as red clump star for calibration")
    
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
    
