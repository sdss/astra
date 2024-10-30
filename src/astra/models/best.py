import datetime
import numpy as np
from astra import __version__
from astra.models.base import BaseModel
from astra.fields import (
    AutoField, FloatField, TextField, ForeignKeyField, IntegerField, BigIntegerField,
    BitField, PixelArray, BasePixelArrayAccessor, LogLambdaArrayAccessor, DateTimeField,
    BooleanField
)
from astra.models.source import Source
from astra.models.spectrum import Spectrum
from playhouse.hybrid import hybrid_property


class MWMBest(BaseModel):

    """The best of possibly many bad options."""
    
    #> Identifiers
    spectrum_pk = ForeignKeyField(Spectrum, index=True, unique=True, lazy_load=False, help_text=Glossary.spectrum_pk)    
    source_pk = ForeignKeyField(Source, index=True, lazy_load=False)
    
    #> Astra Metadata
    v_astra = TextField(default=__version__, help_text=Glossary.v_astra)
    created = DateTimeField(default=datetime.datetime.now, help_text=Glossary.created)
    modified = DateTimeField(default=datetime.datetime.now, help_text=Glossary.modified)
    pipeline = TextField(help_text=Glossary.pipeline)
    task_pk = BigIntegerField(help_text=Glossary.task_pk)

    #> Common Spectrum Path Fields
    release = TextField(help_text=Glossary.release)
    filetype = TextField(help_text=Glossary.filetype)
    telescope = TextField(help_text=Glossary.telescope)

    #> APOGEE Spectrum Metadata
    apred = TextField(null=True, help_text=Glossary.apred)
    apstar = TextField(null=True, help_text=Glossary.apstar)
    obj = TextField(null=True, help_text=Glossary.obj)
    field = TextField(null=True, default="", help_text=Glossary.field) # not used in SDSS-V
    prefix = TextField(null=True, default="", help_text=Glossary.prefix) # not used in SDSS-V

    #> BOSS Spectrum Metadata
    run2d = TextField(null=True, help_text=Glossary.run2d)
    # All other BOSS spectrum metadata is at the Source-level (sdss_id, healpix, etc) as it should be

    #> Spectrum Metadata
    snr = FloatField(null=True, help_text=Glossary.snr)
    mean_fiber = FloatField(null=True, help_text="S/N-weighted mean visit fiber number")
    std_fiber = FloatField(null=True, help_text="Standard deviation of visit fiber numbers")
    gri_gaia_transform_flags = BitField(default=0, help_text="Flags for provenance of ugriz photometry")
    zwarning_flags = BitField(default=0, help_text="BOSS DRP warning flags") 
    spectrum_flags = BitField(default=0, help_text=Glossary.spectrum_flags)

    #> Observing Span
    min_mjd = IntegerField(null=True, help_text="Minimum MJD of visits")
    max_mjd = IntegerField(null=True, help_text="Maximum MJD of visits")
    n_visits = IntegerField(null=True, help_text="Number of visits")
    n_good_visits = IntegerField(null=True, help_text="Number of 'good' visits")
    n_good_rvs = IntegerField(null=True, help_text="Number of 'good' radial velocities")

    #> Radial Velocity
    v_rad = FloatField(null=True, help_text=Glossary.v_rad)
    e_v_rad = FloatField(null=True, help_text=Glossary.e_v_rad)
    std_v_rad = FloatField(null=True, help_text="Standard deviation of visit V_RAD [km/s]")
    median_e_v_rad = FloatField(null=True, help_text=Glossary.median_e_v_rad)    

    #> Radial Velocity (XCSAO)
    xcsao_teff = FloatField(null=True, help_text=Glossary.teff)
    xcsao_e_teff = FloatField(null=True, help_text=Glossary.e_teff)
    xcsao_logg = FloatField(null=True, help_text=Glossary.logg)
    xcsao_e_logg = FloatField(null=True, help_text=Glossary.e_logg)
    xcsao_fe_h = FloatField(null=True, help_text=Glossary.fe_h)
    xcsao_e_fe_h = FloatField(null=True, help_text=Glossary.e_fe_h)
    xcsao_meanrxc = FloatField(null=True, help_text="Cross-correlation R-value (1979AJ.....84.1511T)")

    #> Radial Velocity (Doppler)
    doppler_teff = FloatField(null=True, help_text=Glossary.teff)
    doppler_e_teff = FloatField(null=True, help_text=Glossary.e_teff)
    doppler_logg = FloatField(null=True, help_text=Glossary.logg)
    doppler_e_logg = FloatField(null=True, help_text=Glossary.e_logg)
    doppler_fe_h = FloatField(null=True, help_text=Glossary.fe_h)
    doppler_e_fe_h = FloatField(null=True, help_text=Glossary.e_fe_h)
    doppler_rchi2 = FloatField(null=True, help_text="Reduced chi-square value of DOPPLER fit")
    doppler_flags = BitField(default=0, help_text="DOPPLER flags") # TODO: is this actually STARFLAG from the DRP?

    #> Radial Velocity (X-Correlation)
    xcorr_v_rad = FloatField(null=True, help_text=Glossary.v_rad)
    xcorr_v_rel = FloatField(null=True, help_text=Glossary.v_rel)
    xcorr_e_v_rel = FloatField(null=True, help_text=Glossary.e_v_rel)
    ccfwhm = FloatField(null=True, help_text=Glossary.ccfwhm)
    autofwhm = FloatField(null=True, help_text=Glossary.autofwhm)
    n_components = IntegerField(null=True, help_text=Glossary.n_components)    

    #> Radial Velocity (BOSSNet)
    boss_net_v_rad = FloatField(null=True, help_text=Glossary.v_rad)
    boss_net_e_v_rad = FloatField(null=True, help_text=Glossary.v_rel)
    
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

    #> White Dwarf Classifications
    classification = TextField(null=True, help_text="Classification")
    p_cv = FloatField(null=True, help_text="Cataclysmic variable probability")
    p_da = FloatField(null=True, help_text="DA-type white dwarf probability")
    p_dab = FloatField(null=True, help_text="DAB-type white dwarf probability")
    p_dabz = FloatField(null=True, help_text="DABZ-type white dwarf probability")
    p_dah = FloatField(null=True, help_text="DA (H)-type white dwarf probability")
    p_dahe = FloatField(null=True, help_text="DA (He)-type white dwarf probability")
    p_dao = FloatField(null=True, help_text="DAO-type white dwarf probability")
    p_daz = FloatField(null=True, help_text="DAZ-type white dwarf probability")
    p_da_ms = FloatField(null=True, help_text="DA-MS binary probability")
    p_db = FloatField(null=True, help_text="DB-type white dwarf probability")
    p_dba = FloatField(null=True, help_text="DBA-type white dwarf probability")
    p_dbaz = FloatField(null=True, help_text="DBAZ-type white dwarf probability")
    p_dbh = FloatField(null=True, help_text="DB (H)-type white dwarf probability")
    p_dbz = FloatField(null=True, help_text="DBZ-type white dwarf probability")
    p_db_ms = FloatField(null=True, help_text="DB-MS binary probability")
    p_dc = FloatField(null=True, help_text="DC-type white dwarf probability")
    p_dc_ms = FloatField(null=True, help_text="DC-MS binary probability")
    p_do = FloatField(null=True, help_text="DO-type white dwarf probability")
    p_dq = FloatField(null=True, help_text="DQ-type white dwarf probability")
    p_dqz = FloatField(null=True, help_text="DQZ-type white dwarf probability")
    p_dqpec = FloatField(null=True, help_text="DQ Peculiar-type white dwarf probability")
    p_dz = FloatField(null=True, help_text="DZ-type white dwarf probability")
    p_dza = FloatField(null=True, help_text="DZA-type white dwarf probability")
    p_dzb = FloatField(null=True, help_text="DZB-type white dwarf probability")
    p_dzba = FloatField(null=True, help_text="DZBA-type white dwarf probability")
    p_mwd = FloatField(null=True, help_text="Main sequence star probability")
    p_hotdq = FloatField(null=True, help_text="Hot DQ-type white dwarf probability")     

    #> M-Dwarf Classifications
    spectral_type = TextField(null=True, help_text=Glossary.spectral_type)
    sub_type = FloatField(null=True, help_text=Glossary.sub_type)

    #> Summary Statistics
    rchi2 = FloatField(null=True, help_text=Glossary.rchi2)
    pipeline_flags = BitField(default=0, help_text="Amalgamated pipeline flags")
    
    class Meta:
        indexes = (
            (
                # Unique per source
                (
                    "source_pk",
                    "v_astra",
                ),
                True,
            ),
        )    
