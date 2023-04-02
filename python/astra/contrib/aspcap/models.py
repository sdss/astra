import datetime
from numpy import nan
from peewee import BitField, FloatField, IntegerField, BooleanField, TextField, ForeignKeyField, DateTimeField

from astra.database.astradb import (
    Task, BaseModel, SDSSOutput, DataProduct, Source,
    _version_major, _version_minor, _version_patch,
    _get_sdss_metadata
)

class ASPCAPOutput(BaseModel):
    
    class Meta:
        table_name = "aspcap"
        
    task = ForeignKeyField(
        Task, 
        default=Task.create,
        on_delete="CASCADE", 
        primary_key=True
    )

    completed = DateTimeField(default=datetime.datetime.now)

    version_major = IntegerField(default=_version_major)
    version_minor = IntegerField(default=_version_minor)
    version_patch = IntegerField(default=_version_patch)

    data_product = ForeignKeyField(DataProduct, null=True)
    source = ForeignKeyField(Source, null=True)    
    hdu = IntegerField()

    snr = FloatField(null=True)
    obj = TextField(default="",null=True)
    #mjd = FloatField(null=True)
    plate = TextField(default="",null=True)
    field = TextField(default="",null=True)
    telescope = TextField(null=True)
    instrument = TextField(null=True)
    #fiber = IntegerField(default=-1,null=True)
    #apvisit_pk = IntegerField(null=True) # set foreign relational key to apvisit table
    apstar_pk = IntegerField(default=-1, null=True) # set foreign relational key to apstar table

    #output_data_product = ForeignKeyField(DataProduct, null=True)
    grid = TextField()

    bitmask_aspcap = BitField(default=0)
    warnflag = BooleanField(default=False)
    badflag = BooleanField(default=False)

    teff = FloatField()
    e_teff = FloatField()
    bitmask_teff = BitField(default=0)

    logg = FloatField()
    e_logg = FloatField()
    bitmask_logg = BitField(default=0)

    m_h = FloatField()
    e_m_h = FloatField()
    bitmask_m_h = BitField(default=0)

    v_sini = FloatField(default=nan, null=True)
    e_v_sini = FloatField(default=nan, null=True)
    bitmask_v_sini = BitField(default=0)

    v_micro = FloatField(default=nan, null=True)
    e_v_micro = FloatField(default=nan, null=True)
    bitmask_v_micro = BitField(default=0)

    c_m_atm = FloatField(default=nan, null=True)
    e_c_m_atm = FloatField(default=nan, null=True)
    bitmask_c_m_atm = BitField(default=0)

    n_m_atm = FloatField(default=nan, null=True)
    e_n_m_atm = FloatField(default=nan, null=True)
    bitmask_n_m_atm = BitField(default=0)

    alpha_m_atm = FloatField(default=nan, null=True)
    e_alpha_m_atm = FloatField(default=nan, null=True)
    bitmask_alpha_m_atm = BitField(default=0)

    # Summary statistics for stellar parameter fits.
    chisq = FloatField()

    # elemental abundances.
    al_h = FloatField(default=nan, null=True)
    e_al_h = FloatField(default=nan, null=True)
    bitmask_al_h = BitField(default=0)
    chisq_al_h = FloatField(default=nan, null=True)

    c13_h = FloatField(default=nan, null=True)
    e_c13_h = FloatField(default=nan, null=True)
    bitmask_c13_h = BitField(default=0)
    chisq_c13_h = FloatField(default=nan, null=True)

    # TODO: Initial values used?
    # TODO: covariances?

    ca_h = FloatField(default=nan, null=True)
    e_ca_h = FloatField(default=nan, null=True)
    bitmask_ca_h = BitField(default=0)
    chisq_ca_h = FloatField(default=nan, null=True)

    ce_h = FloatField(default=nan, null=True)
    e_ce_h = FloatField(default=nan, null=True)
    bitmask_ce_h = BitField(default=0)
    chisq_ce_h = FloatField(default=nan, null=True)

    c1_h = FloatField(default=nan, null=True)
    e_c1_h = FloatField(default=nan, null=True)
    bitmask_c1_h = BitField(default=0)
    chisq_c1_h = FloatField(default=nan, null=True)

    c_h = FloatField(default=nan, null=True)
    e_c_h = FloatField(default=nan, null=True)
    bitmask_c_h = BitField(default=0)
    chisq_c_h = FloatField(default=nan, null=True)

    co_h = FloatField(default=nan, null=True)
    e_co_h = FloatField(default=nan, null=True)
    bitmask_co_h = BitField(default=0)
    chisq_co_h = FloatField(default=nan, null=True)

    cr_h = FloatField(default=nan, null=True)
    e_cr_h = FloatField(default=nan, null=True)
    bitmask_cr_h = BitField(default=0)
    chisq_cr_h = FloatField(default=nan, null=True)

    cu_h = FloatField(default=nan, null=True)
    e_cu_h = FloatField(default=nan, null=True)
    bitmask_cu_h = BitField(default=0)
    chisq_cu_h = FloatField(default=nan, null=True)

    fe_h = FloatField(default=nan, null=True)
    e_fe_h = FloatField(default=nan, null=True)
    bitmask_fe_h = BitField(default=0)
    chisq_fe_h = FloatField(default=nan, null=True)

    k_h = FloatField(default=nan, null=True)
    e_k_h = FloatField(default=nan, null=True)
    bitmask_k_h = BitField(default=0)
    chisq_k_h = FloatField(default=nan, null=True)

    mg_h = FloatField(default=nan, null=True)
    e_mg_h = FloatField(default=nan, null=True)
    bitmask_mg_h = BitField(default=0)
    chisq_mg_h = FloatField(default=nan, null=True)

    mn_h = FloatField(default=nan, null=True)
    e_mn_h = FloatField(default=nan, null=True)
    bitmask_mn_h = BitField(default=0)
    chisq_mn_h = FloatField(default=nan, null=True)

    na_h = FloatField(default=nan, null=True)
    e_na_h = FloatField(default=nan, null=True)
    bitmask_na_h = BitField(default=0)
    chisq_na_h = FloatField(default=nan, null=True)

    nd_h = FloatField(default=nan, null=True)
    e_nd_h = FloatField(default=nan, null=True)
    bitmask_nd_h = BitField(default=0)
    chisq_nd_h = FloatField(default=nan, null=True)

    ni_h = FloatField(default=nan, null=True)
    e_ni_h = FloatField(default=nan, null=True)
    bitmask_ni_h = BitField(default=0)
    chisq_ni_h = FloatField(default=nan, null=True)

    n_h = FloatField(default=nan, null=True)
    e_n_h = FloatField(default=nan, null=True)
    bitmask_n_h = BitField(default=0)
    chisq_n_h = FloatField(default=nan, null=True)

    o_h = FloatField(default=nan, null=True)
    e_o_h = FloatField(default=nan, null=True)
    bitmask_o_h = BitField(default=0)
    chisq_o_h = FloatField(default=nan, null=True)

    p_h = FloatField(default=nan, null=True)
    e_p_h = FloatField(default=nan, null=True)
    bitmask_p_h = BitField(default=0)
    chisq_p_h = FloatField(default=nan, null=True)

    si_h = FloatField(default=nan, null=True)
    e_si_h = FloatField(default=nan, null=True)
    bitmask_si_h = BitField(default=0)
    chisq_si_h = FloatField(default=nan, null=True)

    s_h = FloatField(default=nan, null=True)
    e_s_h = FloatField(default=nan, null=True)
    bitmask_s_h = BitField(default=0)
    chisq_s_h = FloatField(default=nan, null=True)

    ti_h = FloatField(default=nan, null=True)
    e_ti_h = FloatField(default=nan, null=True)
    bitmask_ti_h = BitField(default=0)
    chisq_ti_h = FloatField(default=nan, null=True)

    ti2_h = FloatField(default=nan, null=True)
    e_ti2_h = FloatField(default=nan, null=True)
    bitmask_ti2_h = BitField(default=0)
    chisq_ti2_h = FloatField(default=nan, null=True)

    v_h = FloatField(default=nan, null=True)
    e_v_h = FloatField(default=nan, null=True)
    bitmask_v_h = BitField(default=0)
    chisq_v_h = FloatField(default=nan, null=True)




class ASPCAPInitial(SDSSOutput):
    
    teff = FloatField()
    logg = FloatField()
    metals = FloatField()
    lgvsini = FloatField(null=True)
    # BA grid doesn't use these:
    log10vdop = FloatField(null=True)
    o_mg_si_s_ca_ti = FloatField(null=True)
    c = FloatField(null=True)
    n = FloatField(null=True)

    e_teff = FloatField()
    e_logg = FloatField()
    e_metals = FloatField()
    e_log10vdop = FloatField(null=True)
    e_lgvsini = FloatField(null=True)
    e_o_mg_si_s_ca_ti = FloatField(null=True)
    e_c = FloatField(null=True)
    e_n = FloatField(null=True)

    bitmask_teff = BitField(default=0)
    bitmask_logg = BitField(default=0)
    bitmask_metals = BitField(default=0)
    bitmask_log10vdop = BitField(default=0)
    bitmask_lgvsini = BitField(default=0)
    bitmask_o_mg_si_s_ca_ti = BitField(default=0)
    bitmask_c = BitField(default=0)
    bitmask_n = BitField(default=0)

    # TODO: read these in from some bitmask definition
    is_near_teff_edge = bitmask_teff.flag(256)
    is_near_logg_edge = bitmask_logg.flag(256)


    log_chisq_fit = FloatField()
    log_snr_sq = FloatField()
    frac_phot_data_points = FloatField(default=0)

    # This penalized log chisq term is strictly a term defined and used by ASPCAP
    # and not FERRE, but it is easier to understand what is happening when selecting
    # the `best` model if we have a penalized \chisq term.
    penalized_log_chisq_fit = FloatField(null=True)

    # Astra records the time taken *per task*, and infers things like overhead time for each stage
    # of pre_execute, execute, and post_execute.
    # But even one task with a single data model could contain many spectra that we analyse with
    # FERRE, and for performance purposes we want to know the time taken by FERRE.
    # For these reasons, let's store some metadata here, even if we could infer it from other things.
    ferre_time_elapsed = FloatField(null=True)
    ferre_time_load = FloatField(null=True)
    ferre_n_obj = IntegerField(null=True)
    ferre_timeout = BooleanField(null=True)


class ASPCAPStellarParameters(SDSSOutput):
    
    teff = FloatField()
    logg = FloatField()
    metals = FloatField()
    lgvsini = FloatField(null=True)
    # BA grid doesn't use these:
    log10vdop = FloatField(null=True)
    o_mg_si_s_ca_ti = FloatField(null=True)
    c = FloatField(null=True)
    n = FloatField(null=True)

    e_teff = FloatField()
    e_logg = FloatField()
    e_metals = FloatField()
    e_log10vdop = FloatField(null=True)
    e_lgvsini = FloatField(null=True)
    e_o_mg_si_s_ca_ti = FloatField(null=True)
    e_c = FloatField(null=True)
    e_n = FloatField(null=True)

    bitmask_teff = BitField(default=0)
    bitmask_logg = BitField(default=0)
    bitmask_metals = BitField(default=0)
    bitmask_log10vdop = BitField(default=0)
    bitmask_lgvsini = BitField(default=0)
    bitmask_o_mg_si_s_ca_ti = BitField(default=0)
    bitmask_c = BitField(default=0)
    bitmask_n = BitField(default=0)

    # TODO: read these in from some bitmask definition
    is_near_teff_edge = bitmask_teff.flag(256)
    is_near_logg_edge = bitmask_logg.flag(256)


    log_chisq_fit = FloatField()
    log_snr_sq = FloatField()
    frac_phot_data_points = FloatField(default=0)

    # This penalized log chisq term is strictly a term defined and used by ASPCAP
    # and not FERRE, but it is easier to understand what is happening when selecting
    # the `best` model if we have a penalized \chisq term.
    penalized_log_chisq_fit = FloatField(null=True)

    # Astra records the time taken *per task*, and infers things like overhead time for each stage
    # of pre_execute, execute, and post_execute.
    # But even one task with a single data model could contain many spectra that we analyse with
    # FERRE, and for performance purposes we want to know the time taken by FERRE.
    # For these reasons, let's store some metadata here, even if we could infer it from other things.
    ferre_time_elapsed = FloatField(null=True)
    ferre_time_load = FloatField(null=True)
    ferre_n_obj = IntegerField(null=True)
    ferre_timeout = BooleanField(null=True)


    

class ASPCAPAbundances(SDSSOutput):

    teff = FloatField()
    logg = FloatField()
    metals = FloatField()
    lgvsini = FloatField(null=True)
    # BA grid doesn't use these:
    log10vdop = FloatField(null=True)
    o_mg_si_s_ca_ti = FloatField(null=True)
    c = FloatField(null=True)
    n = FloatField(null=True)

    e_teff = FloatField()
    e_logg = FloatField()
    e_metals = FloatField()
    e_log10vdop = FloatField(null=True)
    e_lgvsini = FloatField(null=True)
    e_o_mg_si_s_ca_ti = FloatField(null=True)
    e_c = FloatField(null=True)
    e_n = FloatField(null=True)

    bitmask_teff = BitField(default=0)
    bitmask_logg = BitField(default=0)
    bitmask_metals = BitField(default=0)
    bitmask_log10vdop = BitField(default=0)
    bitmask_lgvsini = BitField(default=0)
    bitmask_o_mg_si_s_ca_ti = BitField(default=0)
    bitmask_c = BitField(default=0)
    bitmask_n = BitField(default=0)

    # TODO: read these in from some bitmask definition
    is_near_teff_edge = bitmask_teff.flag(256)
    is_near_logg_edge = bitmask_logg.flag(256)


    log_chisq_fit = FloatField()
    log_snr_sq = FloatField()
    frac_phot_data_points = FloatField(default=0)

    # This penalized log chisq term is strictly a term defined and used by ASPCAP
    # and not FERRE, but it is easier to understand what is happening when selecting
    # the `best` model if we have a penalized \chisq term.
    penalized_log_chisq_fit = FloatField(null=True)

    # Astra records the time taken *per task*, and infers things like overhead time for each stage
    # of pre_execute, execute, and post_execute.
    # But even one task with a single data model could contain many spectra that we analyse with
    # FERRE, and for performance purposes we want to know the time taken by FERRE.
    # For these reasons, let's store some metadata here, even if we could infer it from other things.
    ferre_time_elapsed = FloatField(null=True)
    ferre_time_load = FloatField(null=True)
    ferre_n_obj = IntegerField(null=True)
    ferre_timeout = BooleanField(null=True)    

    '''
    def __init__(self, data_product, spectrum=None, **kwargs):
        
        try:
            kwds = _get_sdss_metadata(data_product, spectrum, **kwargs)
        except:
            if spectrum is not None:
                print(f"Unable to get metadata for spectrum in data product {data_product} and {spectrum}")
            kwds = kwargs
        else:
            # Inject metadata
            kwds.update(kwargs)        
        try:
            kwds["source_id"] = data_product.source_id
        except:
            print("No source info added with {data_product} {spectrum}")
            None
        
        if "task" not in kwds and "task_id" not in kwds:
            print("OK adding TASK just in time")
            kwds["task"] = Task()

        super(BaseModel, self).__init__(data_product=data_product, **kwds)
        return None    
    '''