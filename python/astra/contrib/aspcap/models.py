from peewee import BitField, FloatField, IntegerField, BooleanField

from astra.database.astradb import SDSSOutput

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