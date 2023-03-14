
from astra.database.astradb import BaseModel
from peewee import AutoField, BigIntegerField, IntegerField, TextField, FloatField, ForeignKeyField, BooleanField, DateTimeField
from playhouse.postgres_ext import ArrayField

class StarMeta(BaseModel):
    pk = AutoField()
    
    astra_version_major = IntegerField()
    astra_version_minor = IntegerField()
    astra_version_patch = IntegerField()

    created = DateTimeField()

    healpix = IntegerField()
    gaia_source_id = BigIntegerField(null=True)
    gaia_data_release = TextField(null=True)

    cat_id = BigIntegerField()
    tic_id = BigIntegerField(null=True)
    cat_id05 = BigIntegerField(null=True)
    cat_id10 = BigIntegerField(null=True)

    ra = FloatField()
    dec = FloatField()
    gaia_ra = FloatField(null=True)
    gaia_dec = FloatField(null=True)
    plx = FloatField(null=True)
    pmra = FloatField(null=True)
    pmde = FloatField(null=True)
    e_pmra = FloatField(null=True)
    e_pmde = FloatField(null=True)
    gaia_v_rad = FloatField(null=True)
    gaia_e_v_rad = FloatField(null=True)
    g_mag = FloatField(null=True)
    bp_mag = FloatField(null=True)
    rp_mag = FloatField(null=True)
    j_mag = FloatField(null=True)
    h_mag = FloatField(null=True)
    k_mag = FloatField(null=True)
    e_j_mag = FloatField(null=True)
    e_h_mag = FloatField(null=True)
    e_k_mag = FloatField(null=True)
    
    carton_0 = TextField(null=True)
    v_xmatch = TextField()

    # Doppler results.
    doppler_teff = FloatField(null=True)
    doppler_e_teff = FloatField(null=True)
    doppler_logg = FloatField(null=True)
    doppler_e_logg = FloatField(null=True)
    doppler_fe_h = FloatField(null=True)
    doppler_e_fe_h = FloatField(null=True)
    doppler_starflag = IntegerField(null=True)
    doppler_version = TextField(null=True)
    doppler_v_rad = FloatField(null=True)
    
    # The RXC SAO results are done per visit, not per star.
    # For convenience we include them here, but we will take
    # The result with the highest S/N.
    
    xcsao_teff = FloatField(null=True)
    xcsao_e_teff = FloatField(null=True)
    xcsao_logg = FloatField(null=True)
    xcsao_e_logg = FloatField(null=True)
    # TODO: Naming of this in files is feh
    xcsao_fe_h = FloatField(null=True)
    xcsao_e_fe_h = FloatField(null=True)            
    xcsao_rxc = FloatField(null=True)
    xcsao_v_rad = FloatField(null=True)
    xcsao_e_v_rad = FloatField(null=True)




class VisitMeta(BaseModel):
    pk = AutoField()
    
    astra_version_major = IntegerField()
    astra_version_minor = IntegerField()
    astra_version_patch = IntegerField()

    created = DateTimeField()

    healpix = IntegerField()
    gaia_source_id = BigIntegerField(null=True)
    gaia_data_release = TextField(null=True)

    cat_id = BigIntegerField()
    tic_id = BigIntegerField(null=True)
    cat_id05 = BigIntegerField(null=True)
    cat_id10 = BigIntegerField(null=True)

    ra = FloatField()
    dec = FloatField()
    gaia_ra = FloatField(null=True)
    gaia_dec = FloatField(null=True)
    plx = FloatField(null=True)
    pmra = FloatField(null=True)
    pmde = FloatField(null=True)
    e_pmra = FloatField(null=True)
    e_pmde = FloatField(null=True)
    gaia_v_rad = FloatField(null=True)
    gaia_e_v_rad = FloatField(null=True)
    g_mag = FloatField(null=True)
    bp_mag = FloatField(null=True)
    rp_mag = FloatField(null=True)
    j_mag = FloatField(null=True)
    h_mag = FloatField(null=True)
    k_mag = FloatField(null=True)
    e_j_mag = FloatField(null=True)
    e_h_mag = FloatField(null=True)
    e_k_mag = FloatField(null=True)
    
    carton_0 = TextField(null=True)
    v_xmatch = TextField()


    # File type stuff
    release = TextField()
    filetype = TextField()
    plate = IntegerField(null=True)
    fiber = IntegerField(null=True)
    field = TextField(null=True)
    apred = TextField(null=True)
    prefix = TextField(null=True)
    mjd = IntegerField(null=True)            

    run2d = TextField(null=True)
    fieldid = TextField(null=True)
    isplate = TextField(null=True)
    catalogid = BigIntegerField(null=True)

    # Common stuff.
    observatory = TextField()
    instrument = TextField()
    hdu_data_index = IntegerField()
    snr = FloatField()
    fps = FloatField()
    in_stack = BooleanField()
    v_shift = FloatField(null=True)

    continuum_theta = ArrayField(FloatField)

    # APOGEE-level stuff.
    v_apred = TextField(null=True)
    nres = ArrayField(FloatField)
    filtsize = IntegerField()
    normsize = IntegerField()
    conscale = BooleanField()

    # Doppler results.
    doppler_teff = FloatField(null=True)
    doppler_e_teff = FloatField(null=True)
    doppler_logg = FloatField(null=True)
    doppler_e_logg = FloatField(null=True)
    doppler_fe_h = FloatField(null=True)
    doppler_e_fe_h = FloatField(null=True)
    doppler_starflag = IntegerField(null=True)
    doppler_version = TextField(null=True)         
    
    date_obs = DateTimeField(null=True)
    exptime = FloatField(null=True)
    fluxflam = FloatField(null=True)
    npairs = IntegerField(null=True)
    dithered = FloatField(null=True)

    jd = FloatField(null=True)
    v_rad = FloatField(null=True)
    e_v_rad = FloatField(null=True)
    v_rel = FloatField(null=True)
    v_bc = FloatField(null=True)
    rchisq = FloatField(null=True)
    n_rv_components = IntegerField(null=True)

    visit_pk = BigIntegerField(null=True)
    rv_visit_pk = BigIntegerField(null=True)

    v_boss = TextField(null=True)
    vjaeger = TextField(null=True)
    vkaiju = TextField(null=True)
    vcoordio = TextField(null=True)
    vcalibs = TextField(null=True)
    versidl = TextField(null=True)
    versutil = TextField(null=True)
    versread = TextField(null=True)
    vers2d = TextField(null=True)
    verscomb = TextField(null=True)
    verslog = TextField(null=True)
    versflat = TextField(null=True)
    didflush = BooleanField(null=True)
    cartid = TextField(null=True)
    psfsky = IntegerField(null=True)
    preject = FloatField(null=True)
    lowrej = IntegerField(null=True)
    highrej = IntegerField(null=True)
    scatpoly = IntegerField(null=True)
    proftype = IntegerField(null=True)
    nfitpoly = IntegerField(null=True)
    skychi2 = FloatField(null=True)
    schi2min = FloatField(null=True)
    schi2max = FloatField(null=True)
    rdnoise0 = FloatField(null=True)

    alt = FloatField(null=True)
    az = FloatField(null=True)
    seeing = FloatField(null=True)
    airmass = FloatField(null=True)
    airtemp = FloatField(null=True)
    dewpoint = FloatField(null=True)
    humidity = FloatField(null=True)
    pressure = FloatField(null=True)
    gustd = FloatField(null=True)
    gusts = FloatField(null=True)
    windd = FloatField(null=True)
    winds = FloatField(null=True)
    moon_dist_mean = FloatField(null=True)
    moon_phase_mean = FloatField(null=True)
    nexp = IntegerField(null=True)
    nguide = IntegerField(null=True)
    tai_beg = DateTimeField(null=True)
    tai_end = DateTimeField(null=True)
    fiber_offset = BooleanField(null=True)
    delta_ra = FloatField(null=True)
    delta_dec = FloatField(null=True)
    zwarning = IntegerField(null=True)

    xcsao_teff = FloatField(null=True)
    xcsao_e_teff = FloatField(null=True)
    xcsao_logg = FloatField(null=True)
    xcsao_e_logg = FloatField(null=True)
    xcsao_fe_h = FloatField(null=True)
    xcsao_e_fe_h = FloatField(null=True)            
    xcsao_rxc = FloatField(null=True)