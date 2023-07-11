from peewee import (
    FloatField,
    BooleanField,
    DateTimeField,
    BigIntegerField,
    IntegerField,
    TextField,
    ForeignKeyField,
)
from astra.models.base import BaseModel
from astra.models.source import Source
from astra.models.spectrum import (Spectrum, SpectrumMixin)
from astra.models.glossary import Glossary
from astra.models.fields import PixelArray, BitField

xcsao_glossary = Glossary("XCSAO")


class BossVisitSpectrum(BaseModel, SpectrumMixin):

    """A BOSS visit spectrum, where a visit is defined by spectra taken on a single MJD."""

    sdss_id = ForeignKeyField(Source, index=True, backref="boss_visit_spectra")

    #> Spectrum identifier
    spectrum_id = ForeignKeyField(
        Spectrum, 
        index=True, 
        lazy_load=False,
    )

    #> Spectral data
    wavelength = PixelArray(
        ext=1, 
        column_name="loglam", 
        transform=lambda x: 10**x,
    )
    flux = PixelArray(ext=1)
    ivar = PixelArray(ext=1)
    wresl = PixelArray(ext=1) # TODO: No help text yet!
    pixel_flags = PixelArray(ext=1, column_name="or_mask")

    #> Data Product Keywords
    release = TextField()        
    run2d = TextField()        
    mjd = IntegerField()        
    fieldid = IntegerField()        
    catalogid = BigIntegerField()        

    #> Software Version Information
    v_boss = TextField(null=True)    
    v_jaeger = TextField(null=True)    
    v_kaiju = TextField(null=True)    
    v_coord = TextField(null=True)    
    v_calibs = TextField(null=True)    
    v_idl = TextField(null=True)    
    v_util = TextField(null=True)    
    v_read = TextField(null=True)    
    v_2d = TextField(null=True)    
    v_comb = TextField(null=True)    
    v_log = TextField(null=True)    
    v_flat = TextField(null=True)  

    #> BOSS Data Reduction Pipeline
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

    #> Observing Conditions
    telescope = TextField(null=True) # This is not a data product keyword, it is only stored in the headers.
    alt = FloatField(null=True)    
    az = FloatField(null=True)    
    seeing = FloatField(null=True)    
    airmass = FloatField(null=True)    
    airtemp = FloatField(null=True)    
    dewpoint = FloatField(null=True)    
    humidity = FloatField(null=True)    
    pressure = FloatField(null=True)    
    gust_direction = FloatField(null=True)    
    gust_speed = FloatField(null=True)    
    wind_direction = FloatField(null=True)    
    wind_speed = FloatField(null=True)    
    moon_dist_mean = FloatField(null=True)    
    moon_phase_mean = FloatField(null=True)    
    n_exp = IntegerField(null=True)    
    n_guide = IntegerField(null=True)    
    tai_beg = DateTimeField(null=True)    
    tai_end = DateTimeField(null=True)    
    fiber_offset = BooleanField(null=True)    
    delta_ra = FloatField(null=True)    
    delta_dec = FloatField(null=True) 

    #> Metadata Flags
    gri_gaia_transform = BitField(default=0, help_text="Flags to track provenance of ugriz photometry") # TODO: should these be _flags?
    zwarning = BitField(default=0) # TODO: rename to _flags?

    #> XCSAO
    #v_rad = FloatField()
    #e_v_rad = FloatField()
    xcsao_teff = FloatField(null=True, help_text=xcsao_glossary.teff)
    xcsao_e_teff = FloatField(null=True, help_text=xcsao_glossary.e_teff)
    xcsao_logg = FloatField(null=True, help_text=xcsao_glossary.logg)
    xcsao_e_logg = FloatField(null=True, help_text=xcsao_glossary.e_logg)
    xcsao_fe_h = FloatField(null=True, help_text=xcsao_glossary.fe_h)
    xcsao_e_fe_h = FloatField(null=True, help_text=xcsao_glossary.e_fe_h)
    xcsao_rxc = FloatField(null=True, help_text="Cross-correlation R-value (1979AJ.....84.1511T)")

    # gri_gaia_transform
    flag_u_gaia_transformed = gri_gaia_transform.flag(2**0, "u photometry provided for plate design is transformed from Gaia mags.")
    flag_u_gaia_outside = gri_gaia_transform.flag(2**1, "u photometry cannot be calculated from Gaia photometry as it lies outside range of colors for which transforms are defined")
    flag_u_gaia_none = gri_gaia_transform.flag(2**2, "u photometry cannot be calculated for this source")
    flag_g_gaia_transformed = gri_gaia_transform.flag(2**3, "g photometry provided for plate design is transformed from Gaia mags.")
    flag_g_gaia_outside = gri_gaia_transform.flag(2**4, "g photometry cannot be calculated from Gaia photometry as it lies outside range of colors for which transforms are defined")
    flag_g_gaia_none = gri_gaia_transform.flag(2**5, "g photometry cannot be calculated for this source")
    flag_r_gaia_transformed = gri_gaia_transform.flag(2**6, "r photometry provided for plate design is transformed from Gaia mags.")
    flag_r_gaia_outside = gri_gaia_transform.flag(2**7, "r photometry cannot be calculated from Gaia photometry as it lies outside range of colors for which transforms are defined")
    flag_r_gaia_none = gri_gaia_transform.flag(2**8, "r photometry cannot be calculated for this source")
    flag_i_gaia_transformed = gri_gaia_transform.flag(2**9, "i photometry provided for plate design is transformed from Gaia mags.")
    flag_i_gaia_outside = gri_gaia_transform.flag(2**10, "i photometry cannot be calculated from Gaia photometry as it lies outside range of colors for which transforms are defined")
    flag_i_gaia_none = gri_gaia_transform.flag(2**11, "i photometry cannot be calculated for this source")
    flag_z_gaia_transformed = gri_gaia_transform.flag(2**12, "z photometry provided for plate design is transformed from Gaia mags.")
    flag_z_gaia_outside = gri_gaia_transform.flag(2**13, "z photometry cannot be calculated from Gaia photometry as it lies outside range of colors for which transforms are defined")
    flag_z_gaia_none = gri_gaia_transform.flag(2**14, "z photometry cannot be calculated for this source")
    flag_gaia_neighbor = gri_gaia_transform.flag(2**15, "bright gaia neighbor close to the star")
    flag_g_panstarrs = gri_gaia_transform.flag(2**16, "g photometry provided by PanSTARRS")
    flag_r_panstarrs = gri_gaia_transform.flag(2**17, "r photometry provided by PanSTARRS")
    flag_i_panstarrs = gri_gaia_transform.flag(2**18, "i photometry provided by PanSTARRS")
    flag_z_panstarrs = gri_gaia_transform.flag(2**19, "z photometry provided by PanSTARRS")
    flag_position_offset = gri_gaia_transform.flag(2**20, "position shifted with respect to center of star (catalogid corresponds to GAIA id)")

    # zwarning
    flag_sky_fiber = zwarning.flag(2**0, "Sky fiber")
    flag_little_wavelength_coverage = zwarning.flag(2**1, "Too little wavelength coverage (WCOVERAGE < 0.18)")
    flag_small_delta_chi2 = zwarning.flag(2**2, "Chi-squared of best fit is too close to that of second best (< 0.01 in reduced chi-squared)")
    flag_negative_model = zwarning.flag(2**3, "Synthetic spectrum is negative (only set for stars and QSOs)")
    flag_many_outliers = zwarning.flag(2**4, "Fraction of points more than 5 sigma away from best model is too large (> 0.05)")
    flag_z_fit_limit = zwarning.flag(2**5, "Chi-squared minimum at edge of the redshift fitting range (Z_ERR set to -1)")
    flag_negative_emission = zwarning.flag(2**6, "A QSO line exhibits negative emission, triggered only in QSO spectra, if C_IV, C_III, Mg_II, H_beta, or H_alpha has LINEAREA + 3 * LINEAREA_ERR < 0")
    flag_unplugged = zwarning.flag(2**7, "The fiber was unplugged or damaged, and the location of any spectrum is unknown")
    flag_bad_target = zwarning.flag(2**8, "Catastrophically bad targeting data (e.g. bad astrometry)")
    flag_no_data = zwarning.flag(2**9, "No data for this fiber, e.g. because spectrograph was broken during this exposure (ivar=0 for all pixels)")

    class Meta:
        indexes = ((("release", "run2d", "fieldid", "mjd", "catalogid"), True),)

    @property
    def path(self):
        return (
            f"$SAS_BASE_DIR/"
            f"sdsswork/bhm/boss/spectro/redux/"
            f"{self.run2d}/spectra/full/{self.pad_fieldid}{self.isplate}/{self.mjd}/"
            f"spec-{self.pad_fieldid}-{self.mjd}-{self.catalogid}.fits"
        )

    @property
    def pad_fieldid(self):
        if (not self.run2d) & (not self.fieldid):
            return ""
        if self.run2d in ['v6_0_1','v6_0_2', 'v6_0_3', 'v6_0_4']:
            return str(self.fieldid)
        return str(self.fieldid).zfill(6)
    
    @property
    def isplate(self):
        if not self.run2d:
            return ''
        if self.run2d in ['v6_0_1','v6_0_2', 'v6_0_3', 'v6_0_4']:
            return 'p'
        return ''
