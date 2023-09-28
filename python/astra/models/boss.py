from peewee import (
    AutoField,
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
from astra.models.fields import PixelArray, BitField

from astra.glossary import Glossary


class BossVisitSpectrum(BaseModel, SpectrumMixin):

    """A BOSS visit spectrum, where a visit is defined by spectra taken on a single MJD."""

    PIXELS = 4648

    pk = AutoField()

    source = ForeignKeyField(
        Source,
        null=True,
        index=True,
        column_name="source_pk",
        backref="boss_visit_spectra"
    )

    #> Spectrum identifier
    spectrum_pk = ForeignKeyField(
        Spectrum,
        index=True, 
        unique=True,
        lazy_load=False,
        help_text=Glossary.spectrum_pk,
    )

    #> Spectral data
    wavelength = PixelArray(
        ext=1, 
        column_name="loglam", 
        transform=lambda v, *a: 10**v,
        pixels=PIXELS,
        help_text=Glossary.wavelength
    )
    flux = PixelArray(ext=1, pixels=PIXELS, help_text=Glossary.flux)
    ivar = PixelArray(ext=1, pixels=PIXELS, help_text=Glossary.ivar)
    wresl = PixelArray(ext=1, pixels=PIXELS, help_text=Glossary.wresl)
    pixel_flags = PixelArray(ext=1, column_name="or_mask", pixels=PIXELS, help_text=Glossary.pixel_flags)

    #> Data Product Keywords
    release = TextField(help_text=Glossary.release)
    filetype = TextField(default="specFull", help_text=Glossary.filetype)
    run2d = TextField(help_text=Glossary.run2d)
    mjd = IntegerField(help_text=Glossary.mjd)
    fieldid = IntegerField(help_text=Glossary.fieldid)
    catalogid = BigIntegerField(help_text=Glossary.catalogid)

    #> Software Version Information
    v_boss = TextField(null=True, help_text=Glossary.v_boss)    
    v_jaeger = TextField(null=True, help_text=Glossary.v_jaeger)    
    v_kaiju = TextField(null=True, help_text=Glossary.v_kaiju)    
    v_coord = TextField(null=True, help_text=Glossary.v_coord)    
    v_calibs = TextField(null=True, help_text=Glossary.v_calibs)    
    v_idl = TextField(null=True, help_text=Glossary.v_idl)    
    v_util = TextField(null=True, help_text=Glossary.v_util)    
    v_read = TextField(null=True, help_text=Glossary.v_read)    
    v_2d = TextField(null=True, help_text=Glossary.v_2d)    
    v_comb = TextField(null=True, help_text=Glossary.v_comb)    
    v_log = TextField(null=True, help_text=Glossary.v_log)    
    v_flat = TextField(null=True, help_text=Glossary.v_flat)  

    #> BOSS Data Reduction Pipeline
    didflush = BooleanField(null=True, help_text=Glossary.didflush)    
    cartid = TextField(null=True, help_text=Glossary.cartid)    
    psfsky = IntegerField(null=True, help_text=Glossary.psfsky)    
    preject = FloatField(null=True, help_text=Glossary.preject)    
    lowrej = IntegerField(null=True, help_text=Glossary.lowrej)    
    highrej = IntegerField(null=True, help_text=Glossary.highrej)    
    scatpoly = IntegerField(null=True, help_text=Glossary.scatpoly)    
    proftype = IntegerField(null=True, help_text=Glossary.proftype)    
    nfitpoly = IntegerField(null=True, help_text=Glossary.nfitpoly)    
    skychi2 = FloatField(null=True, help_text=Glossary.skychi2)    
    schi2min = FloatField(null=True, help_text=Glossary.schi2min)    
    schi2max = FloatField(null=True, help_text=Glossary.schi2max)    
    rdnoise0 = FloatField(null=True, help_text=Glossary.rdnoise0)    

    #> Observing Conditions
    telescope = TextField(null=True, help_text=Glossary.telescope) # This is not a data product keyword, it is only stored in the headers.
    alt = FloatField(null=True, help_text=Glossary.alt)    
    az = FloatField(null=True, help_text=Glossary.az)    
    seeing = FloatField(null=True, help_text=Glossary.seeing)    
    airmass = FloatField(null=True, help_text=Glossary.airmass)    
    airtemp = FloatField(null=True, help_text=Glossary.airtemp)    
    dewpoint = FloatField(null=True, help_text=Glossary.dewpoint)    
    humidity = FloatField(null=True, help_text=Glossary.humidity)    
    pressure = FloatField(null=True, help_text=Glossary.pressure)    
    gust_direction = FloatField(null=True, help_text=Glossary.gust_direction)    
    gust_speed = FloatField(null=True, help_text=Glossary.gust_speed)    
    wind_direction = FloatField(null=True, help_text=Glossary.wind_direction)    
    wind_speed = FloatField(null=True, help_text=Glossary.wind_speed)    
    moon_dist_mean = FloatField(null=True, help_text=Glossary.moon_dist_mean)    
    moon_phase_mean = FloatField(null=True, help_text=Glossary.moon_phase_mean)    
    n_exp = IntegerField(null=True, help_text=Glossary.n_exp)    
    n_guide = IntegerField(null=True, help_text=Glossary.n_guide)    
    tai_beg = DateTimeField(null=True, help_text=Glossary.tai_beg)    
    tai_end = DateTimeField(null=True, help_text=Glossary.tai_end)    
    fiber_offset = BooleanField(null=True, help_text=Glossary.fiber_offset)    
    delta_ra = FloatField(null=True, help_text=Glossary.delta_ra)    
    delta_dec = FloatField(null=True, help_text=Glossary.delta_dec) 

    #> Metadata Flags
    gri_gaia_transform = BitField(default=0, help_text="Flags to track provenance of ugriz photometry") # TODO: should these be _flags?
    zwarning = BitField(default=0, help_text="BOSS DRP warning flags") # TODO: rename to _flags?

    #> XCSAO
    xcsao_v_rad = FloatField(null=True, help_text=Glossary.v_rad)
    xcsao_e_v_rad = FloatField(null=True, help_text=Glossary.e_v_rad)
    xcsao_teff = FloatField(null=True, help_text=Glossary.teff)
    xcsao_e_teff = FloatField(null=True, help_text=Glossary.e_teff)
    xcsao_logg = FloatField(null=True, help_text=Glossary.logg)
    xcsao_e_logg = FloatField(null=True, help_text=Glossary.e_logg)
    xcsao_fe_h = FloatField(null=True, help_text=Glossary.fe_h)
    xcsao_e_fe_h = FloatField(null=True, help_text=Glossary.e_fe_h)
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
