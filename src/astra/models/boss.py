import datetime
from astra import __version__
from astra.models.base import BaseModel
from astra.models.source import Source
from astra.models.spectrum import (Spectrum, SpectrumMixin)
from astra.fields import (
    ArrayField, AutoField, FloatField, BooleanField, DateTimeField, BigIntegerField, IntegerField, TextField,    
    ForeignKeyField, PixelArray, BitField, PickledPixelArrayAccessor, LogLambdaArrayAccessor
)

from astra.glossary import Glossary
class BossVisitSpectrum(BaseModel, SpectrumMixin):

    """A BOSS visit spectrum, where a visit is defined by spectra taken on the same MJD."""

    pk = AutoField()

    #> Identifiers
    spectrum_pk = ForeignKeyField(
        Spectrum,
        null=True,
        index=True,
        unique=True,
        lazy_load=False,
        column_name="spectrum_pk"
    )
    source = ForeignKeyField(
        Source,
        null=True,
        index=True,
        column_name="source_pk",
        backref="boss_visit_spectra"
    )    

    created = DateTimeField(default=datetime.datetime.now)
    modified = DateTimeField(default=datetime.datetime.now)

    #> Spectral data
    wavelength = PixelArray(
        ext=1, 
        column_name="loglam", 
        transform=lambda v, *a: 10**v,
    )
    flux = PixelArray(ext=1)
    ivar = PixelArray(ext=1)
    wresl = PixelArray(ext=1)
    pixel_flags = PixelArray(ext=1, column_name="or_mask")

    @property
    def e_flux(self):
        raise NotImplementedError("TODO: use common ivar_to_error function")
        return self.ivar**-0.5

    #> Data Product Keywords
    release = TextField()
    filetype = TextField(default="specFull")
    run2d = TextField()
    mjd = IntegerField()
    fieldid = IntegerField()
    catalogid = BigIntegerField()
    healpix = IntegerField()

    #> Exposure Information
    n_exp = IntegerField(null=True)
    exptime = FloatField(null=True)

    #> Field/Plate Information
    plateid = IntegerField(null=True)
    cartid = IntegerField(null=True)
    mapid = IntegerField(null=True)
    slitid = IntegerField(null=True)

    #> BOSS Data Reduction Pipeline
    psfsky = IntegerField(null=True)
    preject = FloatField(null=True)
    n_std = IntegerField(null=True)
    n_gal = IntegerField(null=True)
    lowrej = IntegerField(null=True)
    highrej = IntegerField(null=True)
    scatpoly = IntegerField(null=True)
    proftype = IntegerField(null=True)
    nfitpoly = IntegerField(null=True)
    skychi2 = FloatField(null=True)
    schi2min = FloatField(null=True)
    schi2max = FloatField(null=True)

    #> Observing Conditions
    alt = FloatField(null=True)
    az = FloatField(null=True)
    telescope = TextField(null=True)
    seeing = FloatField(null=True)
    airmass = FloatField(null=True)
    airtemp = FloatField(null=True)
    dewpoint = FloatField(null=True)
    humidity = FloatField(null=True)
    pressure = FloatField(null=True)
    dust_a = FloatField(null=True)
    dust_b = FloatField(null=True)
    gust_direction = FloatField(null=True)
    gust_speed = FloatField(null=True)
    wind_direction = FloatField(null=True)
    wind_speed = FloatField(null=True)
    moon_dist_mean = FloatField(null=True)
    moon_phase_mean = FloatField(null=True)
    n_guide = IntegerField(null=True)
    tai_beg = BigIntegerField(null=True)
    tai_end = BigIntegerField(null=True)
    fiber_offset = BooleanField(null=True)
    f_night_time = FloatField(null=True)

    delta_ra = ArrayField(FloatField, null=True)
    delta_dec = ArrayField(FloatField, null=True)

    #> Metadata Flags
    snr = FloatField(null=True)
    gri_gaia_transform_flags = BitField(default=0)
    zwarning_flags = BitField(default=0)
    
    #> XCSAO
    xcsao_v_rad = FloatField(null=True)
    xcsao_e_v_rad = FloatField(null=True)
    xcsao_teff = FloatField(null=True)
    xcsao_e_teff = FloatField(null=True)
    xcsao_logg = FloatField(null=True)
    xcsao_e_logg = FloatField(null=True)
    xcsao_fe_h = FloatField(null=True)
    xcsao_e_fe_h = FloatField(null=True)
    xcsao_rxc = FloatField(null=True)

    # gri_gaia_transform
    flag_u_gaia_transformed = gri_gaia_transform_flags.flag(2**0, help_text="u photometry provided for plate design is transformed from Gaia mags.")
    flag_u_gaia_outside = gri_gaia_transform_flags.flag(2**1, help_text="u photometry cannot be calculated from Gaia photometry as it lies outside range of colors for which transforms are defined")
    flag_u_gaia_none = gri_gaia_transform_flags.flag(2**2, help_text="u photometry cannot be calculated for this source")
    flag_g_gaia_transformed = gri_gaia_transform_flags.flag(2**3, help_text="g photometry provided for plate design is transformed from Gaia mags.")
    flag_g_gaia_outside = gri_gaia_transform_flags.flag(2**4, help_text="g photometry cannot be calculated from Gaia photometry as it lies outside range of colors for which transforms are defined")
    flag_g_gaia_none = gri_gaia_transform_flags.flag(2**5, help_text="g photometry cannot be calculated for this source")
    flag_r_gaia_transformed = gri_gaia_transform_flags.flag(2**6, help_text="r photometry provided for plate design is transformed from Gaia mags.")
    flag_r_gaia_outside = gri_gaia_transform_flags.flag(2**7, help_text="r photometry cannot be calculated from Gaia photometry as it lies outside range of colors for which transforms are defined")
    flag_r_gaia_none = gri_gaia_transform_flags.flag(2**8, help_text="r photometry cannot be calculated for this source")
    flag_i_gaia_transformed = gri_gaia_transform_flags.flag(2**9, help_text="i photometry provided for plate design is transformed from Gaia mags.")
    flag_i_gaia_outside = gri_gaia_transform_flags.flag(2**10, help_text="i photometry cannot be calculated from Gaia photometry as it lies outside range of colors for which transforms are defined")
    flag_i_gaia_none = gri_gaia_transform_flags.flag(2**11, help_text="i photometry cannot be calculated for this source")
    flag_z_gaia_transformed = gri_gaia_transform_flags.flag(2**12, help_text="z photometry provided for plate design is transformed from Gaia mags.")
    flag_z_gaia_outside = gri_gaia_transform_flags.flag(2**13, help_text="z photometry cannot be calculated from Gaia photometry as it lies outside range of colors for which transforms are defined")
    flag_z_gaia_none = gri_gaia_transform_flags.flag(2**14, help_text="z photometry cannot be calculated for this source")
    flag_gaia_neighbor = gri_gaia_transform_flags.flag(2**15, help_text="bright gaia neighbor close to the star")
    flag_g_panstarrs = gri_gaia_transform_flags.flag(2**16, help_text="g photometry provided by PanSTARRS")
    flag_r_panstarrs = gri_gaia_transform_flags.flag(2**17, help_text="r photometry provided by PanSTARRS")
    flag_i_panstarrs = gri_gaia_transform_flags.flag(2**18, help_text="i photometry provided by PanSTARRS")
    flag_z_panstarrs = gri_gaia_transform_flags.flag(2**19, help_text="z photometry provided by PanSTARRS")
    flag_position_offset = gri_gaia_transform_flags.flag(2**20, help_text="position shifted with respect to center of star (catalogid corresponds to GAIA id)")

    # zwarning
    flag_sky_fiber = zwarning_flags.flag(2**0, help_text="Sky fiber")
    flag_little_wavelength_coverage = zwarning_flags.flag(2**1, help_text="Too little wavelength coverage (WCOVERAGE < 0.18)")
    flag_small_delta_chi2 = zwarning_flags.flag(2**2, help_text="Chi-squared of best fit is too close to that of second best (< 0.01 in reduced chi-squared)")
    flag_negative_model = zwarning_flags.flag(2**3, help_text="Synthetic spectrum is negative (only set for stars and QSOs)")
    flag_many_outliers = zwarning_flags.flag(2**4, help_text="Fraction of points more than 5 sigma away from best model is too large (> 0.05)")
    flag_z_fit_limit = zwarning_flags.flag(2**5, help_text="Chi-squared minimum at edge of the redshift fitting range (Z_ERR set to -1)")
    flag_negative_emission = zwarning_flags.flag(2**6, help_text="A QSO line exhibits negative emission, triggered only in QSO spectra, if C_IV, C_III, Mg_II, H_beta, or H_alpha has LINEAREA + 3 * LINEAREA_ERR < 0")
    flag_unplugged = zwarning_flags.flag(2**7, help_text="The fiber was unplugged or damaged, and the location of any spectrum is unknown")
    flag_bad_target = zwarning_flags.flag(2**8, help_text="Catastrophically bad targeting data (e.g. bad astrometry)")
    flag_no_data = zwarning_flags.flag(2**9, help_text="No data for this fiber, e.g. because spectrograph was broken during this exposure (ivar=0 for all pixels)")

    class Meta:
        indexes = (
            (("release", "run2d", "fieldid", "mjd", "catalogid"), True),
            # The folloing index makes it easier to count the number of unique spectra per source (over different reduction versions)
            (("source_pk", "telescope", "mjd", "fieldid", "plateid"), False),
        )

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
