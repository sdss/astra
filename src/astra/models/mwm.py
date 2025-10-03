import datetime
import numpy as np
from astra import __version__
from astra.utils import log
from peewee import fn, DeferredForeignKey
from playhouse.hybrid import hybrid_property
from astra.fields import (
    ArrayField, AutoField, FloatField, BooleanField, DateTimeField, BigIntegerField, IntegerField, TextField,    
    ForeignKeyField, PixelArray, BitField, LogLambdaArrayAccessor
)
from astra.models.base import BaseModel
from astra.models.boss import BossVisitSpectrum
from astra.models.pipeline import PipelineOutputModel
from astra.models.source import Source
from astra.models.spectrum import Spectrum, SpectrumMixin
from astra.glossary import Glossary

from peewee import SQL, ForeignKeyField
from astra import __version__
from astra.fields import (AutoField, FloatField, TextField, BitField, DateTimeField, IntegerField)
from astra.models.base import BaseModel
from astra.utils import version_string_to_integer



class MWMStarMixin(BaseModel):
    
    @property
    def path(self):
        n = f"{self.sdss_id:0>4.0f}"
        sdss_id_groups = f"{n[-4:-2]}/{n[-2:]}"
        return (
            f"$MWM_ASTRA/{self.v_astra}/spectra/star/{sdss_id_groups}/mwmStar-{self.v_astra}-{self.sdss_id}.fits"
        )
        
class MWMVisitMixin(BaseModel):
    
    @property
    def path(self):
        n = f"{self.sdss_id:0>4.0f}"
        sdss_id_groups = f"{n[-4:-2]}/{n[-2:]}"
        return (
            f"$MWM_ASTRA/{self.v_astra}/spectra/visit/{sdss_id_groups}/mwmVisit-{self.v_astra}-{self.sdss_id}.fits"
        )        

get_boss_ext = lambda i: (dict(apo25m=1, lco25m=2)[i.telescope])
get_apogee_ext = lambda i: (dict(apo1m=3, apo25m=3, lco25m=4)[i.telescope])
transform_flat = lambda x, *_: np.array(x).flatten()



class MWMSpectrumProductStatus(BaseModel):

    source_pk = ForeignKeyField(Source, null=False, index=True, lazy_load=False, column_name="source_pk")
    
    #> Astra Metadata
    task_pk = AutoField()
    v_astra = IntegerField(default=version_string_to_integer(__version__))
    created = DateTimeField(default=datetime.datetime.now)
    modified = DateTimeField(default=datetime.datetime.now)
    t_elapsed = FloatField(null=True)
    t_overhead = FloatField(null=True)
    tag = TextField(default="", index=True)
    # The /1000 here is set in `astra.utils.version_string_to_integer` and `astra.utils.version_integer_to_string`.
    v_astra_major_minor = IntegerField(constraints=[SQL("GENERATED ALWAYS AS (v_astra / 1000) STORED")], _hidden=True)

    flags = BitField(default=0, help_text="Flags for the status of the spectrum products")
    flag_skipped_because_no_sdss_id = flags.flag(2**0, "Source was skipped because no SDSS ID exists")
    flag_skipped_because_not_stellar_like = flags.flag(2**1, "Source was skipped because it is not stellar-like")
    flag_attempted_but_exception = flags.flag(2**2, "An exception was raised during processing")
    flag_created_mwm_visit = flags.flag(2**3, "MWM visit spectrum was created")
    flag_created_mwm_star = flags.flag(2**4, "MWM star spectrum was created")
    
    class Meta:
        constraints = [
            SQL("UNIQUE (source_pk, v_astra_major_minor)")
        ]


class BossRestFrameVisitSpectrum(MWMVisitMixin, SpectrumMixin):

    pk = AutoField()

    #> Identifiers
    spectrum_pk = ForeignKeyField(
        Spectrum,
        null=True,
        index=True,
        unique=True,
        lazy_load=False,
        help_text=Glossary.spectrum_pk
    )
    source = ForeignKeyField(
        Source,
        null=True,
        index=True,
        column_name="source_pk",
        help_text=Glossary.source_pk,
        backref="boss_rest_frame_visit_spectra"
    )    
    drp_spectrum_pk = ForeignKeyField(
        BossVisitSpectrum,
        index=True,
        unique=True,
        lazy_load=False,
        field=BossVisitSpectrum.spectrum_pk,
        help_text=Glossary.drp_spectrum_pk,
        backref="resampled_spectrum"
    )    

    #> Data Product Keywords
    release = TextField(index=True, help_text=Glossary.release)
    filetype = TextField(default="mwmVisit", help_text=Glossary.filetype)
    v_astra = TextField(default=__version__, help_text=Glossary.v_astra)        
    healpix = IntegerField(null=True, help_text=Glossary.healpix)
    sdss_id = BigIntegerField(index=True, unique=False, null=False, help_text="SDSS-5 unique identifier")
    
    #> Related Data Product Keywords
    run2d = TextField(help_text=Glossary.run2d)
    mjd = IntegerField(help_text=Glossary.mjd)
    fieldid = IntegerField(help_text=Glossary.fieldid)
    catalogid = BigIntegerField(help_text=Glossary.catalogid)

    #> Exposure Information
    n_exp = IntegerField(null=True, help_text=Glossary.n_exp)    
    exptime = FloatField(null=True, help_text=Glossary.exptime)

    #> Field/Plate Information
    plateid = IntegerField(null=True, help_text=Glossary.plateid)
    cartid = IntegerField(null=True, help_text=Glossary.cartid)
    mapid = IntegerField(null=True, help_text=Glossary.mapid)
    slitid = IntegerField(null=True, help_text=Glossary.slitid)

    #> BOSS Data Reduction Pipeline
    psfsky = IntegerField(null=True, help_text=Glossary.psfsky)    
    preject = FloatField(null=True, help_text=Glossary.preject)    
    n_std = IntegerField(null=True, help_text=Glossary.n_std)
    n_gal = IntegerField(null=True, help_text=Glossary.n_gal)
    lowrej = IntegerField(null=True, help_text=Glossary.lowrej)    
    highrej = IntegerField(null=True, help_text=Glossary.highrej)    
    scatpoly = IntegerField(null=True, help_text=Glossary.scatpoly)    
    proftype = IntegerField(null=True, help_text=Glossary.proftype)    
    nfitpoly = IntegerField(null=True, help_text=Glossary.nfitpoly)    
    skychi2 = FloatField(null=True, help_text=Glossary.skychi2)    
    schi2min = FloatField(null=True, help_text=Glossary.schi2min)    
    schi2max = FloatField(null=True, help_text=Glossary.schi2max)

    #> Observing Conditions
    alt = FloatField(null=True, help_text=Glossary.alt)    
    az = FloatField(null=True, help_text=Glossary.az)    
    telescope = TextField(null=True, help_text=Glossary.telescope) # This is not a data product keyword, it is only stored in the headers.
    seeing = FloatField(null=True, help_text=Glossary.seeing)
    airmass = FloatField(null=True, help_text=Glossary.airmass)    
    airtemp = FloatField(null=True, help_text=Glossary.airtemp)    
    dewpoint = FloatField(null=True, help_text=Glossary.dewpoint)    
    humidity = FloatField(null=True, help_text=Glossary.humidity)    
    pressure = FloatField(null=True, help_text=Glossary.pressure)    
    dust_a = FloatField(null=True, help_text=Glossary.dust_a)
    dust_b = FloatField(null=True, help_text=Glossary.dust_b)    
    gust_direction = FloatField(null=True, help_text=Glossary.gust_direction)    
    gust_speed = FloatField(null=True, help_text=Glossary.gust_speed)    
    wind_direction = FloatField(null=True, help_text=Glossary.wind_direction)    
    wind_speed = FloatField(null=True, help_text=Glossary.wind_speed)    
    moon_dist_mean = FloatField(null=True, help_text=Glossary.moon_dist_mean)    
    moon_phase_mean = FloatField(null=True, help_text=Glossary.moon_phase_mean)    
    n_guide = IntegerField(null=True, help_text=Glossary.n_guide)    
    tai_beg = BigIntegerField(null=True, help_text=Glossary.tai_beg)    
    tai_end = BigIntegerField(null=True, help_text=Glossary.tai_end)       
    fiber_offset = BooleanField(null=True, help_text=Glossary.fiber_offset)
    f_night_time = FloatField(null=True, help_text=Glossary.f_night_time)

    try:
        delta_ra = ArrayField(FloatField, null=True, help_text=Glossary.delta_ra)    
        delta_dec = ArrayField(FloatField, null=True, help_text=Glossary.delta_dec) 
    except:
        log.warning(f"Cannot create delta_ra and delta_dec fields for {__name__}.")

    #> Metadata Flags
    snr = FloatField(null=True, help_text=Glossary.snr)
    in_stack = BooleanField(null=False, help_text=Glossary.in_stack)
    gri_gaia_transform_flags = BitField(default=0, help_text="Flags for provenance of ugriz photometry")
    zwarning_flags = BitField(default=0, help_text="BOSS DRP warning flags") 
    
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
    flag_u_gaia_transformed = gri_gaia_transform_flags.flag(2**0, "u photometry provided for plate design is transformed from Gaia mags.")
    flag_u_gaia_outside = gri_gaia_transform_flags.flag(2**1, "u photometry cannot be calculated from Gaia photometry as it lies outside range of colors for which transforms are defined")
    flag_u_gaia_none = gri_gaia_transform_flags.flag(2**2, "u photometry cannot be calculated for this source")
    flag_g_gaia_transformed = gri_gaia_transform_flags.flag(2**3, "g photometry provided for plate design is transformed from Gaia mags.")
    flag_g_gaia_outside = gri_gaia_transform_flags.flag(2**4, "g photometry cannot be calculated from Gaia photometry as it lies outside range of colors for which transforms are defined")
    flag_g_gaia_none = gri_gaia_transform_flags.flag(2**5, "g photometry cannot be calculated for this source")
    flag_r_gaia_transformed = gri_gaia_transform_flags.flag(2**6, "r photometry provided for plate design is transformed from Gaia mags.")
    flag_r_gaia_outside = gri_gaia_transform_flags.flag(2**7, "r photometry cannot be calculated from Gaia photometry as it lies outside range of colors for which transforms are defined")
    flag_r_gaia_none = gri_gaia_transform_flags.flag(2**8, "r photometry cannot be calculated for this source")
    flag_i_gaia_transformed = gri_gaia_transform_flags.flag(2**9, "i photometry provided for plate design is transformed from Gaia mags.")
    flag_i_gaia_outside = gri_gaia_transform_flags.flag(2**10, "i photometry cannot be calculated from Gaia photometry as it lies outside range of colors for which transforms are defined")
    flag_i_gaia_none = gri_gaia_transform_flags.flag(2**11, "i photometry cannot be calculated for this source")
    flag_z_gaia_transformed = gri_gaia_transform_flags.flag(2**12, "z photometry provided for plate design is transformed from Gaia mags.")
    flag_z_gaia_outside = gri_gaia_transform_flags.flag(2**13, "z photometry cannot be calculated from Gaia photometry as it lies outside range of colors for which transforms are defined")
    flag_z_gaia_none = gri_gaia_transform_flags.flag(2**14, "z photometry cannot be calculated for this source")
    flag_gaia_neighbor = gri_gaia_transform_flags.flag(2**15, "bright gaia neighbor close to the star")
    flag_g_panstarrs = gri_gaia_transform_flags.flag(2**16, "g photometry provided by PanSTARRS")
    flag_r_panstarrs = gri_gaia_transform_flags.flag(2**17, "r photometry provided by PanSTARRS")
    flag_i_panstarrs = gri_gaia_transform_flags.flag(2**18, "i photometry provided by PanSTARRS")
    flag_z_panstarrs = gri_gaia_transform_flags.flag(2**19, "z photometry provided by PanSTARRS")
    flag_position_offset = gri_gaia_transform_flags.flag(2**20, "position shifted with respect to center of star (catalogid corresponds to GAIA id)")

    # zwarning
    flag_sky_fiber = zwarning_flags.flag(2**0, "Sky fiber")
    flag_little_wavelength_coverage = zwarning_flags.flag(2**1, "Too little wavelength coverage (WCOVERAGE < 0.18)")
    flag_small_delta_chi2 = zwarning_flags.flag(2**2, "Chi-squared of best fit is too close to that of second best (< 0.01 in reduced chi-squared)")
    flag_negative_model = zwarning_flags.flag(2**3, "Synthetic spectrum is negative (only set for stars and QSOs)")
    flag_many_outliers = zwarning_flags.flag(2**4, "Fraction of points more than 5 sigma away from best model is too large (> 0.05)")
    flag_z_fit_limit = zwarning_flags.flag(2**5, "Chi-squared minimum at edge of the redshift fitting range (Z_ERR set to -1)")
    flag_negative_emission = zwarning_flags.flag(2**6, "A QSO line exhibits negative emission, triggered only in QSO spectra, if C_IV, C_III, Mg_II, H_beta, or H_alpha has LINEAREA + 3 * LINEAREA_ERR < 0")
    flag_unplugged = zwarning_flags.flag(2**7, "The fiber was unplugged or damaged, and the location of any spectrum is unknown")
    flag_bad_target = zwarning_flags.flag(2**8, "Catastrophically bad targeting data (e.g. bad astrometry)")
    flag_no_data = zwarning_flags.flag(2**9, "No data for this fiber, e.g. because spectrograph was broken during this exposure (ivar=0 for all pixels)")

    #> Spectral data
    wavelength = PixelArray(
        accessor_class=LogLambdaArrayAccessor,
        accessor_kwargs=dict(
            crval=3.5523,
            cdelt=1e-4,
            naxis=4648,
        ),
        help_text=Glossary.wavelength
    )    
    flux = PixelArray(ext=get_boss_ext, transform=transform_flat, help_text=Glossary.flux)
    ivar = PixelArray(ext=get_boss_ext, transform=transform_flat, help_text=Glossary.ivar)
    wresl = PixelArray(ext=get_boss_ext, transform=transform_flat, help_text=Glossary.wresl)
    pixel_flags = PixelArray(ext=get_boss_ext, transform=transform_flat, column_name="or_mask", help_text=Glossary.pixel_flags)

    #> NMF Continuum Model
    continuum = PixelArray(ext=get_boss_ext, transform=transform_flat, help_text=Glossary.continuum)
    nmf_rchi2 = FloatField(null=True, help_text=Glossary.nmf_rchi2)
    nmf_flags = BitField(default=0, help_text="NMF Continuum method flags")

    created = DateTimeField(default=datetime.datetime.now)
    modified = DateTimeField(default=datetime.datetime.now)

    @property
    def e_flux(self):
        with np.errstate(divide='ignore'):
            return self.ivar**-0.5

    class Meta:
        indexes = (
            (
                (
                    "v_astra",
                    "filetype",
                    "release",
                    "run2d",
                    "fieldid",
                    "mjd",
                    "catalogid"
                ), 
                True
            ),
        )


class BossCombinedSpectrum(MWMStarMixin, SpectrumMixin):
    
    pk = AutoField()
    
    #> Identifiers
    spectrum_pk = ForeignKeyField(
        Spectrum,
        null=True,
        index=True,
        unique=True,
        lazy_load=False,
        help_text=Glossary.spectrum_pk,
    )
    # Won't appear in a header group because it is first referenced in `Source`.
    source = ForeignKeyField(
        Source, 
        # We want to allow for spectra to be unassociated with a source so that 
        # we can test with fake spectra, etc, but any pipeline should run their
        # own checks to make sure that spectra and sources are linked.
        null=True, 
        index=True,
        column_name="source_pk",
        help_text=Glossary.source_pk,
        backref="boss_coadded_spectra",
    )    

    #> Data Product Keywords    
    release = TextField(index=True, help_text=Glossary.release)
    filetype = TextField(default="mwmStar", help_text=Glossary.filetype)
    v_astra = TextField(default=__version__, help_text=Glossary.v_astra)        
    healpix = IntegerField(null=True, help_text=Glossary.healpix)
    sdss_id = BigIntegerField(index=True, unique=False, null=False, help_text="SDSS-5 unique identifier")
    
    #> Related Data Product Keywords
    run2d = TextField(help_text=Glossary.run2d)  # Not strictly a data product keyword, but doesn't fit nicely elsewhere
    telescope = TextField(index=True, help_text=Glossary.telescope)
        
    #> Observing Span
    min_mjd = IntegerField(null=True, help_text="Minimum MJD of visits")
    max_mjd = IntegerField(null=True, help_text="Maximum MJD of visits")
    n_visits = IntegerField(null=True, help_text="Number of BOSS visits")
    n_good_visits = IntegerField(null=True, help_text="Number of 'good' BOSS visits")
    n_good_rvs = IntegerField(null=True, help_text="Number of 'good' BOSS radial velocities")

    #> Radial Velocity (XCSAO)
    v_rad = FloatField(null=True, help_text=Glossary.v_rad)
    e_v_rad = FloatField(null=True, help_text=Glossary.e_v_rad)
    std_v_rad = FloatField(null=True, help_text="Standard deviation of visit V_RAD [km/s]")
    median_e_v_rad = FloatField(null=True, help_text=Glossary.median_e_v_rad)    
    xcsao_teff = FloatField(null=True, help_text=Glossary.teff)
    xcsao_e_teff = FloatField(null=True, help_text=Glossary.e_teff)
    xcsao_logg = FloatField(null=True, help_text=Glossary.logg)
    xcsao_e_logg = FloatField(null=True, help_text=Glossary.e_logg)
    xcsao_fe_h = FloatField(null=True, help_text=Glossary.fe_h)
    xcsao_e_fe_h = FloatField(null=True, help_text=Glossary.e_fe_h)
    xcsao_meanrxc = FloatField(null=True, help_text="Cross-correlation R-value (1979AJ.....84.1511T)")

    #> Metadata
    snr = FloatField(null=True, help_text=Glossary.snr)
    gri_gaia_transform_flags = BitField(default=0, help_text="Flags for provenance of ugriz photometry")
    zwarning_flags = BitField(default=0, help_text="BOSS DRP warning flags") 
    # gri_gaia_transform
    flag_u_gaia_transformed = gri_gaia_transform_flags.flag(2**0, "u photometry provided for plate design is transformed from Gaia mags.")
    flag_u_gaia_outside = gri_gaia_transform_flags.flag(2**1, "u photometry cannot be calculated from Gaia photometry as it lies outside range of colors for which transforms are defined")
    flag_u_gaia_none = gri_gaia_transform_flags.flag(2**2, "u photometry cannot be calculated for this source")
    flag_g_gaia_transformed = gri_gaia_transform_flags.flag(2**3, "g photometry provided for plate design is transformed from Gaia mags.")
    flag_g_gaia_outside = gri_gaia_transform_flags.flag(2**4, "g photometry cannot be calculated from Gaia photometry as it lies outside range of colors for which transforms are defined")
    flag_g_gaia_none = gri_gaia_transform_flags.flag(2**5, "g photometry cannot be calculated for this source")
    flag_r_gaia_transformed = gri_gaia_transform_flags.flag(2**6, "r photometry provided for plate design is transformed from Gaia mags.")
    flag_r_gaia_outside = gri_gaia_transform_flags.flag(2**7, "r photometry cannot be calculated from Gaia photometry as it lies outside range of colors for which transforms are defined")
    flag_r_gaia_none = gri_gaia_transform_flags.flag(2**8, "r photometry cannot be calculated for this source")
    flag_i_gaia_transformed = gri_gaia_transform_flags.flag(2**9, "i photometry provided for plate design is transformed from Gaia mags.")
    flag_i_gaia_outside = gri_gaia_transform_flags.flag(2**10, "i photometry cannot be calculated from Gaia photometry as it lies outside range of colors for which transforms are defined")
    flag_i_gaia_none = gri_gaia_transform_flags.flag(2**11, "i photometry cannot be calculated for this source")
    flag_z_gaia_transformed = gri_gaia_transform_flags.flag(2**12, "z photometry provided for plate design is transformed from Gaia mags.")
    flag_z_gaia_outside = gri_gaia_transform_flags.flag(2**13, "z photometry cannot be calculated from Gaia photometry as it lies outside range of colors for which transforms are defined")
    flag_z_gaia_none = gri_gaia_transform_flags.flag(2**14, "z photometry cannot be calculated for this source")
    flag_gaia_neighbor = gri_gaia_transform_flags.flag(2**15, "bright gaia neighbor close to the star")
    flag_g_panstarrs = gri_gaia_transform_flags.flag(2**16, "g photometry provided by PanSTARRS")
    flag_r_panstarrs = gri_gaia_transform_flags.flag(2**17, "r photometry provided by PanSTARRS")
    flag_i_panstarrs = gri_gaia_transform_flags.flag(2**18, "i photometry provided by PanSTARRS")
    flag_z_panstarrs = gri_gaia_transform_flags.flag(2**19, "z photometry provided by PanSTARRS")
    flag_position_offset = gri_gaia_transform_flags.flag(2**20, "position shifted with respect to center of star (catalogid corresponds to GAIA id)")

    # zwarning
    flag_sky_fiber = zwarning_flags.flag(2**0, "Sky fiber")
    flag_little_wavelength_coverage = zwarning_flags.flag(2**1, "Too little wavelength coverage (WCOVERAGE < 0.18)")
    flag_small_delta_chi2 = zwarning_flags.flag(2**2, "Chi-squared of best fit is too close to that of second best (< 0.01 in reduced chi-squared)")
    flag_negative_model = zwarning_flags.flag(2**3, "Synthetic spectrum is negative (only set for stars and QSOs)")
    flag_many_outliers = zwarning_flags.flag(2**4, "Fraction of points more than 5 sigma away from best model is too large (> 0.05)")
    flag_z_fit_limit = zwarning_flags.flag(2**5, "Chi-squared minimum at edge of the redshift fitting range (Z_ERR set to -1)")
    flag_negative_emission = zwarning_flags.flag(2**6, "A QSO line exhibits negative emission, triggered only in QSO spectra, if C_IV, C_III, Mg_II, H_beta, or H_alpha has LINEAREA + 3 * LINEAREA_ERR < 0")
    flag_unplugged = zwarning_flags.flag(2**7, "The fiber was unplugged or damaged, and the location of any spectrum is unknown")
    flag_bad_target = zwarning_flags.flag(2**8, "Catastrophically bad targeting data (e.g. bad astrometry)")
    flag_no_data = zwarning_flags.flag(2**9, "No data for this fiber, e.g. because spectrograph was broken during this exposure (ivar=0 for all pixels)")
    
    #> Spectral Data
    wavelength = PixelArray(
        accessor_class=LogLambdaArrayAccessor,
        accessor_kwargs=dict(
            crval=3.5523,
            cdelt=1e-4,
            naxis=4648,
        ),
        help_text=Glossary.wavelength
    )    
    flux = PixelArray(ext=get_boss_ext, transform=transform_flat, help_text=Glossary.flux)
    ivar = PixelArray(ext=get_boss_ext, transform=transform_flat, help_text=Glossary.ivar)
    pixel_flags = PixelArray(ext=get_boss_ext, transform=transform_flat, help_text=Glossary.pixel_flags)        
    
    #> NMF Continuum Model
    continuum = PixelArray(ext=get_boss_ext, transform=transform_flat, help_text=Glossary.continuum)
    nmf_rectified_model_flux = PixelArray(ext=get_boss_ext, transform=transform_flat, help_text=Glossary.nmf_rectified_model_flux)
    nmf_rchi2 = FloatField(null=True, help_text=Glossary.nmf_rchi2)
    nmf_flags = BitField(default=0, help_text="NMF Continuum method flags")

    created = DateTimeField(default=datetime.datetime.now)
    modified = DateTimeField(default=datetime.datetime.now)

    class Meta:
        indexes = (
            (
                (
                    "release",
                    "filetype",
                    "v_astra",                    
                    "healpix",
                    "sdss_id",                    
                    "telescope",
                    "run2d"
                ),
                True,
            ),
        )
    

class ApogeeCombinedSpectrum(MWMStarMixin, SpectrumMixin):
    
    pk = AutoField()
    
    #> Identifiers
    spectrum_pk = ForeignKeyField(
        Spectrum,
        null=True,
        index=True,
        unique=True,
        lazy_load=False,
        column_name="spectrum_pk",
        help_text=Glossary.spectrum_pk,
    )
    # Won't appear in a header group because it is first referenced in `Source`.
    source = ForeignKeyField(
        Source, 
        # We want to allow for spectra to be unassociated with a source so that 
        # we can test with fake spectra, etc, but any pipeline should run their
        # own checks to make sure that spectra and sources are linked.
        null=True, 
        index=True,
        column_name="source_pk",
        help_text=Glossary.source_pk,
        backref="apogee_coadded_spectra",
    )
    
    #> Data Product Keywords
    release = TextField(index=True, help_text=Glossary.release)
    filetype = TextField(default="mwmStar", help_text=Glossary.filetype)
    v_astra = TextField(default=__version__, help_text=Glossary.v_astra)    
    healpix = IntegerField(help_text=Glossary.healpix) # This should be the same as the Source-level field.
    sdss_id = BigIntegerField(index=True, unique=False, null=True, help_text="SDSS-5 unique identifier")
    
    created = DateTimeField(default=datetime.datetime.now)
    modified = DateTimeField(default=datetime.datetime.now)

    #> Related Data Product Keywords        
    apred = TextField(help_text=Glossary.apred)
    obj = TextField(help_text=Glossary.obj)
    telescope = TextField(help_text=Glossary.telescope)    
            
    #> Observing Span
    min_mjd = IntegerField(null=True, help_text="Minimum MJD of visits")
    max_mjd = IntegerField(null=True, help_text="Maximum MJD of visits")

    #> Number and Quality of Visits
    n_entries = IntegerField(null=True, help_text="apStar entries for this SDSS4_APOGEE_ID") # Only present in DR17
    n_visits = IntegerField(null=True, help_text="Number of APOGEE visits")
    n_good_visits = IntegerField(null=True, help_text="Number of 'good' APOGEE visits")
    n_good_rvs = IntegerField(null=True, help_text="Number of 'good' APOGEE radial velocities")

    #> Summary Statistics
    snr = FloatField(null=True, help_text=Glossary.snr)
    mean_fiber = FloatField(null=True, help_text="S/N-weighted mean visit fiber number")
    std_fiber = FloatField(null=True, help_text="Standard deviation of visit fiber numbers")
    spectrum_flags = BitField(default=0, help_text=Glossary.spectrum_flags)

    #> Radial Velocity (Doppler)
    v_rad = FloatField(null=True, help_text=Glossary.v_rad)
    e_v_rad = FloatField(null=True, help_text=Glossary.e_v_rad)
    std_v_rad = FloatField(null=True, help_text="Standard deviation of visit V_RAD [km/s]")
    median_e_v_rad = FloatField(null=True, help_text=Glossary.median_e_v_rad) # Only in SDSS5
    
    doppler_teff = FloatField(null=True, help_text=Glossary.teff)
    doppler_e_teff = FloatField(null=True, help_text=Glossary.e_teff)
    doppler_logg = FloatField(null=True, help_text=Glossary.logg)
    doppler_e_logg = FloatField(null=True, help_text=Glossary.e_logg)
    doppler_fe_h = FloatField(null=True, help_text=Glossary.fe_h)
    doppler_e_fe_h = FloatField(null=True, help_text=Glossary.e_fe_h)
    doppler_rchi2 = FloatField(null=True, help_text="Reduced chi-square value of DOPPLER fit")
    doppler_flags = BitField(default=0, help_text="DOPPLER flags") 

    #> Radial Velocity (X-Correlation)
    xcorr_v_rad = FloatField(null=True, help_text=Glossary.v_rad)
    xcorr_v_rel = FloatField(null=True, help_text=Glossary.v_rel)
    xcorr_e_v_rel = FloatField(null=True, help_text=Glossary.e_v_rel)
    ccfwhm = FloatField(null=True, help_text=Glossary.ccfwhm)
    autofwhm = FloatField(null=True, help_text=Glossary.autofwhm)
    n_components = IntegerField(null=True, help_text=Glossary.n_components)    

    ##> Provenance
    #input_spectrum_pks = ArrayField(IntegerField, null=True, help_text="DRP visit PKs")

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
    flux = PixelArray(ext=get_apogee_ext, transform=transform_flat, help_text=Glossary.flux)
    ivar = PixelArray(ext=get_apogee_ext, transform=transform_flat, help_text=Glossary.ivar)
    pixel_flags = PixelArray(
        ext=get_apogee_ext, 
        transform=lambda x, *_: np.array(x, dtype=np.uint64).flatten(), 
        help_text=Glossary.pixel_flags
    )
    
    #> NMF Continuum Model
    continuum = PixelArray(ext=get_apogee_ext, transform=transform_flat, help_text=Glossary.continuum)
    nmf_rectified_model_flux = PixelArray(ext=get_apogee_ext, transform=transform_flat, help_text=Glossary.nmf_rectified_model_flux)
    nmf_rchi2 = FloatField(null=True, help_text=Glossary.nmf_rchi2)
    nmf_flags = BitField(default=0, help_text="NMF Continuum method flags")

    created = DateTimeField(default=datetime.datetime.now)
    modified = DateTimeField(default=datetime.datetime.now)


    class Meta:
        indexes = (
            (
                (
                    "release",
                    "filetype",
                    "v_astra",                    
                    "healpix",
                    "sdss_id",                    
                    "telescope",
                    "apred"
                ),
                True,
            ),
        )
    


class ApogeeRestFrameVisitSpectrum(MWMVisitMixin, SpectrumMixin):

    """An APOGEE rest-frame visit spectrum."""

    pk = AutoField()
    wavelength = PixelArray(
        accessor_class=LogLambdaArrayAccessor,
        accessor_kwargs=dict(
            crval=4.179,
            cdelt=6e-6,
            naxis=8575,
        ),
        help_text=Glossary.wavelength
    )
        
    #> Identifiers
    spectrum_pk = ForeignKeyField(
        Spectrum,
        null=True,
        index=True,
        unique=True,
        lazy_load=False,
        help_text=Glossary.spectrum_pk,
        column_name="spectrum_pk"
    )
    # Won't appear in a header group because it is first referenced in `Source`.
    source = ForeignKeyField(
        Source, 
        # We want to allow for spectra to be unassociated with a source so that 
        # we can test with fake spectra, etc, but any pipeline should run their
        # own checks to make sure that spectra and sources are linked.
        null=True, 
        index=True,
        column_name="source_pk",
        help_text=Glossary.source_pk,
        backref="apogee_visit_spectra",
    )
    
    catalogid = BigIntegerField(index=True, null=True, help_text="SDSS input catalog identifier")
    star_pk = BigIntegerField(null=True, unique=False, help_text="APOGEE DRP `star` primary key") # note: unique false
    visit_pk = BigIntegerField(null=True, unique=False, help_text="APOGEE DRP `visit` primary key")
    rv_visit_pk = BigIntegerField(null=True, unique=False, help_text="APOGEE DRP `rv_visit` primary key")

    #> Data Product Keywords
    release = TextField(index=True, help_text=Glossary.release)
    filetype = TextField(default="mwmVisit", help_text=Glossary.filetype)    
    v_astra = TextField(default=__version__, help_text=Glossary.v_astra)    
    healpix = IntegerField(help_text=Glossary.healpix) # This should be the same as the Source-level field.
    sdss_id = BigIntegerField(index=True, unique=False, null=False, help_text="SDSS-5 unique identifier")

    #> Upstream Data Product Keywords
    apred = TextField(index=True, help_text=Glossary.apred)
    plate = TextField(index=True, help_text=Glossary.plate) # most are integers, but not all!
    telescope = TextField(index=True, help_text=Glossary.telescope)
    fiber = IntegerField(index=True, help_text=Glossary.fiber)
    mjd = IntegerField(index=True, help_text=Glossary.mjd)
    field = TextField(index=True, help_text=Glossary.field)
    prefix = TextField(help_text=Glossary.prefix)
    reduction = TextField(default="", help_text=Glossary.reduction) # only used for DR17 apo1m spectra
    obj = TextField(null=True, help_text=Glossary.obj)
    
    #> Observing Conditions
    date_obs = DateTimeField(null=True, help_text=Glossary.date_obs)
    jd = FloatField(null=True, help_text=Glossary.jd)
    exptime = FloatField(null=True, help_text=Glossary.exptime)
    dithered = BooleanField(null=True, help_text=Glossary.dithered)
    f_night_time = FloatField(null=True, help_text=Glossary.f_night_time)
    
    #> Telescope Pointing
    input_ra = FloatField(null=True, help_text=Glossary.input_ra)
    input_dec = FloatField(null=True, help_text=Glossary.input_dec)
    n_frames = IntegerField(null=True, help_text=Glossary.n_frames)
    assigned = IntegerField(null=True, help_text=Glossary.assigned)
    on_target = IntegerField(null=True, help_text=Glossary.on_target)
    valid = IntegerField(null=True, help_text=Glossary.valid)
    fps = BooleanField(null=True, help_text=Glossary.fps)
    
    #> Statistics and Spectrum Quality 
    snr = FloatField(null=True, help_text=Glossary.snr)
    in_stack = BooleanField(null=False, help_text=Glossary.in_stack)
    spectrum_flags = BitField(default=0, help_text=Glossary.spectrum_flags)
    
    # From https://github.com/sdss/apogee_drp/blob/630d3d45ecff840d49cf75ac2e8a31e22b543838/python/apogee_drp/utils/bitmask.py#L110
    # and https://github.com/sdss/apogee/blob/e134409dc14b20f69e68a0d4d34b2c1b5056a901/python/apogee/utils/bitmask.py#L9
    flag_bad_pixels = spectrum_flags.flag(2**0, help_text="Spectrum has many bad pixels (>20%).")
    flag_commissioning = spectrum_flags.flag(2**1, help_text="Commissioning data (MJD <55761); non-standard configuration; poor LSF.")
    flag_bright_neighbor = spectrum_flags.flag(2**2, help_text="Star has neighbor more than 10 times brighter.")
    flag_very_bright_neighbor = spectrum_flags.flag(2**3, help_text="Star has neighbor more than 100 times brighter.")
    flag_low_snr = spectrum_flags.flag(2**4, help_text="Spectrum has low S/N (<5).")
    # 4-8 inclusive are not defined
    flag_persist_high = spectrum_flags.flag(2**9, help_text="Spectrum has at least 20% of pixels in high persistence region.")
    flag_persist_med = spectrum_flags.flag(2**10, help_text="Spectrum has at least 20% of pixels in medium persistence region.")
    flag_persist_low = spectrum_flags.flag(2**11, help_text="Spectrum has at least 20% of pixels in low persistence region.")
    flag_persist_jump_pos = spectrum_flags.flag(2**12, help_text="Spectrum has obvious positive jump in blue chip.")
    flag_persist_jump_neg = spectrum_flags.flag(2**13, help_text="Spectrum has obvious negative jump in blue chip.")
    # 14-15 inclusive are not defined
    flag_suspect_rv_combination = spectrum_flags.flag(2**16, help_text="RVs from synthetic template differ significantly (~2 km/s) from those from combined template.")
    flag_suspect_broad_lines = spectrum_flags.flag(2**17, help_text="Cross-correlation peak with template significantly broader than autocorrelation of template.")
    flag_bad_rv_combination = spectrum_flags.flag(2**18, help_text="RVs from synthetic template differ very significantly (~10 km/s) from those from combined template.")
    flag_rv_reject = spectrum_flags.flag(2**19, help_text="Rejected visit because cross-correlation RV differs significantly from least squares RV.")
    flag_rv_suspect = spectrum_flags.flag(2**20, help_text="Suspect visit (but used!) because cross-correlation RV differs slightly from least squares RV.")
    flag_multiple_suspect = spectrum_flags.flag(2**21, help_text="Suspect multiple components from Gaussian decomposition of cross-correlation.")
    flag_rv_failure = spectrum_flags.flag(2**22, help_text="RV failure.")
    flag_suspect_rotation = spectrum_flags.flag(2**23, help_text="Suspect rotation: cross-correlation peak with template significantly broader than autocorretion of template")
    flag_mtpflux_lt_75 = spectrum_flags.flag(2**24, help_text="Spectrum falls on fiber in MTP block with relative flux < 0.75")
    flag_mtpflux_lt_50 = spectrum_flags.flag(2**25, help_text="Spectrum falls on fiber in MTP block with relative flux < 0.5")
    
    #> Radial Velocity (Doppler)
    v_rad = FloatField(null=True, help_text=Glossary.v_rad)
    v_rel = FloatField(null=True, help_text=Glossary.v_rel)
    e_v_rel = FloatField(null=True, help_text=Glossary.e_v_rel)
    bc = FloatField(null=True, help_text=Glossary.bc)
    
    doppler_teff = FloatField(null=True, help_text=Glossary.teff)
    doppler_e_teff = FloatField(null=True, help_text=Glossary.e_teff)
    doppler_logg = FloatField(null=True, help_text=Glossary.logg)
    doppler_e_logg = FloatField(null=True, help_text=Glossary.e_logg)
    doppler_fe_h = FloatField(null=True, help_text=Glossary.fe_h)
    doppler_e_fe_h = FloatField(null=True, help_text=Glossary.e_fe_h)
    doppler_rchi2 = FloatField(null=True, help_text="Reduced chi-square value of DOPPLER fit")
    doppler_flags = BitField(default=0, help_text="DOPPLER flags") 
    
    #> Radial Velocity (X-Correlation)
    xcorr_v_rad = FloatField(null=True, help_text=Glossary.v_rad)
    xcorr_v_rel = FloatField(null=True, help_text=Glossary.v_rel)
    xcorr_e_v_rel = FloatField(null=True, help_text=Glossary.e_v_rel)
    ccfwhm = FloatField(null=True, help_text=Glossary.ccfwhm)
    autofwhm = FloatField(null=True, help_text=Glossary.autofwhm)
    n_components = IntegerField(null=True, help_text=Glossary.n_components)    

    #> Spectral Data
    flux = PixelArray(ext=get_apogee_ext, help_text=Glossary.flux)
    ivar = PixelArray(ext=get_apogee_ext, help_text=Glossary.ivar)
    pixel_flags = PixelArray(ext=get_apogee_ext, help_text=Glossary.pixel_flags)        
    
    #> NMF Continuum Model
    continuum = PixelArray(ext=get_apogee_ext, help_text=Glossary.continuum)
    nmf_rchi2 = FloatField(null=True, help_text=Glossary.nmf_rchi2)
    nmf_flags = BitField(default=0, help_text="NMF Continuum method flags")
    
    created = DateTimeField(default=datetime.datetime.now)
    modified = DateTimeField(default=datetime.datetime.now)

    @hybrid_property
    def flag_bad(self):
        return (
            self.flag_bad_pixels
        |   self.flag_very_bright_neighbor
        |   self.flag_bad_rv_combination
        |   self.flag_rv_failure
        )

    @hybrid_property
    def flag_warn(self):
        return (self.spectrum_flags > 0)

    class Meta:
        indexes = (
            (
                (
                    "release",
                    "v_astra",
                    "sdss_id",
                    "apred",
                    "mjd",
                    "plate",
                    "telescope",
                    "field",
                    "fiber",
                    "prefix",
                    "reduction",
                ),
                True,
            ),
        )
    
