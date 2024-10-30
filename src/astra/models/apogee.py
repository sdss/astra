import os
import numpy as np
from peewee import DeferredForeignKey, fn
from playhouse.hybrid import hybrid_property
from astra.fields import (
    AutoField, FloatField, BooleanField, DateTimeField, BigIntegerField, IntegerField, TextField,
    ForeignKeyField, PixelArray, BitField, LogLambdaArrayAccessor
)
from astra.models.base import BaseModel
from astra.models.spectrum import (Spectrum, SpectrumMixin)
from astra.models.source import Source

from astropy.constants import c
from astropy import units as u

C_KM_S = c.to(u.km / u.s).value



def _transform_err_to_ivar(err, *args, **kwargs):
    ivar = np.atleast_2d(err)[0]**-2
    ivar[~np.isfinite(ivar)] = 0
    return ivar


# TODO: Move these to a common place in astra.fields

def transform(v, image, instance):
    # Accessor class for the PixelArrays

    path_template = ApogeeVisitSpectrum.get_path_template(instance.release, instance.telescope)

    kwds = instance.__data__.copy()
    # TODO: Evaluate whether we still need this.
    # If `reduction` is defined and filled in the derivative ApogeeVisit* products, then we don't need this any more.
    try:
        kwds.setdefault("reduction", instance.obj)
    except:
        None

    expected_path = os.path.basename(path_template).format(**kwds)

    v = np.atleast_2d(v)
    N, P = v.shape
    for i in range(1, 1 + N):
        try:
            if (image[0].header[f"SFILE{i}"] == expected_path):
                break
        except:
            None
    else:
        raise ValueError(f"Cannot find {expected_path} in {image}")
    
    i -= 1 # put it back to 0-index 
    # offset for stacks
    if N > 2:
        i += 2
    return v[i]
    


class ApogeeVisitSpectrum(BaseModel, SpectrumMixin):

    """An APOGEE visit spectrum, stored in an apVisit data product."""

    pk = AutoField()
    
    # A decision was made here.

    # I want the `spectrum_pk` to be unique across tables. I don't want to run
    # a single INSERT statement to create `spectrum_pk` each time, because that
    # is the limiting speed factor when we have bulk inserts of spectra. If I
    # pre-allocate the `spectrum_pk` values then ends up having big gaps, and I
    # later have to remove and set the sequence value. If I allow `spectrum_pk`
    # to be null then that introduces risky behaviour for the user (since they
    # would need to 'remember' to create a `spectrum_pk` after), and it muddles
    # the primary key, since we'd need an `AutoField()` for primary key, which
    # is different from `spectrum_pk`. And not all of these solutions are
    # consistent between PostgreSQL and SQLite.

    # I'm going to provide a default function of `Spectrum.create`, and then
    # pre-assign all of these in bulk when we do bulk inserts. I wish there
    # was a better way to avoid calling `Spectrum.create` every time, and still
    # enforce a constraint so that the user doesn't have to handle this 
    # themselves, but I think getting that constraint to work well on SQLite
    # and PostgreSQL is hard. 
    
    # Note that this explicitly breaks one of the 'Lessons learned from IPL-2'!

    #> Identifiers
    spectrum_pk = ForeignKeyField(
        Spectrum,
        null=True,
        index=True,
        unique=True,
        lazy_load=False,
#        )
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
        backref="apogee_visit_spectra",
    )
        
    catalogid = BigIntegerField(index=True, null=True)
    star_pk = BigIntegerField(null=True, unique=False) # Note: unique = False
    visit_pk = BigIntegerField(null=True, unique=True)
    rv_visit_pk = BigIntegerField(null=True, unique=True)

    #> Data Product Keywords
    release = TextField(index=True)
    filetype = TextField(default="apVisit")
    apred = TextField(index=True)
    plate = TextField(index=True)
    telescope = TextField(index=True)
    fiber = IntegerField(index=True)
    mjd = IntegerField(index=True)
    field = TextField(index=True)
    prefix = TextField()
    reduction = TextField(default="")
    # Note that above I use `default=''` instead of `null=True` because SQLite
    # seems to have some weird bug with using `null=True` on a key field. It
    # meant that `reduction` was being set to null even when I gave it values.

    # obj is not strictly a data product keyword for apVisit files, but it is
    # needed to find the equivalent visit spectrum stored in the apStar file.
    # You might think that `obj` is equal to `sdss_dr17_apogee_id`, or you
    # might think that this is a source-level attribute and not a spectrum-
    # level attribute, and *either* of those scenarios would be sufficient to
    # motivate us NOT putting `obj` here and instead storing it at the source-
    # level, but the `obj` value is actually calculated PER SPECTRUM based on
    # the INPUT ra and dec, which is why we are storing it here. The next
    # closest attribute, healpix, is a per-source attribute, but this is
    # *correctly* calculated from the CATALOG position (ra, dec) and not the
    # input position to the telescope.
    obj = TextField(null=True)

    #> Spectral Data
    wavelength = PixelArray(ext=4, transform=lambda x, *_: x[::-1, ::-1])
    flux = PixelArray(ext=1, transform=lambda x, *_: x[::-1, ::-1])
    ivar = PixelArray(ext=2, transform=lambda x, *_: x[::-1, ::-1]**-2)
    pixel_flags = PixelArray(ext=3, transform=lambda x, *_: x[::-1, ::-1])
    
    #> Observing Conditions
    date_obs = DateTimeField(null=True)
    jd = FloatField(null=True)
    exptime = FloatField(null=True)
    dithered = BooleanField(null=True)
    f_night_time = FloatField(null=True)
    
    #> Telescope Pointing
    input_ra = FloatField(null=True)
    input_dec = FloatField(null=True)
    n_frames = IntegerField(null=True)
    assigned = IntegerField(null=True)
    on_target = IntegerField(null=True)
    valid = IntegerField(null=True)
    fps = BooleanField(null=True)
    
    #> Statistics and Spectrum Quality 
    snr = FloatField(null=True)
    spectrum_flags = BitField(default=0)
    
    # From https://github.com/sdss/apogee_drp/blob/630d3d45ecff840d49cf75ac2e8a31e22b543838/python/apogee_drp/utils/bitmask.py#L110
    # and https://github.com/sdss/apogee/blob/e134409dc14b20f69e68a0d4d34b2c1b5056a901/python/apogee/utils/bitmask.py#L9
    flag_bad_pixels = spectrum_flags.flag(2**0)
    flag_commissioning = spectrum_flags.flag(2**1)
    flag_bright_neighbor = spectrum_flags.flag(2**2)
    flag_very_bright_neighbor = spectrum_flags.flag(2**3)
    flag_low_snr = spectrum_flags.flag(2**4)
    # 4-8 inclusive are not defined
    flag_persist_high = spectrum_flags.flag(2**9)
    flag_persist_med = spectrum_flags.flag(2**10)
    flag_persist_low = spectrum_flags.flag(2**11)
    flag_persist_jump_pos = spectrum_flags.flag(2**12)
    flag_persist_jump_neg = spectrum_flags.flag(2**13)
    # 14-15 inclusive are not defined
    flag_suspect_rv_combination = spectrum_flags.flag(2**16)
    flag_suspect_broad_lines = spectrum_flags.flag(2**17)
    flag_bad_rv_combination = spectrum_flags.flag(2**18)
    flag_rv_reject = spectrum_flags.flag(2**19)
    flag_rv_suspect = spectrum_flags.flag(2**20)
    flag_multiple_suspect = spectrum_flags.flag(2**21)
    flag_rv_failure = spectrum_flags.flag(2**22)
    flag_suspect_rotation = spectrum_flags.flag(2**23)
    flag_mtpflux_lt_75 = spectrum_flags.flag(2**24)
    flag_mtpflux_lt_50 = spectrum_flags.flag(2**25)
    
    #> Radial Velocity (Doppler)
    v_rad = FloatField(null=True)
    v_rel = FloatField(null=True)
    e_v_rel = FloatField(null=True)
    bc = FloatField(null=True)
    
    doppler_teff = FloatField(null=True)
    doppler_e_teff = FloatField(null=True)
    doppler_logg = FloatField(null=True)
    doppler_e_logg = FloatField(null=True)
    doppler_fe_h = FloatField(null=True)
    doppler_e_fe_h = FloatField(null=True)
    doppler_rchi2 = FloatField(null=True)
    doppler_flags = BitField(default=0)
    
    #> Radial Velocity (X-Correlation)
    xcorr_v_rad = FloatField(null=True)
    xcorr_v_rel = FloatField(null=True)
    xcorr_e_v_rel = FloatField(null=True)
    ccfwhm = FloatField(null=True)
    autofwhm = FloatField(null=True)
    n_components = IntegerField(null=True)

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

    @classmethod
    def get_path_template(cls, release, telescope):
        if release == "sdss5":
            #if apred == "1.3":
            #    # I fucking hate this project.
            #    template = "$SAS_BASE_DIR/../sdss51/sdsswork/mwm/apogee/spectro/redux/ipl-3-{apred}/{apstar}/{telescope}/{healpix_group}/{healpix}/apStar-{apred}-{telescope}-{obj}.fits"
            #else:
            return "$SAS_BASE_DIR/sdsswork/mwm/apogee/spectro/redux/{apred}/visit/{telescope}/{field}/{plate}/{mjd}/apVisit-{apred}-{telescope}-{plate}-{mjd}-{fiber:0>3}.fits"
        else:
            if telescope == "apo1m":
                return "$SAS_BASE_DIR/dr17/apogee/spectro/redux/{apred}/visit/{telescope}/{field}/{mjd}/apVisit-{apred}-{mjd}-{reduction}.fits"
            else:
                return "$SAS_BASE_DIR/dr17/apogee/spectro/redux/{apred}/visit/{telescope}/{field}/{plate}/{mjd}/{prefix}Visit-{apred}-{plate}-{mjd}-{fiber:0>3}.fits"
        

    @property
    def path(self):
        return self.get_path_template(self.release, self.telescope).format(**self.__data__)

    class Meta:
        indexes = (
            (
                (
                    "release",
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


    

class ApogeeVisitSpectrumInApStar(BaseModel, SpectrumMixin):

    """An APOGEE stacked spectrum, stored in an apStar data product."""

    pk = AutoField()

    # Won't appear in a header group because it is first referenced in `Source`.
    source = ForeignKeyField(
        Source, 
        # We want to allow for spectra to be unassociated with a source so that 
        # we can test with fake spectra, etc, but any pipeline should run their
        # own checks to make sure that spectra and sources are linked.
        null=True, 
        index=True,
        column_name="source_pk",
        backref="apogee_visit_spectra_in_apstar",
    )

    #> Spectrum Identifiers
    spectrum_pk = ForeignKeyField(
        Spectrum,
        index=True,
        unique=True,
        lazy_load=False,
    )
    drp_spectrum_pk = ForeignKeyField(
        ApogeeVisitSpectrum,
        index=True,
        unique=True,
        lazy_load=False,
        field=ApogeeVisitSpectrum.spectrum_pk,
    )    

    #> Spectral Data
    wavelength = PixelArray(
        accessor_class=LogLambdaArrayAccessor,
        accessor_kwargs=dict(
            crval=4.179,
            cdelt=6e-6,
            naxis=8575,
        ),
    )
    flux = PixelArray(ext=1, transform=transform)
    ivar = PixelArray(ext=2, transform=lambda *a, **k: _transform_err_to_ivar(transform(*a, **k)))
    pixel_flags = PixelArray(ext=3, transform=transform)

    #> Data Product Keywords
    release = TextField()
    filetype = TextField(default="apStar")
    apred = TextField()
    apstar = TextField(default="stars")
    obj = TextField()
    telescope = TextField()
    # Healpix is only used in SDSS-V, and may not appear in this data product keywords group (since it appears in Source).
    # Here we repeat it because the APOGEE DRP has a habit of incorrectly computing healpix, so we need to store their version so that we can access paths.
    healpix = IntegerField(null=True)
    # field is not used in SDSS-V, but we need a non-null default value otherwise postgres allows rows with same values in all other fields of an index
    field = TextField(default="", null=False)
    prefix = TextField(default="", null=False)
    plate = TextField()
    mjd = IntegerField()
    fiber = IntegerField()
    reduction = TextField(null=True, default="")

    @property
    def path(self):
        #if self.apred == "1.3":
        #    template = "$SAS_BASE_DIR/../sdss51/sdsswork/mwm/apogee/spectro/redux/ipl-3-{apred}/{apstar}/{telescope}/{healpix_group}/{healpix}/apStar-{apred}-{telescope}-{obj}.fits"
        #else:
        template = {
            "sdss5": "$SAS_BASE_DIR/sdsswork/mwm/apogee/spectro/redux/{apred}/{apstar}/{telescope}/{healpix_group}/{healpix}/apStar-{apred}-{telescope}-{obj}.fits",
            "dr17": "$SAS_BASE_DIR/dr17/apogee/spectro/redux/{apred}/{apstar}/{telescope}/{field}/{prefix}Star-{apred}-{obj}.fits"
        }[self.release]

        kwds = self.__data__.copy()
        if self.release == "sdss5":
            healpix = self.healpix or self.source.healpix
            kwds["healpix"] = healpix
            kwds["healpix_group"] = "{:d}".format(int(healpix) // 1000)
        
        return template.format(**kwds)


    class Meta:
        indexes = (
            (
                (
                    "release",
                    "apred",
                    "apstar",
                    "obj",
                    "telescope",
                    #"healpix",
                    "field",
                    "prefix",
                    "plate",
                    "mjd",
                    "fiber"
                ),
                True,
            ),
        )

_transform = lambda x, *_: np.atleast_2d(x)[0]

def _transform_coadded_spectrum(v, image, instance):
    # Accessor class for the PixelArrays
    v = np.atleast_2d(v)
    return v[0]
    if image[0].header["NVISITS"] == 1:
        raise ValueError("No coadded spectrum")
    else:
        return v[0]

    
class ApogeeCoaddedSpectrumInApStar(BaseModel, SpectrumMixin):

    """
    A co-added (stacked) APOGEE spectrum of a star, which is stored in an `apStar` data product.
    """

    pk = AutoField()

    # Won't appear in a header group because it is first referenced in `Source`.
    source = ForeignKeyField(
        Source, 
        # We want to allow for spectra to be unassociated with a source so that 
        # we can test with fake spectra, etc, but any pipeline should run their
        # own checks to make sure that spectra and sources are linked.
        null=True, 
        index=True,
        column_name="source_pk",
        backref="apogee_coadded_spectra_in_apstar",
    )

    #> Identifiers
    star_pk = BigIntegerField(null=True, unique=True)
    spectrum_pk = ForeignKeyField(
        Spectrum,
        null=True,
        index=True,
        unique=True,
        lazy_load=False,
    )

    #> Data Product Keywords
    release = TextField()
    filetype = TextField(default="apStar")
    apred = TextField()
    apstar = TextField(default="stars")
    obj = TextField()
    telescope = TextField()
    healpix = IntegerField(null=True)
    # see comment earlier about nullables with field/prefix
    field = TextField(null=False, default="")
    prefix = TextField(null=False, default="")

    #> Observing Span
    min_mjd = IntegerField(null=True)
    max_mjd = IntegerField(null=True)

    #> Number and Quality of Visits
    n_entries = IntegerField(null=True)
    n_visits = IntegerField(null=True)
    n_good_visits = IntegerField(null=True)
    n_good_rvs = IntegerField(null=True)

    #> Summary Statistics
    snr = FloatField(null=True)
    mean_fiber = FloatField(null=True)
    std_fiber = FloatField(null=True)
    spectrum_flags = BitField(default=0)

    #> Radial Velocity (Doppler)
    v_rad = FloatField(null=True)
    e_v_rad = FloatField(null=True)
    std_v_rad = FloatField(null=True)
    median_e_v_rad = FloatField(null=True)
    
    doppler_teff = FloatField(null=True)
    doppler_e_teff = FloatField(null=True)
    doppler_logg = FloatField(null=True)
    doppler_e_logg = FloatField(null=True)
    doppler_fe_h = FloatField(null=True)
    doppler_e_fe_h = FloatField(null=True)
    doppler_rchi2 = FloatField(null=True)
    doppler_flags = BitField(default=0)

    #> Radial Velocity (X-Correlation)
    xcorr_v_rad = FloatField(null=True)
    xcorr_v_rel = FloatField(null=True)
    xcorr_e_v_rel = FloatField(null=True)
    ccfwhm = FloatField(null=True)
    autofwhm = FloatField(null=True)
    n_components = IntegerField(null=True)

    #> Spectral Data
    wavelength = PixelArray(
        accessor_class=LogLambdaArrayAccessor,
        accessor_kwargs=dict(
            crval=4.179,
            cdelt=6e-6,
            naxis=8575,
        ),
    )
    flux = PixelArray(ext=1, transform=_transform_coadded_spectrum)
    ivar = PixelArray(ext=2, transform=lambda *a, **k: _transform_err_to_ivar(_transform_coadded_spectrum(*a, **k)))
    pixel_flags = PixelArray(ext=3, transform=_transform_coadded_spectrum)

    @property
    def path(self):
        template = {
            "sdss5": "$SAS_BASE_DIR/sdsswork/mwm/apogee/spectro/redux/{apred}/{apstar}/{telescope}/{healpix_group}/{healpix}/apStar-{apred}-{telescope}-{obj}.fits",
            "dr17": "$SAS_BASE_DIR/dr17/apogee/spectro/redux/{apred}/{apstar}/{telescope}/{field}/{prefix}Star-{apred}-{obj}.fits"
        }[self.release]

        kwds = {}
        if self.release == "sdss5":
            kwds["healpix_group"] = "{:d}".format(int(self.healpix) // 1000)

        return template.format(
            **self.__data__,
            **kwds
        )        

    class Meta:
        indexes = (
            (
                (
                    "release",
                    "apred",
                    "apstar",
                    "obj",
                    "telescope",
                    #"healpix",
                    "field",
                    "prefix",
                ),
                True,
            ),
        )            