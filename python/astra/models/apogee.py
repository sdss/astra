from peewee import (
    AutoField,
    FloatField,
    BooleanField,
    DateTimeField,
    BigIntegerField,
    IntegerField,
    TextField,
    ForeignKeyField,
    DeferredForeignKey,
    fn,
)
import numpy as np
import os
from playhouse.hybrid import hybrid_property
from astra.models.fields import PixelArray, BitField
from astra.models.base import BaseModel
from astra.models.spectrum import (Spectrum, SpectrumMixin)
from astra.models.source import Source

from astra.glossary import Glossary

from astropy.constants import c
from astropy import units as u

C_KM_S = c.to(u.km / u.s).value



def _transform_err_to_ivar(err, *args, **kwargs):
    ivar = np.atleast_2d(err)[0]**-2
    ivar[~np.isfinite(ivar)] = 0
    return ivar




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
    )
    catalogid = BigIntegerField(index=True, null=True, help_text="SDSS input catalog identifier")
    visit_pk = BigIntegerField(null=True, unique=True, help_text="APOGEE DRP `visit` primary key")
    rv_visit_pk = BigIntegerField(null=True, unique=True, help_text="APOGEE DRP `rv_visit` primary key")

    #> Data Product Keywords
    release = TextField(index=True, help_text=Glossary.release)
    filetype = TextField(default="apStar", help_text=Glossary.filetype)
    apred = TextField(index=True, help_text=Glossary.apred)
    plate = TextField(index=True, help_text=Glossary.plate) # most are integers, but not all!
    telescope = TextField(index=True, help_text=Glossary.telescope)
    fiber = IntegerField(index=True, help_text=Glossary.fiber)
    mjd = IntegerField(index=True, help_text=Glossary.mjd)
    field = TextField(index=True, help_text=Glossary.field)
    prefix = TextField(help_text=Glossary.prefix)
    reduction = TextField(default="", help_text=Glossary.reduction) # only used for DR17 apo1m spectra
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
    obj = TextField(null=True, help_text=Glossary.obj)

    #> Spectral Data
    wavelength = PixelArray(
        ext=4, 
        transform=lambda x, *_: x[::-1, ::-1],
    )
    flux = PixelArray(
        ext=1,
        transform=lambda x, *_: x[::-1, ::-1],
        help_text="Flux [10^-17 ergs/s/cm^2/A]"
    )
    ivar = PixelArray(
        ext=2,
        transform=lambda x, *_: x[::-1, ::-1]**-2,
    )
    pixel_flags = PixelArray(
        ext=3,
        transform=lambda x, *_: x[::-1, ::-1],
    )
    
    #> Observing Conditions
    date_obs = DateTimeField(null=True, help_text=Glossary.date_obs)
    jd = FloatField(null=True, help_text=Glossary.jd)
    exptime = FloatField(null=True, help_text=Glossary.exptime)
    dithered = BooleanField(null=True, help_text=Glossary.dithered)
    
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
    spectrum_flags = BitField(default=0, help_text=Glossary.spectrum_flags)
    
    # From https://github.com/sdss/apogee_drp/blob/630d3d45ecff840d49cf75ac2e8a31e22b543838/python/apogee_drp/utils/bitmask.py#L110
    flag_bad_pixels = spectrum_flags.flag(2**0, help_text="Spectrum has many bad pixels (>20%).")
    flag_commissioning = spectrum_flags.flag(2**1, help_text="Commissioning data (MJD <55761); non-standard configuration; poor LSF.")
    flag_bright_neighbor = spectrum_flags.flag(2**2, help_text="Star has neighbor more than 10 times brighter.")
    flag_very_bright_neighbor = spectrum_flags.flag(2**3, help_text="Star has neighbor more than 100 times brighter.")
    flag_low_snr = spectrum_flags.flag(2**4, help_text="Spectrum has low S/N (<5).")
    flag_persist_high = spectrum_flags.flag(2**5, help_text="Spectrum has at least 20% of pixels in high persistence region.")
    flag_persist_med = spectrum_flags.flag(2**6, help_text="Spectrum has at least 20% of pixels in medium persistence region.")
    flag_persist_low = spectrum_flags.flag(2**7, help_text="Spectrum has at least 20% of pixels in low persistence region.")
    flag_persist_jump_pos = spectrum_flags.flag(2**8, help_text="Spectrum has obvious positive jump in blue chip.")
    flag_persist_jump_neg = spectrum_flags.flag(2**9, help_text="Spectrum has obvious negative jump in blue chip.")
    flag_suspect_rv_combination = spectrum_flags.flag(2**10, help_text="RVs from synthetic template differ significantly (~2 km/s) from those from combined template.")
    flag_suspect_broad_lines = spectrum_flags.flag(2**11, help_text="Cross-correlation peak with template significantly broader than autocorrelation of template.")
    flag_bad_rv_combination = spectrum_flags.flag(2**12, help_text="RVs from synthetic template differ very significantly (~10 km/s) from those from combined template.")
    flag_rv_reject = spectrum_flags.flag(2**13, help_text="Rejected visit because cross-correlation RV differs significantly from least squares RV.")
    flag_rv_suspect = spectrum_flags.flag(2**14, help_text="Suspect visit (but used!) because cross-correlation RV differs slightly from least squares RV.")
    flag_multiple_suspect = spectrum_flags.flag(2**15, help_text="Suspect multiple components from Gaussian decomposition of cross-correlation.")
    flag_rv_failure = spectrum_flags.flag(2**16, help_text="RV failure.")

    #> Radial Velocity (Doppler)
    v_rad = FloatField(null=True, help_text=Glossary.v_rad)
    v_rel = FloatField(null=True, help_text=Glossary.v_rel)
    e_v_rel = FloatField(null=True, help_text=Glossary.e_v_rel)
    bc = FloatField(null=True, help_text=Glossary.bc)
    doppler_rchi2 = FloatField(null=True, help_text=Glossary.doppler_rchi2)
    
    doppler_teff = FloatField(null=True, help_text=Glossary.teff)
    doppler_e_teff = FloatField(null=True, help_text=Glossary.e_teff)
    doppler_logg = FloatField(null=True, help_text=Glossary.logg)
    doppler_e_logg = FloatField(null=True, help_text=Glossary.e_logg)
    doppler_fe_h = FloatField(null=True, help_text=Glossary.fe_h)
    doppler_e_fe_h = FloatField(null=True, help_text=Glossary.e_fe_h)
    doppler_flags = BitField(default=0, help_text="Doppler flags") # TODO: is this actually STARFLAG from the DRP?

    #> Radial Velocity (X-Correlation)
    xcorr_v_rad = FloatField(null=True, help_text=Glossary.v_rad)
    xcorr_v_rel = FloatField(null=True, help_text=Glossary.v_rel)
    xcorr_e_v_rel = FloatField(null=True, help_text=Glossary.e_v_rel)
    ccfwhm = FloatField(null=True, help_text=Glossary.ccfwhm)
    autofwhm = FloatField(null=True, help_text=Glossary.autofwhm)
    n_components = IntegerField(null=True, help_text=Glossary.n_components)    

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
        default=Spectrum.create,
        help_text=Glossary.spectrum_pk
    )
    drp_spectrum_pk = ForeignKeyField(
        ApogeeVisitSpectrum,
        lazy_load=False,
        field=ApogeeVisitSpectrum.spectrum_pk,
        help_text=Glossary.drp_spectrum_pk
    )    

    #> Spectral Data
    @property
    def wavelength(self):
        # TODO: Do I need to store this as a PixelArray?
        return 10**(4.179 + 6e-6 * np.arange(8575))

    flux = PixelArray(
        ext=1,
        transform=transform,
        help_text=Glossary.flux,
        pixels=8575
    )
    ivar = PixelArray(
        ext=2,
        transform=lambda *a, **k: _transform_err_to_ivar(transform(*a, **k)),
        help_text=Glossary.ivar,
        pixels=8575
    )
    pixel_flags = PixelArray(
        ext=3,
        transform=transform,
        help_text=Glossary.pixel_flags,
        pixels=8575
    )

    #> Data Product Keywords
    release = TextField(help_text=Glossary.release)
    filetype = TextField(default="apStar", help_text=Glossary.filetype)
    apred = TextField(help_text=Glossary.apred)
    apstar = TextField(default="stars", help_text=Glossary.apstar)
    obj = TextField(help_text=Glossary.obj)
    telescope = TextField(help_text=Glossary.telescope)
    # Healpix is only used in SDSS-V, and may not appear in this data product keywords group (since it appears in Source).
    # Here we repeat it because the APOGEE DRP has a habit of incorrectly computing healpix, so we need to store their version so that we can access paths.
    healpix = IntegerField(null=True, help_text=Glossary.healpix)
    field = TextField(null=True, help_text=Glossary.field) # not used in SDSS-V
    prefix = TextField(null=True, help_text=Glossary.prefix) # not used in SDSS-V
    plate = TextField(help_text=Glossary.plate)
    mjd = IntegerField(help_text=Glossary.mjd)
    fiber = IntegerField(help_text=Glossary.fiber)
    reduction = TextField(default="", help_text=Glossary.reduction) # only used for DR17 apo1m spectra



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
                    "healpix",
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
    if image[0].header["NVISITS"] == 1:
        raise ValueError("No coadded spectrum")
    else:
        return v[0]

    
class ApogeeCoaddedSpectrumInApStar(BaseModel, SpectrumMixin):

    """
    A co-added (stacked) APOGEE spectrum of a star, which is stored in an `apStar` data product.
    """

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

    #> Spectrum Identifiers
    spectrum_pk = ForeignKeyField(
        Spectrum,
        index=True,
        lazy_load=False,
        primary_key=True,
        default=Spectrum.create,
        help_text=Glossary.spectrum_pk
    )

    #> Data Product Keywords
    release = TextField(help_text=Glossary.release)
    filetype = TextField(default="apStar", help_text=Glossary.filetype)
    apred = TextField(help_text=Glossary.apred)
    apstar = TextField(default="stars", help_text=Glossary.apstar)
    obj = TextField(help_text=Glossary.obj)
    telescope = TextField(help_text=Glossary.telescope)
    healpix = IntegerField(null=True, help_text=Glossary.healpix) 
    field = TextField(null=True, help_text=Glossary.field) # not used in SDSS-V
    prefix = TextField(null=True, help_text=Glossary.prefix) # not used in SDSS-V

    #> Spectral Data
    @property
    def wavelength(self):
        # TODO: Do I need to store this as a PixelArray?
        return 10**(4.179 + 6e-6 * np.arange(8575))

    flux = PixelArray(
        ext=1,
        transform=_transform_coadded_spectrum,
        pixels=8575,
        help_text=Glossary.flux
    )
    ivar = PixelArray(
        ext=2,
        transform=lambda *a, **k: _transform_err_to_ivar(_transform_coadded_spectrum(*a, **k)),
        pixels=8575,
        help_text=Glossary.ivar
    )
    pixel_flags = PixelArray(
        ext=3,
        transform=_transform_coadded_spectrum,
        pixels=8575,
        help_text=Glossary.pixel_flags
    )


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
                    "healpix",
                    "field",
                    "prefix",
                ),
                True,
            ),
        )            