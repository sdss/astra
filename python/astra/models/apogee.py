from peewee import (
    FloatField,
    BooleanField,
    DateTimeField,
    BigIntegerField,
    IntegerField,
    TextField,
    ForeignKeyField,
)
from playhouse.hybrid import hybrid_property

from astra.models.fields import PixelArray, BitField
from astra.models.base import BaseModel
from astra.models.spectrum import (Spectrum, SpectrumMixin)
from astra.models.source import Source


class ApogeeVisitSpectrum(BaseModel, SpectrumMixin):

    """An APOGEE visit spectrum."""
    
    # Won't appear in a header group because it is first referenced in `Source`.
    sdss_id = ForeignKeyField(
        Source, 
        index=True, 
        backref="apogee_visit_spectra",
    )

    #> Spectrum Identifier
    spectrum_id = ForeignKeyField(
        Spectrum, 
        index=True, 
        primary_key=True,
        lazy_load=False,
    )

    #> Data Product Keywords
    release = TextField()
    apred = TextField()
    mjd = IntegerField()
    plate = IntegerField()
    telescope = TextField()
    field = TextField()
    fiber = IntegerField()
    prefix = TextField()

    # Pixel arrays
    wavelength = PixelArray(
        ext=4, 
        transform=lambda x: x[::-1, ::-1],
    )
    flux = PixelArray(
        ext=1,
        transform=lambda x: x[::-1, ::-1],
    )
    e_flux = PixelArray(
        ext=2,
        transform=lambda x: x[::-1, ::-1],
    )
    pixel_flags = PixelArray(
        ext=3,
        transform=lambda x: x[::-1, ::-1],
    )
    
    #> APOGEE Identifiers
    apvisit_pk = BigIntegerField(null=True)
    sdss4_dr17_apogee_id = TextField(null=True)

    #> Observing Conditions
    date_obs = DateTimeField()
    jd = FloatField()
    exptime = FloatField()
    dithered = BooleanField()
    
    #> Telescope Pointing
    n_frames = IntegerField()
    assigned = IntegerField()
    on_target = IntegerField()
    valid = IntegerField()
    
    #> Statistics and Spectrum Quality 
    snr = FloatField()
    spectrum_flags = BitField()
    
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


    @property
    def path(self):
        templates = {
            "sdss5": "$SAS_BASE_DIR/sdsswork/mwm/apogee/spectro/redux/{apred}/visit/{telescope}/{field}/{plate}/{mjd}/apVisit-{apred}-{telescope}-{plate}-{mjd}-{fiber:0>3}.fits",
            "dr17": "$SAS_BASE_DIR/dr17/apogee/spectro/redux/{apred}/visit/{telescope}/{field}/{plate}/{mjd}/{prefix}Visit-{apred}-{plate}-{mjd}-{fiber:0>3}.fits"
        }
        return templates[self.release].format(**self.__data__)
    
    
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
                ),
                True,
            ),
        )