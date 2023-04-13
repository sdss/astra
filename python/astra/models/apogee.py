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
from astra.models.base import BaseModel, SpectrumMixin, Source, UniqueSpectrum
from astra.models.glossary import Glossary


class ApogeeVisitSpectrum(BaseModel, SpectrumMixin):

    """An APOGEE visit spectrum."""

    source = ForeignKeyField(
        Source, 
        index=True, 
        backref="apogee_visit_spectra",
        help_text=Glossary.source_id
    )
    spectrum_id = ForeignKeyField(
        UniqueSpectrum, 
        index=True, 
        lazy_load=False,
        help_text=Glossary.spectrum_id
    )

    # Data Product Keywords
    release = TextField(help_text=Glossary.release)
    apred = TextField(help_text=Glossary.apred)
    mjd = IntegerField(help_text=Glossary.mjd)
    plate = IntegerField(help_text=Glossary.plate)
    telescope = TextField(help_text=Glossary.telescope)
    field = TextField(help_text=Glossary.field)
    fiber = IntegerField(help_text=Glossary.fiber)
    prefix = TextField(help_text=Glossary.prefix)

    # Pixel arrays
    wavelength = PixelArray(
        ext=4, 
        transform=lambda x: x[::-1, ::-1],
        help_text=Glossary.wavelength
    )
    flux = PixelArray(
        ext=1,
        transform=lambda x: x[::-1, ::-1],
        help_text=Glossary.flux
    )
    e_flux = PixelArray(
        ext=2,
        transform=lambda x: x[::-1, ::-1],
        help_text=Glossary.e_flux
    )
    pixel_flags = PixelArray(
        ext=3,
        transform=lambda x: x[::-1, ::-1],
        help_text=Glossary.pixel_flags
    )
    
    # Identifiers
    apvisit_pk = BigIntegerField(help_text=Glossary.apvisit_pk, null=True)
    sdss4_dr17_apogee_id = TextField(help_text=Glossary.sdss4_dr17_apogee_id, null=True)

    # Observing meta.
    date_obs = DateTimeField(help_text=Glossary.date_obs)
    jd = FloatField(help_text=Glossary.jd)
    exptime = FloatField(help_text=Glossary.exptime)
    dithered = BooleanField(help_text=Glossary.dithered)
    # The following are null-able because they do not exist in DR17.
    n_frames = IntegerField(help_text=Glossary.n_frames, null=True)
    assigned = IntegerField(help_text=Glossary.assigned, null=True)    
    on_target = IntegerField(help_text=Glossary.on_target, null=True)
    valid = IntegerField(help_text=Glossary.valid, null=True)

    #: Statistics 
    snr = FloatField(help_text=Glossary.snr)

    # From https://github.com/sdss/apogee_drp/blob/630d3d45ecff840d49cf75ac2e8a31e22b543838/python/apogee_drp/utils/bitmask.py#L110
    spectrum_flags = BitField(help_text=Glossary.spectrum_flags)
    bad_pixels_flag = spectrum_flags.flag(1, help_text="Spectrum has many bad pixels (>20%).")
    commissioning_flag = spectrum_flags.flag(2, help_text="Commissioning data (MJD <55761); non-standard configuration; poor LSF.")
    bright_neighbor_flag = spectrum_flags.flag(4, help_text="Star has neighbor more than 10 times brighter.")
    very_bright_neighbor_flag = spectrum_flags.flag(8, help_text="Star has neighbor more than 100 times brighter.")
    low_snr_flag = spectrum_flags.flag(16, help_text="Spectrum has low S/N (<5).")

    persist_high_flag = spectrum_flags.flag(32, help_text="Spectrum has at least 20% of pixels in high persistence region.")
    persist_med_flag = spectrum_flags.flag(64, help_text="Spectrum has at least 20% of pixels in medium persistence region.")
    persist_low_flag = spectrum_flags.flag(128, help_text="Spectrum has at least 20% of pixels in low persistence region.")
    persist_jump_pos_flag = spectrum_flags.flag(256, help_text="Spectrum has obvious positive jump in blue chip.")
    persist_jump_neg_flag = spectrum_flags.flag(512, help_text="Spectrum has obvious negative jump in blue chip.")

    suspect_rv_combination_flag = spectrum_flags.flag(1024, help_text="RVs from synthetic template differ significantly (~2 km/s) from those from combined template.")
    suspect_broad_lines_flag = spectrum_flags.flag(2048, help_text="Cross-correlation peak with template significantly broader than autocorrelation of template.")
    bad_rv_combination_flag = spectrum_flags.flag(4096, help_text="RVs from synthetic template differ very significantly (~10 km/s) from those from combined template.")
    rv_reject_flag = spectrum_flags.flag(8192, help_text="Rejected visit because cross-correlation RV differs significantly from least squares RV.")
    rv_suspect_flag = spectrum_flags.flag(16384, help_text="Suspect visit (but used!) because cross-correlation RV differs slightly from least squares RV.")
    multiple_suspect_flag = spectrum_flags.flag(32768, help_text="Suspect multiple components from Gaussian decomposition of cross-correlation.")
    rv_failure_flag = spectrum_flags.flag(65536, help_text="RV failure.")


    @hybrid_property
    def bad_flag(self):
        return (
            self.bad_pixels_flag
        |   self.very_bright_neighbor_flag
        |   self.bad_rv_combination_flag
        |   self.rv_failure_flag
        )
    

    @hybrid_property
    def warn_flag(self):
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