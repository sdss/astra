from peewee import (
    FloatField,
    BooleanField,
    DateTimeField,
    BigIntegerField,
    IntegerField,
    TextField,
    ForeignKeyField,
)
from astra.models.base import BaseModel, SpectrumMixin, Source, UniqueSpectrum
from astra.models.glossary import Glossary
from astra.models.fields import PixelArray, BitField

xcsao_glossary = Glossary("XCSAO")


class BossVisitSpectrum(BaseModel, SpectrumMixin):

    """A BOSS visit spectrum, where a visit is defined by spectra taken on a single MJD."""

    source = ForeignKeyField(
        Source,
        index=True,
        backref="boss_visit_spectra",
        help_text=Glossary.source_id
    )
    spectrum_id = ForeignKeyField(
        UniqueSpectrum, 
        index=True, 
        lazy_load=False,
        help_text=Glossary.spectrum_id
    )

    wavelength = PixelArray(
        ext=1, 
        column_name="loglam", 
        transform=lambda x: 10**x,
        help_text=Glossary.wavelength
    )
    flux = PixelArray(ext=1, help_text=Glossary.flux)
    ivar = PixelArray(ext=1, help_text=Glossary.ivar)
    wresl = PixelArray(ext=1) # TODO: No help text yet!

    # These are needed for identifying the specFull file.
    release = TextField(help_text=Glossary.release)
    run2d = TextField(help_text=Glossary.run2d)
    mjd = IntegerField(help_text=Glossary.mjd)
    fieldid = IntegerField(help_text=Glossary.fieldid)
    catalogid = BigIntegerField(help_text=Glossary.catalogid)

    # The rest is optional metadata.
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
    zwarning = IntegerField(null=True, help_text=Glossary.zwarning)

    xcsao_teff = FloatField(null=True, help_text=xcsao_glossary.teff)
    xcsao_e_teff = FloatField(null=True, help_text=xcsao_glossary.e_teff)
    xcsao_logg = FloatField(null=True, help_text=xcsao_glossary.logg)
    xcsao_e_logg = FloatField(null=True, help_text=xcsao_glossary.e_logg)
    xcsao_fe_h = FloatField(null=True, help_text=xcsao_glossary.fe_h)
    xcsao_e_fe_h = FloatField(null=True, help_text=xcsao_glossary.e_fe_h)
    xcsao_rxc = FloatField(null=True, help_text=xcsao_glossary.rxc)

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
