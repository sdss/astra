import os
import numpy as np

from peewee import (
    AutoField,
    FloatField,
    BooleanField,
    DateTimeField,
    BigIntegerField,
    IntegerField,
    TextField,
    Model,
    ForeignKeyField,
    Node,
    Field,
    BigBitField,
    VirtualField,
    ColumnBase
)
from astropy.io import fits
from playhouse.hybrid import hybrid_property
from playhouse.sqlite_ext import SqliteExtDatabase, JSONField

from astra.models.glossary import Glossary
from astra.models.fields import PixelArray, BitField, PixelArrayAccessorHDF
from astra.utils import expand_path

__all__ = ["Source", "UniqueSpectrum", "Spectrum", "BaseModel", "BossVisitSpectrum", "apMADGICSSpectrum", "database"]

database = SqliteExtDatabase(":memory:", thread_safe=True, pragmas={"foreign_keys": 1})
schema = None



class BaseModel(Model):
    class Meta:
        database = database
        schema = schema
        legacy_table_names = False



class Spectrum:

    def plot(self, rectified=False, **kwargs):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        x, y = (self.wavelength, self.flux)
        c = self.continuum if rectified else 1
        ax.plot(x, y / c, c='k')

        #ax.plot(x, self.model_flux)
        return fig



class Source(BaseModel):

    """ An astronomical source. """

    #: Identifiers
    id = AutoField(help_text=Glossary.source_id)
    healpix = IntegerField(help_text=Glossary.healpix)
    gaia_dr3_source_id = BigIntegerField(help_text=Glossary.gaia_dr3_source_id, null=True)
    tic_v8_id = BigIntegerField(help_text=Glossary.tic_v8_id, null=True)
    sdss4_dr17_apogee_id = TextField(help_text=Glossary.sdss4_dr17_apogee_id, null=True)
    sdss4_dr17_field = TextField(help_text=Glossary.sdss4_dr17_field, null=True)

    #: Astrometry
    ra = FloatField(help_text=Glossary.ra)
    dec = FloatField(help_text=Glossary.dec)
    plx = FloatField(help_text=Glossary.plx, null=True, verbose_name="parallax")
    e_plx = FloatField(help_text=Glossary.e_plx, null=True, verbose_name="e_parallax")
    pmra = FloatField(help_text=Glossary.pmra, null=True)
    e_pmra = FloatField(help_text=Glossary.e_pmra, null=True)
    pmde = FloatField(help_text=Glossary.pmde, null=True)
    e_pmde = FloatField(help_text=Glossary.e_pmde, null=True)
    gaia_v_rad = FloatField(help_text=Glossary.gaia_v_rad, null=True)
    gaia_e_v_rad = FloatField(help_text=Glossary.gaia_e_v_rad, null=True)

    #: Photometry
    g_mag = FloatField(help_text=Glossary.g_mag, null=True)
    bp_mag = FloatField(help_text=Glossary.bp_mag, null=True)
    rp_mag = FloatField(help_text=Glossary.rp_mag, null=True)
    j_mag = FloatField(help_text=Glossary.j_mag, null=True)
    e_j_mag = FloatField(help_text=Glossary.e_j_mag, null=True)
    h_mag = FloatField(help_text=Glossary.h_mag, null=True)
    e_h_mag = FloatField(help_text=Glossary.e_h_mag, null=True)
    k_mag = FloatField(help_text=Glossary.k_mag, null=True)
    e_k_mag = FloatField(help_text=Glossary.e_k_mag, null=True)

    #: Extinction
    

    #: Targeting
    carton_0 = TextField(help_text=Glossary.carton_0, null=True)
    carton_flags = BigBitField(help_text=Glossary.carton_flags, null=True)

    '''
    @property
    def carton_primary_keys(self):
        """ Return the primary keys of the cartons that this source is assigned. """
        i, cartons, cur_size = (0, [], len(self.carton_flags._buffer))
        while True:
            byte_num, byte_offset = divmod(i, 8)
            if byte_num >= cur_size:
                break
            if bool(self.carton_flags._buffer[byte_num] & (1 << byte_offset)):
                cartons.append(i)
            i += 1
        return cartons
    '''


    @property
    def spectra(self):
        for expr, column in self.dependencies():
            if Spectrum in column.model.__mro__[1:]:
                yield from column.model.select().where(expr)



class UniqueSpectrum(BaseModel):

    """ A one dimensional spectrum. """

    id = AutoField(help_text=Glossary.spectrum_id)


class BossVisitSpectrum(BaseModel, Spectrum):

    """A BOSS visit spectrum, where a visit is defined by spectra taken on a single MJD."""

    source = ForeignKeyField(Source, index=True, backref="boss_visit_spectra")
    spectrum_id = ForeignKeyField(UniqueSpectrum, index=True, lazy_load=False)

    wavelength = PixelArray(
        help_text="Wavelength in a vacuum [Angstrom]",
        ext=1, column_name="loglam", transform=lambda x: 10**x
    )
    flux = PixelArray(ext=1)
    ivar = PixelArray(ext=1)
    wresl = PixelArray(ext=1)

    # These are needed for identifying the specFull file.
    release = TextField(help_text=Glossary.release)
    run2d = TextField(help_text=Glossary.run2d)
    mjd = IntegerField(help_text=Glossary.mjd)
    fieldid = IntegerField(help_text=Glossary.fieldid)
    catalogid = BigIntegerField(help_text=Glossary.catalogid)

    # The rest is optional metadata.
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
    zwarning = IntegerField(null=True)

    xcsao_teff = FloatField(null=True)
    xcsao_e_teff = FloatField(null=True)
    xcsao_logg = FloatField(null=True)
    xcsao_e_logg = FloatField(null=True)
    xcsao_fe_h = FloatField(null=True)
    xcsao_e_fe_h = FloatField(null=True)
    xcsao_rxc = FloatField(null=True)

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


class MWMBossVisitSpectrum(BaseModel, Spectrum):

    source = ForeignKeyField(Source, index=True, backref="mwm_boss_visit_spectra")
    spectrum_id = ForeignKeyField(UniqueSpectrum, index=True, lazy_load=False)
    
    #: Data product keywords
    release = TextField(help_text=Glossary.release)
    v_astra = TextField(help_text=Glossary.v_astra)
    run2d = TextField(help_text=Glossary.run2d)
    apred = TextField(help_text=Glossary.apred)
    catalogid = BigIntegerField(help_text=Glossary.catalogid, index=True)
    component = TextField(help_text=Glossary.component, default="")

    #: Observing conditions
    telescope = TextField(help_text=Glossary.telescope)
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

    #: Data reduction pipeline
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
    zwarning = IntegerField(null=True)

    #: XCSAO pipeline
    xcsao_teff = FloatField(null=True)
    xcsao_e_teff = FloatField(null=True)
    xcsao_logg = FloatField(null=True)
    xcsao_e_logg = FloatField(null=True)
    xcsao_fe_h = FloatField(null=True)
    xcsao_e_fe_h = FloatField(null=True)
    xcsao_rxc = FloatField(null=True)

    # TODO: 
    # [ ] used in stack
    # [ ] v_shift
    # [ ] v_bc
    # [ ] pixel_flags


    # TODO: accessor function that takes instance information as well so that we can use one lambda for all?

    _get_ext = lambda x: dict(apo25m=1, lco25m=2)[x.telescope]
    flux = PixelArray(ext=_get_ext, column_name="flux", transform=lambda x: x)
    ivar = PixelArray(ext=_get_ext, column_name="e_flux", transform=lambda x: x**-2)
    




class ApogeeVisitSpectrum(BaseModel, Spectrum):

    source = ForeignKeyField(Source, index=True, backref="apogee_visit_spectra")
    spectrum_id = ForeignKeyField(UniqueSpectrum, index=True, lazy_load=False)

    #: Data Product Keywords
    release = TextField(help_text=Glossary.release)
    apred = TextField(help_text=Glossary.apred)
    mjd = IntegerField(help_text=Glossary.mjd)
    plate = IntegerField(help_text=Glossary.plate)
    telescope = TextField(help_text=Glossary.telescope)
    field = TextField(help_text=Glossary.field)
    fiber = IntegerField(help_text=Glossary.fiber)
    prefix = TextField(help_text=Glossary.prefix)

    wavelength = PixelArray(ext=4, transform=lambda x: x[::-1, ::-1])
    flux = PixelArray(ext=1, transform=lambda x: x[::-1, ::-1])
    e_flux = PixelArray(ext=2, transform=lambda x: x[::-1, ::-1])
    pixel_flags = PixelArray(ext=3, transform=lambda x: x[::-1, ::-1])
    
    #: Identifiers
    apvisit_pk = BigIntegerField(help_text=Glossary.apvisit_pk, null=True)
    sdss4_dr17_apogee_id = TextField(help_text=Glossary.sdss4_dr17_apogee_id, null=True)

    #: Observing meta.
    date_obs = DateTimeField(help_text=Glossary.date_obs)
    jd = FloatField(help_text=Glossary.jd)
    exptime = FloatField(help_text=Glossary.exptime)
    n_frames = IntegerField(help_text=Glossary.n_frames, null=True)
    dithered = BooleanField(help_text=Glossary.dithered, null=True)
    assigned = IntegerField(help_text=Glossary.assigned, null=True)    
    on_target = IntegerField(help_text=Glossary.on_target, null=True)
    valid = IntegerField(help_text=Glossary.valid, null=True)

    #: Statistics 
    snr = FloatField(help_text=Glossary.snr)

    # From https://github.com/sdss/apogee_drp/blob/630d3d45ecff840d49cf75ac2e8a31e22b543838/python/apogee_drp/utils/bitmask.py#L110
    flags = BitField(help_text="Data reduction pipeline flags for this spectrum.")
    flag_bad_pixels = flags.flag(0, help_text="Spectrum has many bad pixels (>20%).")
    flag_commissioning = flags.flag(1, help_text="Commissioning data (MJD <55761); non-standard configuration; poor LSF.")
    flag_bright_neighbor = flags.flag(2, help_text="Star has neighbor more than 10 times brighter.")
    flag_very_bright_neighbor = flags.flag(3, help_text="Star has neighbor more than 100 times brighter.")
    flag_low_snr = flags.flag(4, help_text="Spectrum has low S/N (<5).")

    flag_persist_high = flags.flag(9, help_text="Spectrum has at least 20% of pixels in high persistence region.")
    flag_persist_med = flags.flag(10, help_text="Spectrum has at least 20% of pixels in medium persistence region.")
    flag_persist_low = flags.flag(11, help_text="Spectrum has at least 20% of pixels in low persistence region.")
    flag_persist_jump_pos = flags.flag(12, help_text="Spectrum has obvious positive jump in blue chip.")
    flag_persist_jump_neg = flags.flag(13, help_text="Spectrum has obvious negative jump in blue chip.")

    flag_suspect_rv_combination = flags.flag(16, help_text="RVs from synthetic template differ significantly (~2 km/s) from those from combined template.")
    flag_suspect_broad_lines = flags.flag(17, help_text="Cross-correlation peak with template significantly broader than autocorrelation of template.")
    flag_bad_rv_combination = flags.flag(18, help_text="RVs from synthetic template differ very significantly (~10 km/s) from those from combined template.")
    flag_rv_reject = flags.flag(19, help_text="Rejected visit because cross-correlation RV differs significantly from least squares RV.")
    flag_rv_suspect = flags.flag(20, help_text="Suspect visit (but used!) because cross-correlation RV differs slightly from least squares RV.")
    flag_multiple_suspect = flags.flag(21, help_text="Suspect multiple components from Gaussian decomposition of cross-correlation.")
    flag_rv_failure = flags.flag(22, help_text="RV failure.")

    @hybrid_property
    def flag_bad(self):
        return (
            self.bad_pixels_flag
        |   self.very_bright_neighbor_flag
        |   self.bad_rv_combination_flag
        |   self.rv_failure_flag
        )
    
    @hybrid_property
    def flag_warn(self):
        return self.flags > 0


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




class MWMStarBossSpectrum(BaseModel, Spectrum):

    source = ForeignKeyField(Source, index=True, backref="mwm_star_boss_spectra")
    spectrum_id = ForeignKeyField(UniqueSpectrum, index=True, lazy_load=False)

    release = TextField(help_text=Glossary.release)
    v_astra = TextField(help_text=Glossary.v_astra)
    run2d = TextField(help_text=Glossary.run2d)
    apred = TextField(help_text=Glossary.apred)
    catalogid = BigIntegerField(help_text=Glossary.catalogid, index=True)
    component = TextField(help_text=Glossary.component, default="")

    # mjd_start
    # mjd_end
    # n_visits

    telescope = TextField(help_text=Glossary.telescope)

    _get_ext = lambda x: dict(apo25m=1, lco25m=2)[x.telescope]
    flux = PixelArray(ext=_get_ext, column_name="flux", transform=lambda x: x[0])
    ivar = PixelArray(ext=_get_ext, column_name="e_flux", transform=lambda x: x[0]**-2)
    
    @property
    def path(self):
        return (
            f"$MWM_ASTRA/"
            f"{self.v_astra}/{self.run2d}-{self.apred}/spectra/star/"
            f"{(int(self.catalogid) // 100) % 100:0>2.0f}/{int(self.catalogid) % 100:0>2.0f}/"
            f"mwmStar-{self.v_astra}-{self.catalogid}{self.component}.fits"
        )

    class Meta:
        indexes = (
            (
                (
                    "release",
                    "v_astra",
                    "run2d",
                    "apred",
                    "catalogid",
                    "component",
                    "telescope",
                    "mjd_start",
                    "mjd_end",
                ),
                True,
            ),
        )


# MWMStarApogeeSpectrum? ApogeeCoaddedSpectrum?
class MWMStarApogeeSpectrum(BaseModel, Spectrum):

    source = ForeignKeyField(Source, index=True, backref="mwm_star_apogee_spectra")
    spectrum_id = ForeignKeyField(UniqueSpectrum, index=True, lazy_load=False)

    release = TextField(help_text=Glossary.release)
    v_astra = TextField(help_text=Glossary.v_astra)
    run2d = TextField(help_text=Glossary.run2d)
    apred = TextField(help_text=Glossary.apred)
    catalogid = BigIntegerField(help_text=Glossary.catalogid, index=True)
    component = TextField(help_text=Glossary.component, default="")

    # mjd_start
    # mjd_end
    # n_visits

    telescope = TextField(help_text=Glossary.telescope)

    _get_ext = lambda x: dict(apo25m=3, lco25m=4)[x.telescope]
    flux = PixelArray(ext=_get_ext, column_name="flux", transform=lambda x: x[0])
    ivar = PixelArray(ext=_get_ext, column_name="e_flux", transform=lambda x: x[0]**-2)
    
    @property
    def path(self):
        return (
            f"$MWM_ASTRA/"
            f"{self.v_astra}/{self.run2d}-{self.apred}/spectra/star/"
            f"{(int(self.catalogid) // 100) % 100:0>2.0f}/{int(self.catalogid) % 100:0>2.0f}/"
            f"mwmStar-{self.v_astra}-{self.catalogid}{self.component}.fits"
        )

    class Meta:
        indexes = (
            (
                (
                    "release",
                    "v_astra",
                    "run2d",
                    "apred",
                    "catalogid",
                    "component",
                    "telescope",
                    "mjd_start",
                    "mjd_end",
                ),
                True,
            ),
        )



class apMADGICSSpectrum(BaseModel, Spectrum):

    """A spectrum from the apMADGICS pipeline."""

    source = ForeignKeyField(Source, index=True, backref="apogee_visit_spectra")
    spectrum_id = ForeignKeyField(UniqueSpectrum, index=True, lazy_load=False)

    # TODO: replace this with something that is recognised as a field?
    @property
    def wavelength(self):
        return 10**(4.179 + 6e-6 * np.arange(8575))

    flux = PixelArray(
        column_name="x_starLines_v0",
        accessor_class=PixelArrayAccessorHDF,
        transform=lambda x: 1 + x[125:]
    )
    ivar = PixelArray(
        column_name="x_starLines_err_v0",
        accessor_class=PixelArrayAccessorHDF,
        transform=lambda x: x[125:]**-2
    )

    row_index = IntegerField(index=True)
    v_rad_pixel = PixelArray(column_name="RV_pixoff_final", accessor_class=PixelArrayAccessorHDF)

    release = TextField()
    telescope = TextField()
    field = TextField()
    plate = IntegerField()
    mjd = IntegerField()
    fiber = IntegerField()


    @property
    def path(self):
        return "/uufs/chpc.utah.edu/common/home/sdss50/sdsswork/users/u6039752/working/2023_03_24/outdir_wu/apMADGICS_out.h5"

    



if __name__ == "__main__":
    database.create_tables([
        Source,
        UniqueSpectrum,
        BossVisitSpectrum,
        ApogeeVisitSpectrum,
        apMADGICSSpectrum
    ])

    source = Source.create(ra=1, dec=1, healpix=1)


    spectrum_id = UniqueSpectrum.create().id
    spec = apMADGICSSpectrum(
        source=source,
        spectrum_id=spectrum_id,
        row_index=1
    )
    #  "/uufs/chpc.utah.edu/common/home/sdss50/sdsswork/users/u6039752/working/2023_03_24/input_list.txt"

    # Ingest each spectrum, get initial values for ASPCAP.

    '''
        ap = ApogeeVisitSpectrum.create(
        release="dr17",
        apred="dr17",
        prefix="ap",
        field="AQM_216.74_+36.33",
        fiber=26,
        mjd=59313,
        plate=15305,
        telescope="apo25m",
        spectrum_id=UniqueSpectrum.create().id, 
        source=source
    )
    
    raise a
    from astra.database.astradb import DataProduct

    dp = (
        DataProduct
        .select()
        .where(
            DataProduct.filetype == "specFull"
        )
        .order_by(DataProduct.id.desc())
        .first()
    )


    kwargs = {'mjd': 59146,
    'run2d': 'v6_0_9',
    'fieldid': 15000,
    'isplate': 'p',
    'catalogid': 4375787390}

    # At ingestion time:
    # - would like to get photometry information from catalog
    # - would like to have source information, but not necessary for running tests (eg with fake spectra)
    # - needs to ingest metadata from the file

    kwargs = {"mjd": 59976, "run2d": "v6_0_9", "fieldid": 102906, "isplate": "", "catalogid": 27021597917769274}        


    source = Source.create(ra=1, dec=1, healpix=1)
    spec = BossVisitSpectrum.create(
        release="sdss5", 
        run2d=kwargs["run2d"],
        fieldid=kwargs["fieldid"],
        catalogid=kwargs["catalogid"],
        mjd=kwargs["mjd"], 
        spectrum_id=UniqueSpectrum.create().id, 
        source=source, 
    )


    ap = ApogeeVisitSpectrum.create(
        release="sdss5",
        apred="1.0",
        prefix
        mjd=5000,
        plate="foo",
        telescope="apo25m",
        field="bar",
        fiber=200,
        spectrum_id=UniqueSpectrum.create().id, 
        source=source
    )
    
    raise a
    from astra.tools.spectrum import Spectrum1D


    from time import time


    t_init = time()
    spec.flux
    spec.ivar
    print(time() - t_init)

    N = spec.flux.size
    f = np.memmap("flux.memmap", dtype=np.float32, mode="w+", shape=spec.flux.shape)
    f[:] = spec.flux
    f.flush()
    del f
    
    #t_init = time()
    #np.memmap("flux.memmap", dtype=np.float32, offset=4, mode="r+", shape=10)
    #print(time() - t_init)



    t_init = time()
    s = Spectrum1D.read(spec.path)
    print(time() - t_init)

    '''
