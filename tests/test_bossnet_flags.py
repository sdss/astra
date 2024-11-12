import pytest
import peewee
import os
os.environ["ASTRA_DATABASE_PATH"] = ":memory:"

def test_bossnet_flags():
        
    import datetime
    from astra import __version__
    from astra.fields import AutoField, ForeignKeyField, DateTimeField
    from astra.models.base import database, BaseModel
    from astra.models.source import Source
    from astra.models.spectrum import Spectrum, SpectrumMixin
    from astra.models.bossnet import BossNet
    from astra.models.boss import BossVisitSpectrum

    class DummyBossVisitSpectrum(BaseModel, SpectrumMixin):

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

    models = (Source, Spectrum, BossNet, DummyBossVisitSpectrum)
    database.create_tables(models)

    Source.create()
    Spectrum.create()
    s = DummyBossVisitSpectrum.create(
        source_pk=1,
        spectrum_pk=1,
        release="sdss5",
        filetype="specFull",
        run2d="run2d",
        mjd=1,
        fieldid=1,
        catalogid=1,
        healpix=1
    )

    scenarios = [
        ({}, (lambda r: r.flag_runtime_exception, lambda r: r.result_flags > 0)),
        (dict(teff=5000), (lambda r: not r.flag_unreliable_teff, lambda r: not r.flag_runtime_exception)),
        (dict(logg=3), (lambda r: not r.flag_unreliable_logg, )),
        (dict(fe_h=-1), (lambda r: not r.flag_unreliable_fe_h, )),
        (dict(teff=5000, logg=3, fe_h=-1), (lambda r: not r.flag_unreliable_teff, lambda r: not r.flag_unreliable_logg, lambda r: not r.flag_unreliable_fe_h, lambda r: not r.flag_runtime_exception)),
        (dict(teff=1699), (lambda r: r.flag_unreliable_teff, lambda r: r.result_flags > 0)),
        (dict(teff=100001), (lambda r: r.flag_unreliable_teff, lambda r: r.result_flags > 0)),
        (dict(teff=5000, fe_h=0, logg=-1.1), (lambda r: r.flag_unreliable_logg, lambda r: r.result_flags > 0)),
        (dict(teff=5000, fe_h=0, logg=10.1), (lambda r: r.flag_unreliable_logg, lambda r: r.result_flags > 0)),
        (dict(teff=5000, logg=3, fe_h=-4.1), (lambda r: r.flag_unreliable_fe_h, lambda r: r.result_flags > 0)),
        (dict(teff=5000, logg=3, fe_h=2.1), (lambda r: r.flag_unreliable_fe_h, lambda r: r.result_flags > 0)),
        (dict(teff=3100, logg=3, fe_h=-1), (lambda r: r.flag_unreliable_fe_h, lambda r: r.result_flags > 0)),
        (dict(teff=5000, logg=6, fe_h=-1), (lambda r: r.flag_unreliable_fe_h, lambda r: r.result_flags > 0)),
        (dict(teff=3100, logg=5, fe_h=-1), (lambda r: r.flag_suspicious_fe_h, )),
    ]

    for kwds, expectations in scenarios:
        r = BossNet.from_spectrum(s, **kwds)
        for n, fun in enumerate(expectations):
            assert fun(r), f"Failed on scenario {n} with {kwds}"
        r.save()
        r = BossNet.get(task_pk=r.task_pk)
        for n, fun in enumerate(expectations):
            assert fun(r), f"Failed on scenario {n} with {kwds}"
        r.delete_instance()
        


