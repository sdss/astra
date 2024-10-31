import pytest
import peewee
import os
os.environ["ASTRA_DATABASE_PATH"] = ":memory:"

def test_pipeline_unique_constraint_spectrum_major_minor():

    from astra import __version__
    from astra.models.source import Source
    from astra.models.spectrum import Spectrum
    from astra.models.pipeline import PipelineOutputModel
    from astra.utils import version_string_to_integer, version_integer_to_string

    class Dummy(PipelineOutputModel):
        pass
    
    for model in (Source, Spectrum, Dummy):        
        model.create_table()

    Source.create()
    Spectrum.create()

    r1 = Dummy.create(source_pk=1, spectrum_pk=1)

    with pytest.raises(peewee.IntegrityError):
        Dummy.create(source_pk=1, spectrum_pk=1)

    with pytest.raises(peewee.IntegrityError):
        Dummy.create(spectrum_pk=1)

    # Try bump the version
    major, minor, patch = map(int, __version__.split("."))
    Dummy.create(spectrum_pk=1, v_astra=version_string_to_integer(f"{major}.{minor+1}.{patch}"))
    Dummy.create(spectrum_pk=1, v_astra=version_string_to_integer(f"{major+1}.{minor}.{patch}"))



def test_pipeline_replace_on_conflict():
        
    from astra.fields import IntegerField
    from astra import task
    from astra.models.apogee import ApogeeVisitSpectrum
    from astra.models.source import Source
    from astra.models.spectrum import Spectrum
    from astra.models.pipeline import PipelineOutputModel
    from astra.utils import version_string_to_integer, version_integer_to_string
    from typing import Iterable
    from time import sleep

    class ThisDummy(PipelineOutputModel):
        number = IntegerField()

    
    @task
    def dummy_task(spectra) -> Iterable[ThisDummy]:
        for i, spectrum in enumerate(spectra):
            yield ThisDummy.from_spectrum(spectrum, number=i)

    for model in (Source, Spectrum, ThisDummy, ApogeeVisitSpectrum):        
        model.create_table()

    Source.create()
    Spectrum.create()
    Spectrum.create()

    s = ApogeeVisitSpectrum.create(spectrum_pk=1, source_pk=1, release="test", apred="apred", plate="plate", telescope="telescope", fiber=0, mjd=0, field="field", prefix="ap")
    s2 = ApogeeVisitSpectrum.create(spectrum_pk=2, source_pk=1, release="test", apred="apred", plate="plate", telescope="telescope", fiber=1, mjd=0, field="field", prefix="ap")

    r1 = list(dummy_task([s]))[0].__data__
    sleep(1)
    r2 = list(dummy_task([s]))[0].__data__

    assert r1["created"] == r2["created"]
    assert r2["modified"] > r1["modified"]
    assert r1["task_pk"] == r2["task_pk"]
    assert r1["spectrum_pk"] == r2["spectrum_pk"]

    r3 = list(dummy_task([s2, s]))[1].__data__
    assert r3["number"] > r2["number"]
    assert r3["spectrum_pk"] == r1["spectrum_pk"]
    assert r3["task_pk"] == r1["task_pk"]

