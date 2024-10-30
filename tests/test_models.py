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

    
