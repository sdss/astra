import datetime
from astra import __version__
from astra.fields import (AutoField, DateTimeField, FloatField, TextField, IntegerField, ForeignKeyField)
from astra.models.source import Source
from astra.models.spectrum import Spectrum
from astra.models.pipeline import PipelineOutputModel

class Corv(PipelineOutputModel):

    """A result from the `corv` pipeline."""

    source_pk = ForeignKeyField(Source, null=True, index=True, lazy_load=False)
    spectrum_pk = ForeignKeyField(Spectrum, index=True, lazy_load=False)
    
    #> Astra Metadata
    task_pk = AutoField()
    v_astra = TextField(default=__version__)
    created = DateTimeField(default=datetime.datetime.now)
    modified = DateTimeField(default=datetime.datetime.now)
    t_elapsed = FloatField(null=True)
    t_overhead = FloatField(null=True)
    tag = TextField(default="", index=True)

    #> Radial Velocity (corv)
    v_rad = FloatField(null=True)
    e_v_rad = FloatField(null=True)

    #> Stellar Parameters
    teff = FloatField(null=True)
    e_teff = FloatField(null=True)
    logg = FloatField(null=True)
    e_logg = FloatField(null=True)

    #> Initial values
    initial_teff = FloatField(null=True)
    initial_logg = FloatField(null=True)
    initial_v_rad = FloatField(null=True)

    #> Summary Statistics
    rchi2 = FloatField(null=True)
    result_flags = IntegerField(null=True)
