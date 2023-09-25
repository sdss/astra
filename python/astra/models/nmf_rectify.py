import datetime
import numpy as np
from peewee import (
    AutoField,
    IntegerField,
    FloatField,
    TextField,
    ForeignKeyField,
    DateTimeField,
    BooleanField,
    fn,
)
from playhouse.postgres_ext import ArrayField

from astra.models.base import BaseModel
from astra.models.spectrum import Spectrum
from astra.models.source import Source

from astra import __version__
from astra.glossary import Glossary



# The actual training set contains the continuum-normalized fluxes, labels, error arrays, etc.
# These two models are simply to link spectra to records in the database


class ApogeeNMFRectification(BaseModel):

    source_pk = ForeignKeyField(Source, null=True, index=True, lazy_load=False)
    spectrum_pk = ForeignKeyField(
        Spectrum, 
        index=True, 
        lazy_load=False,
        help_text=Glossary.spectrum_pk
    )
    
    #> Astra Metadata
    task_pk = AutoField(help_text=Glossary.task_pk)
    v_astra = TextField(default=__version__, help_text=Glossary.v_astra)
    created = DateTimeField(default=datetime.datetime.now, help_text=Glossary.created)
    t_elapsed = FloatField(null=True, help_text=Glossary.t_elapsed)
    t_overhead = FloatField(null=True, help_text=Glossary.t_overhead)
    tag = TextField(default="", index=True, help_text=Glossary.tag)
    
    #> Continuum Fitting
    theta = ArrayField(FloatField)
    L = FloatField()
    deg = IntegerField()
    n_components = IntegerField()
    rchi2 = FloatField()
    