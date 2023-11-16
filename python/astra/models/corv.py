import datetime
from peewee import (
    FloatField,
    DateTimeField,
    TextField,
    ForeignKeyField,
    AutoField,
    IntegerField
)
from astra import __version__
from astra.models.base import BaseModel
from astra.models.source import Source
from astra.models.spectrum import Spectrum
from astra.models.pipeline import PipelineOutputMixin

from astra.glossary import Glossary


class Corv(BaseModel, PipelineOutputMixin):

    """A result from the `corv` pipeline."""

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

    #> Radial Velocity (corv)
    v_rad = FloatField(null=True, help_text=Glossary.v_rad)
    e_v_rad = FloatField(null=True, help_text=Glossary.e_v_rad)    

    #> Stellar Parameters
    teff = FloatField(null=True, help_text=Glossary.teff)
    e_teff = FloatField(null=True, help_text=Glossary.e_teff)
    logg = FloatField(null=True, help_text=Glossary.logg)
    e_logg = FloatField(null=True, help_text=Glossary.e_logg)

    #> Initial values
    initial_teff = FloatField(null=True, help_text=Glossary.initial_teff)
    initial_logg = FloatField(null=True, help_text=Glossary.initial_logg)
    initial_v_rad = FloatField(null=True, help_text="Initial radial velocity [km/s]")

    #> Summary Statistics
    rchi2 = FloatField(null=True, help_text=Glossary.rchi2) 
    result_flags = IntegerField(null=True, help_text=Glossary.result_flags)