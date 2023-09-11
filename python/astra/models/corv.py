import datetime
from peewee import (
    FloatField,
    BooleanField,
    DateTimeField,
    BigIntegerField,
    IntegerField,
    TextField,
    ForeignKeyField,
    AutoField
)
from astra import __version__
from astra.models.base import BaseModel
from astra.models.source import Source
from astra.models.spectrum import Spectrum

from astra.glossary import Glossary


class Corv(BaseModel, PipelineOutputMixin):

    """A result from the `corv` pipeline."""

    source_id = ForeignKeyField(Source, null=True, index=True, lazy_load=False)
    spectrum_id = ForeignKeyField(
        Spectrum, 
        index=True, 
        lazy_load=False,
        help_text=Glossary.spectrum_id
    )
    
    #> Astra Metadata
    task_id = AutoField(help_text=Glossary.task_id)
    v_astra = TextField(default=__version__, help_text=Glossary.v_astra)
    created = DateTimeField(default=datetime.datetime.now)
    t_elapsed = FloatField(null=True, help_text=Glossary.t_elapsed)
    t_overhead = FloatField(null=True)
    tag = TextField(default="", index=True, help_text=Glossary.tag)

    #> Radial Velocity (corv)
    v_rad = FloatField(null=True, help_text=Glossary.v_rad)
    e_v_rad = FloatField(null=True, help_text=Glossary.e_v_rad)    
    initial_v_rad = FloatField(null=True, help_text="Initial radial velocity [km/s]")

    #> Stellar Parameters
    teff = FloatField(null=True, help_text=Glossary.teff)
    e_teff = FloatField(null=True, help_text=Glossary.e_teff)
    logg = FloatField(null=True, help_text=Glossary.logg)
    e_logg = FloatField(null=True, help_text=Glossary.e_logg)

    #> Summary Statistics
    dof = IntegerField(null=True, help_text=Glossary.dof)
    chi2 = FloatField(null=True, help_text=Glossary.chi2)
    rchi2 = FloatField(null=True, help_text=Glossary.rchi2)