import numpy as np
import datetime
from peewee import (
    AutoField,
    FloatField,
    TextField,
    ForeignKeyField,
    BitField,
    DateTimeField
)

from astra import __version__
from astra.models.base import BaseModel
from astra.models.fields import BitField
from astra.models.source import Source
from astra.models.spectrum import Spectrum
from astra.models.pipeline import PipelineOutputMixin

from astra.glossary import Glossary

class Clam(BaseModel, PipelineOutputMixin):

    """A result from the Clam pipeline."""

    source_pk = ForeignKeyField(Source, null=True, index=True, lazy_load=False, help_text=Glossary.source_pk)
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

    #> Stellar Labels
    teff = FloatField(null=True)
    e_teff = FloatField(null=True)
    logg = FloatField(null=True)
    e_logg = FloatField(null=True)
    m_h = FloatField(null=True)
    e_m_h = FloatField(null=True)
    n_m = FloatField(null=True)
    e_n_m = FloatField(null=True)
    c_m = FloatField(null=True)
    e_c_m = FloatField(null=True)
    v_micro = FloatField(null=True)
    e_v_micro = FloatField(null=True)
    v_sini = FloatField(null=True)
    e_v_sini = FloatField(null=True)

    #> Initial position
    initial_teff = FloatField(null=True)
    initial_logg = FloatField(null=True)
    initial_m_h = FloatField(null=True)
    initial_n_m = FloatField(null=True)
    initial_c_m = FloatField(null=True)
    initial_v_micro = FloatField(null=True)
    initial_v_sini = FloatField(null=True)

    #> Summary Statistics
    rchi2 = FloatField(null=True)
    result_flags = BitField(default=0, help_text="Flags describing the results")
    flag_spectrum_io_error = result_flags.flag(2**0, help_text="Spectrum I/O error")
    flag_runtime_error = result_flags.flag(2**1, help_text="Runtime error")