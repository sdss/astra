import datetime
import numpy as np
from peewee import fn
from astra.fields import (
    ArrayField,
    AutoField,
    IntegerField,
    FloatField,
    TextField,
    ForeignKeyField,
    DateTimeField,
    BooleanField,
    BitField
)
from astra.models.base import BaseModel
from astra.models.spectrum import Spectrum
from astra.models.source import Source

from astra import __version__

class NMFRectify(BaseModel):

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
    
    #> Continuum Fitting
    log10_W = ArrayField(FloatField, null=True, help_text="log10(W) NMF coefficients to compute spectra")
    continuum_theta = ArrayField(FloatField, null=True, help_text="Continuum coefficients")
    L = FloatField(help_text="Sinusoidal length scale for continuum")
    deg = IntegerField(help_text="Sinusoidal degree for continuum")
    rchi2 = FloatField(null=True, help_text=Glossary.rchi2)
    joint_rchi2 = FloatField(null=True, help_text="Joint reduced chi^2 from simultaneous fit")
    nmf_flags = BitField(default=0, help_text="NMF Continuum method flags") #TODO: rename as nmf_flags
    flag_initialised_from_small_w = nmf_flags.flag(2**0)

    flag_could_not_read_spectrum = nmf_flags.flag(2**3)
    flag_runtime_exception = nmf_flags.flag(2**4)
