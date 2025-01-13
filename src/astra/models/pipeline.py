import datetime
from peewee import SQL, ForeignKeyField
from astra import __version__
from astra.fields import (AutoField, FloatField, TextField, BitField, DateTimeField, IntegerField)
from astra.models.base import BaseModel
from astra.models.source import Source
from astra.models.spectrum import Spectrum
from astra.utils import version_string_to_integer

class PipelineOutputModel(BaseModel):

    source_pk = ForeignKeyField(Source, null=True, index=True, lazy_load=False, column_name="source_pk")
    spectrum_pk = ForeignKeyField(Spectrum, index=True, lazy_load=False, column_name="spectrum_pk")
    
    #> Astra Metadata
    task_pk = AutoField()
    v_astra = IntegerField(default=version_string_to_integer(__version__))
    created = DateTimeField(default=datetime.datetime.now)
    modified = DateTimeField(default=datetime.datetime.now)
    t_elapsed = FloatField(null=True)
    t_overhead = FloatField(null=True)
    tag = TextField(default="", index=True)
    # The /1000 here is set in `astra.utils.version_string_to_integer` and `astra.utils.version_integer_to_string`.
    v_astra_major_minor = IntegerField(constraints=[SQL("GENERATED ALWAYS AS (v_astra / 1000) STORED")], _hidden=True)

    class Meta:
        constraints = [
            SQL("UNIQUE (spectrum_pk, v_astra_major_minor)")
        ]

    @classmethod
    def from_spectrum(cls, spectrum, **kwargs):

        kwds = kwargs.copy()
        given_source_pk = kwds.pop("source_pk", None)
        given_spectrum_pk = kwds.pop("spectrum_pk", None)

        if given_source_pk is not None and given_source_pk != spectrum.source_pk:
            raise ValueError(f"`source_pk` mismatch between `spectrum` and `source_pk` argument ({spectrum.source_pk} != {given_source_pk})")
        if given_spectrum_pk is not None and given_spectrum_pk != spectrum.spectrum_pk:
            raise ValueError(f"`spectrum_pk` mismatch between `spectrum` and `spectrum_pk` argument ({spectrum.spectrum_pk} != {given_spectrum_pk})")

        kwds.update({
            "spectrum_pk": spectrum.spectrum_pk,
            "source_pk": spectrum.source_pk,
        })
        return cls(**kwds)
