import datetime
from playhouse.hybrid import hybrid_property
from astra import __version__
from astra.fields import (
    AutoField,
    FloatField,
    TextField,
    ForeignKeyField,
    BitField,
    DateTimeField,    
)
from astra.models.base import BaseModel
from astra.models.source import Source
from astra.models.spectrum import Spectrum
from astra.models.pipeline import PipelineOutputMixin

class MDwarfType(BaseModel, PipelineOutputMixin):

    """M-dwarf type classifier."""

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

    #> M Dwarf Type
    spectral_type = TextField()
    sub_type = FloatField()
    rchi2 = FloatField()
    continuum = FloatField(null=True)
    result_flags = BitField(default=0)
    
    @hybrid_property
    def flag_bad(self):
        return (self.result_flags > 0)

    @flag_bad.expression
    def flag_bad(self):
        return (self.result_flags > 0)