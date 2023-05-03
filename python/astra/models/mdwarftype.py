from peewee import (
    AutoField,
    FloatField,
    TextField,
    ForeignKeyField,
    BitField,
)
from playhouse.hybrid import hybrid_property

from astra import __version__
from astra.models.base import BaseModel
from astra.models.fields import BitField
from astra.models.source import Source
from astra.models.spectrum import Spectrum
from astra.models.pipeline import PipelineOutputMixin


class MDwarfType(BaseModel, PipelineOutputMixin):

    """A result from the M-dwarf type classifier."""
    
    sdss_id = ForeignKeyField(Source, index=True)
    spectrum_id = ForeignKeyField(Spectrum, index=True, lazy_load=False)
    
    #> Astra Metadata
    task_id = AutoField()
    v_astra = TextField(default=__version__)
    t_elapsed = FloatField(null=True)
    tag = TextField(default="", index=True)
    
    #> Results
    continuum = FloatField(null=True)
    spectral_type = TextField()
    sub_type = FloatField()
    reduced_chi_sq = FloatField()
    result_flags = BitField(default=0)
    
    @hybrid_property
    def flag_bad(self):
        return (self.result_flags > 0)

    @flag_bad.expression
    def flag_bad(self):
        return (self.result_flags > 0)
    
