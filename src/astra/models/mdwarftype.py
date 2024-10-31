from playhouse.hybrid import hybrid_property
from astra.fields import (FloatField, TextField, BitField)
from astra.models.pipeline import PipelineOutputModel

class MDwarfType(PipelineOutputModel):

    """M-dwarf type classifier."""

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
