import datetime
from peewee import (
    AutoField,
    FloatField,
    TextField,
    ForeignKeyField,
    BitField,
    DateTimeField
)
from playhouse.hybrid import hybrid_property
from astra import __version__
from astra.models.base import BaseModel
from astra.models.fields import BitField
from astra.models.source import Source
from astra.models.spectrum import Spectrum
from astra.models.pipeline import PipelineOutputMixin
from astra.glossary import Glossary

class MDwarfType(BaseModel, PipelineOutputMixin):

    """M-dwarf type classifier."""
    
    sdss_id = ForeignKeyField(Source, null=True, index=True, lazy_load=False, help_text=Glossary.sdss_id)
    spectrum_id = ForeignKeyField(Spectrum, index=True, lazy_load=False, help_text=Glossary.spectrum_id)
    
    #> Astra Metadata
    task_id = AutoField(help_text=Glossary.task_id)
    v_astra = TextField(default=__version__, help_text=Glossary.v_astra)
    created = DateTimeField(default=datetime.datetime.now, help_text=Glossary.created)
    t_elapsed = FloatField(null=True, help_text=Glossary.t_elapsed)
    t_overhead = FloatField(null=True, help_text=Glossary.t_overhead)
    tag = TextField(default="", index=True, help_text=Glossary.tag)

    #> M Dwarf Type
    spectral_type = TextField(help_text=Glossary.spectral_type)
    sub_type = FloatField(help_text=Glossary.sub_type)
    rchi2 = FloatField(help_text=Glossary.rchi2)
    continuum = FloatField(null=True, help_text="Scalar continuum value used")
    result_flags = BitField(default=0, help_text=Glossary.result_flags)
    
    @hybrid_property
    def flag_bad(self):
        return (self.result_flags > 0)

    @flag_bad.expression
    def flag_bad(self):
        return (self.result_flags > 0)