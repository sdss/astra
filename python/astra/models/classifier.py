from peewee import (
    AutoField,
    FloatField,
    TextField,
    ForeignKeyField,
    BitField,
    IntegerField,
    DateTimeField
)
import datetime
from astra import __version__
from astra.models.base import BaseModel
from astra.models.fields import BitField
from astra.models.source import Source
from astra.models.spectrum import Spectrum
from astra.models.pipeline import PipelineOutputMixin
from astra.glossary import Glossary

SMALL = -1e+20

class SpectrumClassification(BaseModel, PipelineOutputMixin):

    """A spectrum-level classification."""
    
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
    
    #> Classification
    p_cv = FloatField(default=0)
    lp_cv = FloatField(default=SMALL)
    p_fgkm = FloatField(default=0)
    lp_fgkm = FloatField(default=SMALL)
    p_oba = FloatField(default=0)
    lp_oba = FloatField(default=SMALL)
    p_wd = FloatField(default=0)
    lp_wd = FloatField(default=SMALL)
    p_sb2 = FloatField(default=0)
    lp_sb2 = FloatField(default=SMALL)
    p_yso = FloatField(default=0)
    lp_yso = FloatField(default=SMALL)
    classification_flags = BitField(default=0)

    flag_most_likely_cv = classification_flags.flag(2**0, "Most likely is a cataclysmic variable")
    flag_most_likely_fgkm = classification_flags.flag(2**1, "Most likely is a FGKM-type star")
    flag_most_likely_oba = classification_flags.flag(2**2, "Most likely is a OBA-type star")
    flag_most_likely_wd = classification_flags.flag(2**3, "Most likely is a white dwarf")
    flag_most_likely_sb2 = classification_flags.flag(2**4, "Most likely is a spectroscopic binary (SB2)")
    flag_most_likely_yso = classification_flags.flag(2**5, "Most likely is a young stellar object")

'''    

class SourceClassification(BaseModel, PipelineOutputMixin):

    """A source-level classification."""

    sdss_id = ForeignKeyField(Source, index=True)
    
    #> Astra Metadata
    task_id = AutoField()
    v_astra = TextField(default=__version__)
    t_elapsed = FloatField(null=True)
    tag = TextField(default="", index=True)

    #> Spectrum-level Classifications
    num_apogee_classifications = IntegerField(default=0)
    num_boss_classifications = IntegerField(default=0)

    #> Source Classification
    p_cv = FloatField(default=0)
    lp_cv = FloatField(default=SMALL)
    p_fgkm = FloatField(default=0)
    lp_fgkm = FloatField(default=SMALL)
    p_oba = FloatField(default=0)
    lp_oba = FloatField(default=SMALL)
    p_wd = FloatField(default=0)
    lp_wd = FloatField(default=SMALL)
    p_sb2 = FloatField(default=0)
    lp_sb2 = FloatField(default=SMALL)
    p_yso = FloatField(default=0)
    lp_yso = FloatField(default=SMALL)
    classification_flags = BitField(default=0)
    
    flag_most_likely_cv = classification_flags.flag(2**0, "Most likely is a cataclysmic variable")
    flag_most_likely_fgkm = classification_flags.flag(2**1, "Most likely is a FGKM-type star")
    flag_most_likely_oba = classification_flags.flag(2**2, "Most likely is a OBA-type star")
    flag_most_likely_wd = classification_flags.flag(2**3, "Most likely is a white dwarf")
    flag_most_likely_sb2 = classification_flags.flag(2**4, "Most likely is a spectroscopic binary (SB2)")
    flag_most_likely_yso = classification_flags.flag(2**5, "Most likely is a young stellar object")
'''