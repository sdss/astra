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
from playhouse.hybrid import hybrid_property

from astra import __version__
from astra.models.base import BaseModel
from astra.models.fields import BitField
from astra.models.source import Source
from astra.models.spectrum import Spectrum
from astra.models.pipeline import PipelineOutputMixin

from astra.glossary import Glossary

class BossNet(BaseModel, PipelineOutputMixin):

    """A result from the BOSSNet pipeline."""

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
    
    #> Stellar Parameters
    teff = FloatField(null=True, help_text=Glossary.teff)
    e_teff = FloatField(null=True, help_text=Glossary.e_teff)
    logg = FloatField(null=True, help_text=Glossary.logg)
    e_logg = FloatField(null=True, help_text=Glossary.e_logg)
    fe_h = FloatField(null=True, help_text=Glossary.fe_h)
    e_fe_h = FloatField(null=True, help_text=Glossary.e_fe_h)
    result_flags = BitField(default=0, help_text=Glossary.result_flags)
    flag_runtime_exception = result_flags.flag(2**0, help_text="Exception occurred at runtime")
    flag_unreliable_teff = result_flags.flag(2**1, help_text="`teff` is outside the range of 1700 K and 100,000 K")
    flag_unreliable_logg = result_flags.flag(2**2, help_text="`logg` is outside the range of -1 and 10")
    flag_unreliable_fe_h = result_flags.flag(2**3, help_text="`teff` < 3200 K or `logg` > 5 or `fe_h` is outside the range of -4 and 2")
    flag_suspicious_fe_h = result_flags.flag(2**4, help_text="[Fe/H] below `teff` < 3900 K and with 3 < `logg` < 6 may be suspicious")
    
    v_rad = FloatField(null=True, help_text=Glossary.v_rad)
    e_v_rad = FloatField(null=True, help_text=Glossary.e_v_rad)
    
    @hybrid_property
    def flag_warn(self):
        return (
            self.flag_unreliable_fe_h
        |   self.flag_suspicious_fe_h
        )
    
    @flag_warn.expression
    def flag_warn(self):
        return (
            self.flag_unreliable_fe_h
        |   self.flag_suspicious_fe_h
        )

    @hybrid_property
    def flag_bad(self):
        return (
            self.flag_unreliable_teff
        |   self.flag_unreliable_logg
        |   self.flag_unreliable_fe_h
        |   self.flag_runtime_exception
    )


    @flag_bad.expression
    def flag_bad(self):
        return (
            self.flag_unreliable_teff
        |   self.flag_unreliable_logg
        |   self.flag_unreliable_fe_h
        |   self.flag_runtime_exception
    )
    

# TODO: Move this to happen at runtime
def apply_result_flags():
    
    (
        BossNet
        .update(result_flags=BossNet.flag_runtime_exception.set())
        .where(BossNet.teff.is_null())
        .execute()
    )
    
    (
        BossNet
        .update(result_flags=BossNet.flag_unreliable_teff.set())
        .where(
            (BossNet.teff < 1700) | (BossNet.teff > 100_000)
        )
        .execute()
    )    
    (
        BossNet
        .update(result_flags=BossNet.flag_unreliable_logg.set())
        .where(
            (BossNet.logg < -1) | (BossNet.logg > 10)
        )
        .execute()
    )    
    (
        BossNet
        .update(result_flags=BossNet.flag_unreliable_fe_h.set())
        .where(
            (BossNet.teff < 3200)
        |   (BossNet.logg > 5)
        |   (BossNet.fe_h < -4)
        |   (BossNet.fe_h > 2)
        )
        .execute()
    )
    (
        BossNet
        .update(result_flags=BossNet.flag_suspicious_fe_h.set())
        .where(
            (BossNet.teff < 3900)
        &   ((6 > BossNet.logg) & (BossNet.logg > 3))
        )
        .execute()
    )