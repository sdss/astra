import numpy as np
import datetime
from playhouse.hybrid import hybrid_property

from astra import __version__
from astra.fields import (AutoField, FloatField, TextField, ForeignKeyField, BitField, DateTimeField)
from astra.models.base import BaseModel
from astra.models.source import Source
from astra.models.spectrum import Spectrum
from astra.models.pipeline import PipelineOutputMixin

class BossNet(BaseModel, PipelineOutputMixin):

    """A result from the BOSSNet pipeline."""

    source_pk = ForeignKeyField(Source, null=True, index=True, lazy_load=False)
    spectrum_pk = ForeignKeyField(
        Spectrum, 
        index=True, 
        lazy_load=False,
    )
    
    #> Astra Metadata
    task_pk = AutoField()
    v_astra = TextField(default=__version__)
    created = DateTimeField(default=datetime.datetime.now)
    modified = DateTimeField(default=datetime.datetime.now)
    t_elapsed = FloatField(null=True)
    t_overhead = FloatField(null=True)
    tag = TextField(default="", index=True)
    
    #> Stellar Parameters
    teff = FloatField(null=True)
    e_teff = FloatField(null=True)
    logg = FloatField(null=True)
    e_logg = FloatField(null=True)
    fe_h = FloatField(null=True)
    e_fe_h = FloatField(null=True)
    result_flags = BitField(default=0)
    flag_runtime_exception = result_flags.flag(2**0, help_text="Exception occurred at runtime")
    flag_unreliable_teff = result_flags.flag(2**1, help_text="`teff` is outside the range of 1700 K and 100,000 K")
    flag_unreliable_logg = result_flags.flag(2**2, help_text="`logg` is outside the range of -1 and 10")
    flag_unreliable_fe_h = result_flags.flag(2**3, help_text="`teff` < 3200 K or `logg` > 5 or `fe_h` is outside the range of -4 and 2")
    flag_suspicious_fe_h = result_flags.flag(2**4, help_text="[Fe/H] below `teff` < 3900 K and with 3 < `logg` < 6 may be suspicious")
    
    bn_v_r = FloatField(null=True)
    e_bn_v_r = FloatField(null=True)
    
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