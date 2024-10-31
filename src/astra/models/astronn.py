import numpy as np
import datetime
from peewee import fn
from astra import __version__
from astra.fields import (AutoField, BitField, FloatField, TextField, ForeignKeyField, DateTimeField)
from astra.models.base import BaseModel
from astra.models.source import Source
from astra.models.spectrum import Spectrum
from playhouse.hybrid import hybrid_property

from astra.models.pipeline import PipelineOutputModel

class AstroNN(PipelineOutputModel):

    """A result from the AstroNN pipeline."""

    #> Stellar Labels
    teff = FloatField(null=True)
    e_teff = FloatField(null=True)
    logg = FloatField(null=True)
    e_logg = FloatField(null=True)
    c_h = FloatField(null=True)
    e_c_h = FloatField(null=True)
    c_1_h = FloatField(null=True)
    e_c_1_h = FloatField(null=True)
    n_h = FloatField(null=True)
    e_n_h = FloatField(null=True)
    o_h = FloatField(null=True)
    e_o_h = FloatField(null=True)
    na_h = FloatField(null=True)
    e_na_h = FloatField(null=True)
    mg_h = FloatField(null=True)
    e_mg_h = FloatField(null=True)
    al_h = FloatField(null=True)
    e_al_h = FloatField(null=True)
    si_h = FloatField(null=True)
    e_si_h = FloatField(null=True)
    p_h = FloatField(null=True)
    e_p_h = FloatField(null=True)
    s_h = FloatField(null=True)
    e_s_h = FloatField(null=True)
    k_h = FloatField(null=True)
    e_k_h = FloatField(null=True)
    ca_h = FloatField(null=True)
    e_ca_h = FloatField(null=True)
    ti_h = FloatField(null=True)
    e_ti_h = FloatField(null=True)
    ti_2_h = FloatField(null=True)
    e_ti_2_h = FloatField(null=True)
    v_h = FloatField(null=True)
    e_v_h = FloatField(null=True)
    cr_h = FloatField(null=True)
    e_cr_h = FloatField(null=True)
    mn_h = FloatField(null=True)
    e_mn_h = FloatField(null=True)
    fe_h = FloatField(null=True)
    e_fe_h = FloatField(null=True)
    co_h = FloatField(null=True)
    e_co_h = FloatField(null=True)
    ni_h = FloatField(null=True)
    e_ni_h = FloatField(null=True)

    #> Raw Stellar Labels
    raw_teff = FloatField(null=True)
    raw_e_teff = FloatField(null=True)
    raw_logg = FloatField(null=True)
    raw_e_logg = FloatField(null=True)
    raw_c_h = FloatField(null=True)
    raw_e_c_h = FloatField(null=True)
    raw_c_1_h = FloatField(null=True)
    raw_e_c_1_h = FloatField(null=True)
    raw_n_h = FloatField(null=True)
    raw_e_n_h = FloatField(null=True)
    raw_o_h = FloatField(null=True)
    raw_e_o_h = FloatField(null=True)
    raw_na_h = FloatField(null=True)
    raw_e_na_h = FloatField(null=True)
    raw_mg_h = FloatField(null=True)
    raw_e_mg_h = FloatField(null=True)
    raw_al_h = FloatField(null=True)
    raw_e_al_h = FloatField(null=True)
    raw_si_h = FloatField(null=True)
    raw_e_si_h = FloatField(null=True)
    raw_p_h = FloatField(null=True)
    raw_e_p_h = FloatField(null=True)
    raw_s_h = FloatField(null=True)
    raw_e_s_h = FloatField(null=True)
    raw_k_h = FloatField(null=True)
    raw_e_k_h = FloatField(null=True)
    raw_ca_h = FloatField(null=True)
    raw_e_ca_h = FloatField(null=True)
    raw_ti_h = FloatField(null=True)
    raw_e_ti_h = FloatField(null=True)
    raw_ti_2_h = FloatField(null=True)
    raw_e_ti_2_h = FloatField(null=True)
    raw_v_h = FloatField(null=True)
    raw_e_v_h = FloatField(null=True)
    raw_cr_h = FloatField(null=True)
    raw_e_cr_h = FloatField(null=True)
    raw_mn_h = FloatField(null=True)
    raw_e_mn_h = FloatField(null=True)
    raw_fe_h = FloatField(null=True)
    raw_e_fe_h = FloatField(null=True)
    raw_co_h = FloatField(null=True)
    raw_e_co_h = FloatField(null=True)
    raw_ni_h = FloatField(null=True)
    raw_e_ni_h = FloatField(null=True)

    #> Summary Statistics
    result_flags = BitField(default=0)
    
    #> Flag definitions
    flag_uncertain_logg = result_flags.flag(2**0, help_text="Surface gravity is uncertain (`e_logg` > 0.2 and abs(e_logg/logg) > 0.075)")
    flag_uncertain_teff = result_flags.flag(2**1, help_text="Effective temperature is uncertain (`e_teff` > 300)")
    flag_uncertain_fe_h = result_flags.flag(2**2, help_text="Iron abundance is uncertain (`abs(e_fe_h/fe_h) > 0.12`)")
    flag_no_result = result_flags.flag(2**11, help_text="Exception raised when loading spectra")

    
    @hybrid_property
    def flag_warn(self):
        return (self.result_flags > 0)

    @flag_warn.expression
    def flag_warn(self):
        return (self.result_flags > 0)
        
    @hybrid_property
    def flag_bad(self):
        return (self.flag_uncertain_logg & self.flag_uncertain_fe_h & self.flag_uncertain_teff)

    @flag_bad.expression
    def flag_bad(self):
        return (self.flag_uncertain_logg & self.flag_uncertain_fe_h & self.flag_uncertain_teff)


def apply_flags():
    (
        AstroNN
        .update(result_flags=AstroNN.flag_uncertain_logg.set())
        .where(
            (AstroNN.raw_e_logg > 0.2)
        &   (fn.abs(AstroNN.raw_e_logg / AstroNN.raw_logg) > 0.075)
        )
        .execute()
    )
    (
        AstroNN
        .update(result_flags=AstroNN.flag_uncertain_teff.set())
        .where(
            (AstroNN.raw_e_teff > 300)
        )
        .execute()
    )    
    (
        AstroNN
        .update(result_flags=AstroNN.flag_uncertain_fe_h.set())
        .where(
            (fn.abs(AstroNN.raw_e_fe_h / AstroNN.raw_fe_h) > 0.12)
        )
        .execute()
    )

    
def apply_noise_model():
    
    import pickle
    from astra.utils import expand_path
    with open(expand_path(f"$MWM_ASTRA/{__version__}/aux/AstroNN_corrections.pkl"), "rb") as fp:
        corrections, reference = pickle.load(fp)

    update_kwds = {}
    for label_name, kwds in corrections.items():
        offset, scale = kwds["offset"], kwds["scale"]
        update_kwds[f"e_{label_name}"] = scale * getattr(AstroNN, f"raw_e_{label_name}") + offset
        
    (
        AstroNN
        .update(**update_kwds)
        .execute()
    )
    
