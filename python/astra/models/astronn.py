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

from astra import __version__
from astra.models.base import BaseModel
from astra.models.fields import BitField
from astra.models.source import Source
from astra.models.spectrum import Spectrum
from astra.models.pipeline import PipelineOutputMixin
from playhouse.hybrid import hybrid_property
from peewee import fn

from astra.glossary import Glossary

class AstroNN(BaseModel, PipelineOutputMixin):

    """A result from the AstroNN pipeline."""

    source_pk = ForeignKeyField(Source, null=True, index=True, lazy_load=False, help_text=Glossary.source_pk)
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

    #> Stellar Labels
    teff = FloatField(null=True, help_text="Effective temperature [K]")
    e_teff = FloatField(null=True, help_text="Error on effective temperature [K]")
    logg = FloatField(null=True, help_text="Surface gravity [dex]")
    e_logg = FloatField(null=True, help_text="Error on surface gravity [dex]")
    c_h = FloatField(null=True, help_text="Carbon abundance [dex]")
    e_c_h = FloatField(null=True, help_text="Error on carbon abundance [dex]")
    c_1_h = FloatField(null=True, help_text="Carbon I abundance [dex]")
    e_c_1_h = FloatField(null=True, help_text="Error on carbon I abundance [dex]")
    n_h = FloatField(null=True, help_text="Nitrogen abundance [dex]")
    e_n_h = FloatField(null=True,  help_text="Error on nitrogen abundance [dex]")
    o_h = FloatField(null=True, help_text="Oxygen abundance [dex]")
    e_o_h = FloatField(null=True, help_text="Error on oxygen abundance [dex]")
    na_h = FloatField(null=True, help_text="Sodium abundance [dex]")
    e_na_h = FloatField(null=True,  help_text="Error on sodium abundance [dex]")
    mg_h = FloatField(null=True, help_text="Magnesium abundance [dex]")
    e_mg_h = FloatField(null=True, help_text="Error on magnesium abundance [dex]")
    al_h = FloatField(null=True, help_text="Aluminum abundance [dex]")
    e_al_h = FloatField(null=True, help_text="Error on aluminum abundance [dex]")
    si_h = FloatField(null=True, help_text="Silicon abundance [dex]")
    e_si_h = FloatField(null=True, help_text="Error on silicon abundance [dex]")
    p_h = FloatField(null=True, help_text="Phosphorus abundance [dex]")
    e_p_h = FloatField(null=True, help_text="Error on phosphorus abundance [dex]")
    s_h = FloatField(null=True, help_text="Sulfur abundance [dex]")
    e_s_h = FloatField(null=True, help_text="Error on sulfur abundance [dex]")
    k_h = FloatField(null=True, help_text="Potassium abundance [dex]")
    e_k_h = FloatField(null=True, help_text="Error on potassium abundance [dex]")
    ca_h = FloatField(null=True, help_text="Calcium abundance [dex]")
    e_ca_h = FloatField(null=True,  help_text="Error on calcium abundance [dex]")
    ti_h = FloatField(null=True, help_text="Titanium abundance [dex]")
    e_ti_h = FloatField(null=True, help_text="Error on titanium abundance [dex]")
    ti_2_h = FloatField(null=True, help_text="Titanium II abundance [dex]")
    e_ti_2_h = FloatField(null=True, help_text="Error on titanium II abundance [dex]")
    v_h = FloatField(null=True, help_text="Vanadium abundance [dex]")
    e_v_h = FloatField(null=True, help_text="Error on vanadium abundance [dex]")
    cr_h = FloatField(null=True, help_text="Chromium abundance [dex]")
    e_cr_h = FloatField(null=True, help_text="Error on chromium abundance [dex]")
    mn_h = FloatField(null=True, help_text="Manganese abundance [dex]")
    e_mn_h = FloatField(null=True, help_text="Error on manganese abundance [dex]")
    fe_h = FloatField(null=True, help_text="Iron abundance [dex]")
    e_fe_h = FloatField(null=True, help_text="Error on iron abundance [dex]")
    co_h = FloatField(null=True, help_text="Cobalt abundance [dex]")
    e_co_h = FloatField(null=True, help_text="Error on cobalt abundance [dex]")
    ni_h = FloatField(null=True, help_text="Nickel abundance [dex]")
    e_ni_h = FloatField(null=True, help_text="Error on nickel abundance [dex]")

    raw_teff = FloatField(null=True, help_text="Raw Effective temperature [K]")
    raw_e_teff = FloatField(null=True, help_text="Raw error on effective temperature [K]")
    raw_logg = FloatField(null=True, help_text="Raw surface gravity [dex]")
    raw_e_logg = FloatField(null=True, help_text="Raw error on surface gravity [dex]")
    raw_c_h = FloatField(null=True, help_text="Raw carbon abundance [dex]")
    raw_e_c_h = FloatField(null=True, help_text="Raw error on carbon abundance [dex]")
    raw_c_1_h = FloatField(null=True, help_text="Raw carbon I abundance [dex]")
    raw_e_c_1_h = FloatField(null=True, help_text="Raw error on carbon I abundance [dex]")
    raw_n_h = FloatField(null=True, help_text="Raw nitrogen abundance [dex]")
    raw_e_n_h = FloatField(null=True,  help_text="Raw error on nitrogen abundance [dex]")
    raw_o_h = FloatField(null=True, help_text="Raw oxygen abundance [dex]")
    raw_e_o_h = FloatField(null=True, help_text="Raw error on oxygen abundance [dex]")
    raw_na_h = FloatField(null=True, help_text="Raw sodium abundance [dex]")
    raw_e_na_h = FloatField(null=True,  help_text="Raw error on sodium abundance [dex]")
    raw_mg_h = FloatField(null=True, help_text="Raw magnesium abundance [dex]")
    raw_e_mg_h = FloatField(null=True, help_text="Raw error on magnesium abundance [dex]")
    raw_al_h = FloatField(null=True, help_text="Raw aluminum abundance [dex]")
    raw_e_al_h = FloatField(null=True, help_text="Raw error on aluminum abundance [dex]")
    raw_si_h = FloatField(null=True, help_text="Raw silicon abundance [dex]")
    raw_e_si_h = FloatField(null=True, help_text="Raw error on silicon abundance [dex]")
    raw_p_h = FloatField(null=True, help_text="Raw phosphorus abundance [dex]")
    raw_e_p_h = FloatField(null=True, help_text="Raw error on phosphorus abundance [dex]")
    raw_s_h = FloatField(null=True, help_text="Raw sulfur abundance [dex]")
    raw_e_s_h = FloatField(null=True, help_text="Raw error on sulfur abundance [dex]")
    raw_k_h = FloatField(null=True, help_text="Raw potassium abundance [dex]")
    raw_e_k_h = FloatField(null=True, help_text="Raw error on potassium abundance [dex]")
    raw_ca_h = FloatField(null=True, help_text="Raw calcium abundance [dex]")
    raw_e_ca_h = FloatField(null=True,  help_text="Raw error on calcium abundance [dex]")
    raw_ti_h = FloatField(null=True, help_text="Raw titanium abundance [dex]")
    raw_e_ti_h = FloatField(null=True, help_text="Raw error on titanium abundance [dex]")
    raw_ti_2_h = FloatField(null=True, help_text="Raw titanium II abundance [dex]")
    raw_e_ti_2_h = FloatField(null=True, help_text="Raw error on titanium II abundance [dex]")
    raw_v_h = FloatField(null=True, help_text="Raw vanadium abundance [dex]")
    raw_e_v_h = FloatField(null=True, help_text="Raw error on vanadium abundance [dex]")
    raw_cr_h = FloatField(null=True, help_text="Raw chromium abundance [dex]")
    raw_e_cr_h = FloatField(null=True, help_text="Raw error on chromium abundance [dex]")
    raw_mn_h = FloatField(null=True, help_text="Raw manganese abundance [dex]")
    raw_e_mn_h = FloatField(null=True, help_text="Raw error on manganese abundance [dex]")
    raw_fe_h = FloatField(null=True, help_text="Raw iron abundance [dex]")
    raw_e_fe_h = FloatField(null=True, help_text="Raw error on iron abundance [dex]")
    raw_co_h = FloatField(null=True, help_text="Raw cobalt abundance [dex]")
    raw_e_co_h = FloatField(null=True, help_text="Raw error on cobalt abundance [dex]")
    raw_ni_h = FloatField(null=True, help_text="Raw nickel abundance [dex]")
    raw_e_ni_h = FloatField(null=True, help_text="Raw error on nickel abundance [dex]")

    #> Summary Statistics
    result_flags = BitField(default=0, help_text="Flags describing the results")
    
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
        return ((self.flag_uncertain_logg & self.flag_uncertain_fe_h) | self.flag_uncertain_teff)

    @flag_bad.expression
    def flag_bad(self):
        return ((self.flag_uncertain_logg & self.flag_uncertain_fe_h) | self.flag_uncertain_teff)


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
    # Unset all in order to apply new
    (
        AstroNN
        .update(result_flags=AstroNN.flag_uncertain_fe_h.clear())
        .execute()
    )
    (
        AstroNN
        .update(result_flags=AstroNN.flag_uncertain_fe_h.set())
        .where(
            (AstroNN.raw_e_fe_h > 0.2)
        &   (fn.abs(AstroNN.raw_e_fe_h / AstroNN.raw_fe_h) > 0.12)
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
    
