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

from astra.glossary import Glossary

class AstroNN(BaseModel, PipelineOutputMixin):

    """A result from the AstroNN pipeline."""

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

    #> Summary Statistics
    result_flags = BitField(default=0, help_text="Flags describing the results")

    #> Flag definitions
    flag_no_result = result_flags.flag(2**11, help_text="Exception raised when loading spectra")