import numpy as np
from peewee import (
    AutoField,
    FloatField,
    TextField,
    ForeignKeyField,
    BitField,
)

from astra import __version__
from astra.models.base import BaseModel
from astra.models.fields import BitField
from astra.models.source import Source
from astra.models.spectrum import Spectrum
from astra.models.pipeline import PipelineOutputMixin


class AstroNN(BaseModel, PipelineOutputMixin):

    """A result from the AstroNN pipeline."""

    source_id = ForeignKeyField(Source, null=True, index=True, lazy_load=False)
    spectrum_id = ForeignKeyField(Spectrum, index=True, lazy_load=False)
    
    #> Astra Metadata
    task_id = AutoField()
    v_astra = TextField(default=__version__)
    t_elapsed = FloatField(null=True)
    tag = TextField(default="", index=True)

    #> Task Parameters


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

    #> Summary Statistics
    result_flags = BitField(default=0)