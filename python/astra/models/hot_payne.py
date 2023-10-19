import numpy as np
import datetime
from peewee import (
    AutoField,
    FloatField,
    TextField,
    ForeignKeyField,
    DateTimeField,
    IntegerField
)

from astra import __version__
from astra.models.base import BaseModel
from astra.models.source import Source
from astra.models.fields import BitField
from astra.models.spectrum import Spectrum
from astra.models.pipeline import PipelineOutputMixin

from astra.glossary import Glossary
from playhouse.postgres_ext import ArrayField


class HotPayne(BaseModel, PipelineOutputMixin):

    """A result from the HotPayne pipeline."""

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
    v_micro = FloatField(null=True, help_text=Glossary.v_micro)    
    e_v_micro = FloatField(null=True, help_text=Glossary.e_v_micro)    
    v_sini = FloatField(null=True, help_text=Glossary.v_sini)
    e_v_sini = FloatField(null=True, help_text=Glossary.e_v_sini)
    
    #> Chemical Abundances
    he_h = FloatField(null=True, help_text=Glossary.he_h)
    e_he_h = FloatField(null=True, help_text=Glossary.e_he_h)
    c_fe = FloatField(null=True, help_text=Glossary.c_fe)
    e_c_fe = FloatField(null=True, help_text=Glossary.e_c_fe)
    n_fe = FloatField(null=True, help_text=Glossary.n_fe)
    e_n_fe = FloatField(null=True, help_text=Glossary.e_n_fe)
    o_fe = FloatField(null=True, help_text=Glossary.o_fe)
    e_o_fe = FloatField(null=True, help_text=Glossary.e_o_fe)
    si_fe = FloatField(null=True, help_text=Glossary.si_fe)
    e_si_fe = FloatField(null=True, help_text=Glossary.e_si_fe)
    s_fe = FloatField(null=True)
    e_s_fe = FloatField(null=True)

    #> Metadata    
    covar = ArrayField(FloatField, null=True, help_text=Glossary.covar)
    chi2 = FloatField(null=True, help_text=Glossary.chi2)
    
    # BOSS DRP redshift
    z = FloatField(null=True)
    z_err = FloatField(null=True)
    
    # Method
    result_flags = BitField(null=True, help_text=Glossary.result_flags)
    flag_results_with_hydrogen_mask = result_flags.flag(2**0, help_text="Results with hydrogen lines masked")
    flag_results_without_hydrogen_mask = result_flags.flag(2**1, help_text="Results without hydrogen lines masked")
