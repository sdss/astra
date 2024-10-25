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
    he_h = FloatField(null=True, help_text=Glossary.he_h)
    e_he_h = FloatField(null=True, help_text=Glossary.e_he_h)
    c_h = FloatField(null=True, help_text=Glossary.c_h)
    e_c_h = FloatField(null=True, help_text=Glossary.e_c_h)
    n_h = FloatField(null=True, help_text=Glossary.n_h)
    e_n_h = FloatField(null=True, help_text=Glossary.e_n_h)
    o_h = FloatField(null=True, help_text=Glossary.o_h)
    e_o_h = FloatField(null=True, help_text=Glossary.e_o_h)
    si_h = FloatField(null=True, help_text=Glossary.si_h)
    e_si_h = FloatField(null=True, help_text=Glossary.e_si_h)
    s_h = FloatField(null=True, help_text=Glossary.s_h)
    e_s_h = FloatField(null=True, help_text=Glossary.e_s_h)
    covar = ArrayField(FloatField, help_text=Glossary.covar)
    chi2 = FloatField(null=True, help_text=Glossary.chi2)
    result_flags = IntegerField(default=0, help_text="Result flags")
    
    #> Stellar Parameters from Full-Spectrum Fit
    teff_fullspec = FloatField(null=True, help_text=Glossary.teff)
    e_teff_fullspec = FloatField(null=True, help_text=Glossary.e_teff)
    logg_fullspec = FloatField(null=True, help_text=Glossary.logg)
    e_logg_fullspec = FloatField(null=True, help_text=Glossary.e_logg)
    fe_h_fullspec = FloatField(null=True, help_text=Glossary.fe_h)
    e_fe_h_fullspec = FloatField(null=True, help_text=Glossary.e_fe_h)
    v_micro_fullspec = FloatField(null=True, help_text=Glossary.v_micro)
    e_v_micro_fullspec = FloatField(null=True, help_text=Glossary.e_v_micro)
    v_sini_fullspec = FloatField(null=True, help_text=Glossary.v_sini)
    e_v_sini_fullspec = FloatField(null=True, help_text=Glossary.e_v_sini)
    he_h_fullspec = FloatField(null=True, help_text=Glossary.he_h)
    e_he_h_fullspec = FloatField(null=True, help_text=Glossary.e_he_h)
    c_h_fullspec = FloatField(null=True, help_text=Glossary.c_h)
    e_c_h_fullspec = FloatField(null=True, help_text=Glossary.e_c_h)
    n_h_fullspec = FloatField(null=True, help_text=Glossary.n_h)
    e_n_h_fullspec = FloatField(null=True, help_text=Glossary.e_n_h)
    o_h_fullspec = FloatField(null=True, help_text=Glossary.o_h)
    e_o_h_fullspec = FloatField(null=True, help_text=Glossary.e_o_h)
    si_h_fullspec = FloatField(null=True, help_text=Glossary.si_h)
    e_si_h_fullspec = FloatField(null=True, help_text=Glossary.e_si_h)
    s_h_fullspec = FloatField(null=True, help_text=Glossary.s_h)
    e_s_h_fullspec = FloatField(null=True, help_text=Glossary.e_s_h)
    covar_fullspec = ArrayField(FloatField, help_text=Glossary.covar)
    chi2_fullspec = FloatField(null=True, help_text=Glossary.chi2)
    
    #> Stellar Parameters with H-lines Masked
    teff_hmasked = FloatField(null=True, help_text=Glossary.teff)
    e_teff_hmasked = FloatField(null=True, help_text=Glossary.e_teff)
    logg_hmasked = FloatField(null=True, help_text=Glossary.logg)
    e_logg_hmasked = FloatField(null=True, help_text=Glossary.e_logg)
    fe_h_hmasked = FloatField(null=True, help_text=Glossary.fe_h)
    e_fe_h_hmasked = FloatField(null=True, help_text=Glossary.e_fe_h)
    v_micro_hmasked = FloatField(null=True, help_text=Glossary.v_micro)
    e_v_micro_hmasked = FloatField(null=True, help_text=Glossary.e_v_micro)
    v_sini_hmasked = FloatField(null=True, help_text=Glossary.v_sini)
    e_v_sini_hmasked = FloatField(null=True, help_text=Glossary.e_v_sini)
    he_h_hmasked = FloatField(null=True, help_text=Glossary.he_h)
    e_he_h_hmasked = FloatField(null=True, help_text=Glossary.e_he_h)
    c_h_hmasked = FloatField(null=True, help_text=Glossary.c_h)
    e_c_h_hmasked = FloatField(null=True, help_text=Glossary.e_c_h)
    n_h_hmasked = FloatField(null=True, help_text=Glossary.n_h)
    e_n_h_hmasked = FloatField(null=True, help_text=Glossary.e_n_h)
    o_h_hmasked = FloatField(null=True, help_text=Glossary.o_h)
    e_o_h_hmasked = FloatField(null=True, help_text=Glossary.e_o_h)
    si_h_hmasked = FloatField(null=True, help_text=Glossary.si_h)
    e_si_h_hmasked = FloatField(null=True, help_text=Glossary.e_si_h)
    s_h_hmasked = FloatField(null=True, help_text=Glossary.s_h)
    e_s_h_hmasked = FloatField(null=True, help_text=Glossary.e_s_h)
    covar_hmasked = ArrayField(FloatField, help_text=Glossary.covar)
    chi2_hmasked = FloatField(null=True, help_text=Glossary.chi2)    
    
    
    #> Formal uncertainties
    raw_e_teff = FloatField(null=True, help_text=Glossary.e_teff)
    raw_e_logg = FloatField(null=True, help_text=Glossary.e_logg)
    raw_e_fe_h = FloatField(null=True, help_text=Glossary.e_fe_h)
    raw_e_v_micro = FloatField(null=True, help_text=Glossary.e_v_micro)
    raw_e_v_sini = FloatField(null=True, help_text=Glossary.e_v_sini)
    raw_e_he_h = FloatField(null=True, help_text=Glossary.e_he_h)
    raw_e_c_h = FloatField(null=True, help_text=Glossary.e_c_h)
    raw_e_n_h = FloatField(null=True, help_text=Glossary.e_n_h)
    raw_e_o_h = FloatField(null=True, help_text=Glossary.e_o_h)
    raw_e_si_h = FloatField(null=True, help_text=Glossary.e_si_h)
    raw_e_s_h = FloatField(null=True, help_text=Glossary.e_s_h)
    raw_e_teff_fullspec = FloatField(null=True, help_text=Glossary.e_teff)
    raw_e_logg_fullspec = FloatField(null=True, help_text=Glossary.e_logg)
    raw_e_fe_h_fullspec = FloatField(null=True, help_text=Glossary.e_fe_h)
    raw_e_v_micro_fullspec = FloatField(null=True, help_text=Glossary.e_v_micro)
    raw_e_v_sini_fullspec = FloatField(null=True, help_text=Glossary.e_v_sini)
    raw_e_he_h_fullspec = FloatField(null=True, help_text=Glossary.e_he_h)
    raw_e_c_h_fullspec = FloatField(null=True, help_text=Glossary.e_c_h)
    raw_e_n_h_fullspec = FloatField(null=True, help_text=Glossary.e_n_h)
    raw_e_o_h_fullspec = FloatField(null=True, help_text=Glossary.e_o_h)
    raw_e_si_h_fullspec = FloatField(null=True, help_text=Glossary.e_si_h)
    raw_e_s_h_fullspec = FloatField(null=True, help_text=Glossary.e_s_h)
    raw_e_teff_hmasked = FloatField(null=True, help_text=Glossary.e_teff)
    raw_e_logg_hmasked = FloatField(null=True, help_text=Glossary.e_logg)
    raw_e_fe_h_hmasked = FloatField(null=True, help_text=Glossary.e_fe_h)
    raw_e_v_micro_hmasked = FloatField(null=True, help_text=Glossary.e_v_micro)
    raw_e_v_sini_hmasked = FloatField(null=True, help_text=Glossary.e_v_sini)
    raw_e_he_h_hmasked = FloatField(null=True, help_text=Glossary.e_he_h)
    raw_e_c_h_hmasked = FloatField(null=True, help_text=Glossary.e_c_h)
    raw_e_n_h_hmasked = FloatField(null=True, help_text=Glossary.e_n_h)
    raw_e_o_h_hmasked = FloatField(null=True, help_text=Glossary.e_o_h)
    raw_e_si_h_hmasked = FloatField(null=True, help_text=Glossary.e_si_h)
    raw_e_s_h_hmasked = FloatField(null=True, help_text=Glossary.e_s_h)


def set_formal_errors():
    (
        HotPayne
        .update(
            raw_e_teff=HotPayne.e_teff,
            raw_e_logg=HotPayne.e_logg,
            raw_e_fe_h=HotPayne.e_fe_h,
            raw_e_v_micro=HotPayne.e_v_micro,
            raw_e_v_sini=HotPayne.e_v_sini,
            raw_e_he_h=HotPayne.e_he_h,
            raw_e_c_h=HotPayne.e_c_h,
            raw_e_n_h=HotPayne.e_n_h,
            raw_e_o_h=HotPayne.e_o_h,
            raw_e_si_h=HotPayne.e_si_h,
            raw_e_s_h=HotPayne.e_s_h,
            raw_e_teff_fullspec=HotPayne.e_teff_fullspec,
            raw_e_logg_fullspec=HotPayne.e_logg_fullspec,
            raw_e_fe_h_fullspec=HotPayne.e_fe_h_fullspec,
            raw_e_v_micro_fullspec=HotPayne.e_v_micro_fullspec,
            raw_e_v_sini_fullspec=HotPayne.e_v_sini_fullspec,
            raw_e_he_h_fullspec=HotPayne.e_he_h_fullspec,
            raw_e_c_h_fullspec=HotPayne.e_c_h_fullspec,
            raw_e_n_h_fullspec=HotPayne.e_n_h_fullspec,
            raw_e_o_h_fullspec=HotPayne.e_o_h_fullspec,
            raw_e_si_h_fullspec=HotPayne.e_si_h_fullspec,
            raw_e_s_h_fullspec=HotPayne.e_s_h_fullspec,
            raw_e_teff_hmasked=HotPayne.e_teff_hmasked,
            raw_e_logg_hmasked=HotPayne.e_logg_hmasked,
            raw_e_fe_h_hmasked=HotPayne.e_fe_h_hmasked,
            raw_e_v_micro_hmasked=HotPayne.e_v_micro_hmasked,
            raw_e_v_sini_hmasked=HotPayne.e_v_sini_hmasked,
            raw_e_he_h_hmasked=HotPayne.e_he_h_hmasked,
            raw_e_c_h_hmasked=HotPayne.e_c_h_hmasked,
            raw_e_n_h_hmasked=HotPayne.e_n_h_hmasked,
            raw_e_o_h_hmasked=HotPayne.e_o_h_hmasked,
            raw_e_si_h_hmasked=HotPayne.e_si_h_hmasked,
            raw_e_s_h_hmasked=HotPayne.e_s_h_hmasked,
        )
        .execute()
    )
    
def apply_noise_model():
    
    with open(expand_path(f"$MWM_ASTRA/{__version__}/aux/HotPayne_corrections.pkl"), "rb") as fp:
        corrections, reference = pickle.load(fp)

    update_kwds = {}
    for label_name, kwds in corrections.items():
        offset, scale = kwds["offset"], kwds["scale"]
        update_kwds[f"e_{label_name}"] = scale * getattr(HotPayne, f"raw_e_{label_name}") + offset
        
    (
        HotPayne
        .update(**update_kwds)
        .execute()
    )
    
            