import datetime
import numpy as np
from peewee import (
    AutoField,
    IntegerField,
    FloatField,
    TextField,
    ForeignKeyField,
    DateTimeField,
    BooleanField,
    fn,
)
from astra.models.base import BaseModel
from astra.models.fields import BitField
from astra.models.spectrum import Spectrum
from astra.models.source import Source

from astra import __version__
from astra.utils import log, expand_path
from astra.glossary import Glossary



# The actual training set contains the continuum-normalized fluxes, labels, error arrays, etc.
# These two models are simply to link spectra to records in the database

'''
class TrainingSet(BaseModel):

    pk = AutoField()
    name = TextField(unique=True)
    description = TextField(null=True)
    n_labels = IntegerField(null=True)
    n_spectra = IntegerField(null=True)

    v_astra = TextField(default=__version__, help_text=Glossary.v_astra)
    created = DateTimeField(default=datetime.datetime.now, help_text=Glossary.created)

    @property
    def path(self):
        return f"$MWM_ASTRA/pipelines/TheCannon/{self.name}.pkl"
    
    @property
    def absolute_path(self):
        return expand_path(self.path)


class TrainingSetSpectrum(BaseModel):

    training_set_pk = ForeignKeyField(TrainingSet)
    spectrum_pk = ForeignKeyField(Spectrum)
    source_pk = ForeignKeyField(Source, null=True)
'''


class TheCannon(BaseModel):

    """Stellar labels estimated using The Cannon."""

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
    teff = FloatField(default=np.nan, help_text=Glossary.teff)
    e_teff = FloatField(default=np.nan, help_text=Glossary.e_teff)
    logg = FloatField(default=np.nan, help_text=Glossary.logg)
    e_logg = FloatField(default=np.nan, help_text=Glossary.e_logg)
    fe_h = FloatField(default=np.nan, help_text=Glossary.fe_h)
    e_fe_h = FloatField(default=np.nan, help_text=Glossary.e_fe_h)
    v_micro = FloatField(default=np.nan, help_text=Glossary.v_micro)
    e_v_micro = FloatField(default=np.nan, help_text=Glossary.e_v_micro)
    v_macro = FloatField(default=np.nan, help_text=Glossary.v_macro)
    e_v_macro = FloatField(default=np.nan, help_text=Glossary.e_v_macro)
    
    #> Chemical Abundances
    c_fe = FloatField(default=np.nan)
    e_c_fe = FloatField(default=np.nan)
    n_fe = FloatField(default=np.nan)
    e_n_fe = FloatField(default=np.nan)
    o_fe = FloatField(default=np.nan)
    e_o_fe = FloatField(default=np.nan)
    na_fe = FloatField(default=np.nan)
    e_na_fe = FloatField(default=np.nan)
    mg_fe = FloatField(default=np.nan)
    e_mg_fe = FloatField(default=np.nan)
    al_fe = FloatField(default=np.nan)
    e_al_fe = FloatField(default=np.nan)
    si_fe = FloatField(default=np.nan)
    e_si_fe = FloatField(default=np.nan)
    s_fe = FloatField(default=np.nan)
    e_s_fe = FloatField(default=np.nan)
    k_fe = FloatField(default=np.nan)
    e_k_fe = FloatField(default=np.nan)
    ca_fe = FloatField(default=np.nan)
    e_ca_fe = FloatField(default=np.nan)
    ti_fe = FloatField(default=np.nan)
    e_ti_fe = FloatField(default=np.nan)
    v_fe = FloatField(default=np.nan)
    e_v_fe = FloatField(default=np.nan)
    cr_fe = FloatField(default=np.nan)
    e_cr_fe = FloatField(default=np.nan)
    mn_fe = FloatField(default=np.nan)
    e_mn_fe = FloatField(default=np.nan)
    ni_fe = FloatField(default=np.nan)
    e_ni_fe = FloatField(default=np.nan)
    
    #> Summary Statistics
    chi2 = FloatField(default=np.nan)
    rchi2 = FloatField(default=np.nan)

    #> Metadata
    ier = IntegerField(default=-1)
    nfev = IntegerField(default=-1)
    x0_index = IntegerField(default=-1)
    result_flags = BitField(default=0)
    flag_fitting_failure = result_flags.flag(2**0, "Fitting failure")
    
    @property
    def intermediate_output_path(self):
        parts = f"{self.source_pk}"
        group_dir = f"{parts[-4:-2]}/{parts[-2:]}"
        return f"$MWM_ASTRA/{self.v_astra}/pipelines/TheCannon/{group_dir}/{self.source_pk}-{self.spectrum_pk}.pkl"
    