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
import pickle
from astra.utils import expand_path
from astra.models.fields import BitField, PixelArray, BasePixelArrayAccessor, LogLambdaArrayAccessor
from astra.models.base import BaseModel
from astra.models.fields import BitField
from astra.models.spectrum import Spectrum
from astra.models.source import Source

from astra import __version__
from astra.utils import log, expand_path
from astra.glossary import Glossary




class TheCannonPixelArrayAccessor(BasePixelArrayAccessor):
    
    def __get__(self, instance, instance_type=None):
        if instance is not None:
            try:
                return instance.__pixel_data__[self.name]
            except (AttributeError, KeyError):
                # Load them all.
                instance.__pixel_data__ = {}

                with open(expand_path(instance.intermediate_output_path), "rb") as fp:
                    continuum, rectified_model_flux = pickle.load(fp)
                
                continuum = continuum.flatten()
                rectified_model_flux = rectified_model_flux.flatten()
                model_flux = continuum * rectified_model_flux
                
                instance.__pixel_data__.setdefault("continuum", continuum)
                instance.__pixel_data__.setdefault("model_flux", model_flux)
                instance.__pixel_data__.setdefault("rectified_model_flux", rectified_model_flux)
                
                return instance.__pixel_data__[self.name]

        return self.field


class TheCannonPixelArray(PixelArray):
    
    def __init__(self, ext=None, column_name=None, transform=None, accessor_class=TheCannonPixelArrayAccessor, help_text=None, **kwargs):
        super(TheCannonPixelArray, self).__init__(
            ext=ext,
            column_name=column_name,
            transform=transform,
            accessor_class=accessor_class,
            help_text=help_text,
            **kwargs
        )
        



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
    c_fe = FloatField(default=np.nan, help_text="[C/Fe] abundance ratio")
    e_c_fe = FloatField(default=np.nan, help_text="Error on [C/Fe] abundance ratio")
    n_fe = FloatField(default=np.nan, help_text="[N/Fe] abundance ratio")
    e_n_fe = FloatField(default=np.nan, help_text="Error on [N/Fe] abundance ratio")
    o_fe = FloatField(default=np.nan, help_text="[O/Fe] abundance ratio")
    e_o_fe = FloatField(default=np.nan, help_text="Error on [O/Fe] abundance ratio")
    na_fe = FloatField(default=np.nan, help_text="[Na/Fe] abundance ratio")
    e_na_fe = FloatField(default=np.nan, help_text="Error on [Na/Fe] abundance ratio")
    mg_fe = FloatField(default=np.nan, help_text="[Mg/Fe] abundance ratio")
    e_mg_fe = FloatField(default=np.nan, help_text="Error on [Mg/Fe] abundance ratio")
    al_fe = FloatField(default=np.nan, help_text="[Al/Fe] abundance ratio")
    e_al_fe = FloatField(default=np.nan, help_text="Error on [Al/Fe] abundance ratio")
    si_fe = FloatField(default=np.nan, help_text="[Si/Fe] abundance ratio")
    e_si_fe = FloatField(default=np.nan, help_text="Error on [Si/Fe] abundance ratio")
    s_fe = FloatField(default=np.nan, help_text="[S/Fe] abundance ratio")
    e_s_fe = FloatField(default=np.nan, help_text="Error on [S/Fe] abundance ratio")
    k_fe = FloatField(default=np.nan, help_text="[K/Fe] abundance ratio")
    e_k_fe = FloatField(default=np.nan, help_text="Error on [K/Fe] abundance ratio")
    ca_fe = FloatField(default=np.nan, help_text="[Ca/Fe] abundance ratio")
    e_ca_fe = FloatField(default=np.nan, help_text="Error on [Ca/Fe] abundance ratio")
    ti_fe = FloatField(default=np.nan, help_text="[Ti/Fe] abundance ratio")
    e_ti_fe = FloatField(default=np.nan, help_text="Error on [Ti/Fe] abundance ratio")
    v_fe = FloatField(default=np.nan, help_text="[V/Fe] abundance ratio")
    e_v_fe = FloatField(default=np.nan, help_text="Error on [V/Fe] abundance ratio")
    cr_fe = FloatField(default=np.nan, help_text="[Cr/Fe] abundance ratio")
    e_cr_fe = FloatField(default=np.nan, help_text="Error on [Cr/Fe] abundance ratio")
    mn_fe = FloatField(default=np.nan, help_text="[Mn/Fe] abundance ratio")
    e_mn_fe = FloatField(default=np.nan, help_text="Error on [Mn/Fe] abundance ratio")
    ni_fe = FloatField(default=np.nan, help_text="[Ni/Fe] abundance ratio")
    e_ni_fe = FloatField(default=np.nan, help_text="Error on [Ni/Fe] abundance ratio")
    
    #> Summary Statistics
    chi2 = FloatField(default=np.nan, help_text=Glossary.chi2)
    rchi2 = FloatField(default=np.nan, help_text=Glossary.rchi2)

    #> Formal uncertainties
    raw_e_teff = FloatField(default=np.nan, help_text=Glossary.raw_e_teff)
    raw_e_logg = FloatField(default=np.nan, help_text=Glossary.raw_e_logg)
    raw_e_fe_h = FloatField(default=np.nan, help_text=Glossary.raw_e_fe_h)
    raw_e_v_micro = FloatField(default=np.nan, help_text=Glossary.raw_e_v_micro)
    raw_e_v_macro = FloatField(default=np.nan, help_text=Glossary.raw_e_v_macro)
    raw_e_c_fe = FloatField(default=np.nan, help_text="Raw error on [C/Fe] abundance ratio")
    raw_e_n_fe = FloatField(default=np.nan, help_text="Raw error on [N/Fe] abundance ratio")
    raw_e_o_fe = FloatField(default=np.nan, help_text="Raw error on [O/Fe] abundance ratio")
    raw_e_na_fe = FloatField(default=np.nan, help_text="Raw error on [Na/Fe] abundance ratio")
    raw_e_mg_fe = FloatField(default=np.nan, help_text="Raw error on [Mg/Fe] abundance ratio")
    raw_e_al_fe = FloatField(default=np.nan, help_text="Raw error on [Al/Fe] abundance ratio")
    raw_e_si_fe = FloatField(default=np.nan, help_text="Raw error on [Si/Fe] abundance ratio")
    raw_e_s_fe = FloatField(default=np.nan, help_text="Raw error on [S/Fe] abundance ratio")
    raw_e_k_fe = FloatField(default=np.nan, help_text="Raw error on [K/Fe] abundance ratio")
    raw_e_ca_fe = FloatField(default=np.nan, help_text="Raw error on [Ca/Fe] abundance ratio")
    raw_e_ti_fe = FloatField(default=np.nan, help_text="Raw error on [Ti/Fe] abundance ratio")
    raw_e_v_fe = FloatField(default=np.nan, help_text="Raw error on [V/Fe] abundance ratio")
    raw_e_cr_fe = FloatField(default=np.nan, help_text="Raw error on [Cr/Fe] abundance ratio")
    raw_e_mn_fe = FloatField(default=np.nan, help_text="Raw error on [Mn/Fe] abundance ratio")
    raw_e_ni_fe = FloatField(default=np.nan, help_text="Raw error on [Ni/Fe] abundance ratio")

    #> Metadata
    ier = IntegerField(default=-1, help_text="Returned state from optimizer")
    nfev = IntegerField(default=-1, help_text="Number of function evaluations")
    x0_index = IntegerField(default=-1, help_text="Index of initial guess used")
    result_flags = BitField(default=0, help_text="Result flags")
    flag_fitting_failure = result_flags.flag(2**0, "Fitting failure")
    
    #> Spectral data
    wavelength = PixelArray(
        accessor_class=LogLambdaArrayAccessor,
        accessor_kwargs=dict(
            crval=4.179,
            cdelt=6e-6,
            naxis=8575,
        ),
        help_text=Glossary.wavelength
    )
    model_flux = TheCannonPixelArray(help_text=Glossary.model_flux)
    continuum = TheCannonPixelArray(help_text=Glossary.continuum)    
    
    @property
    def intermediate_output_path(self):
        parts = f"{self.source_pk}"
        group_dir = f"{parts[-4:-2]}/{parts[-2:]}"
        return f"$MWM_ASTRA/{self.v_astra}/pipelines/TheCannon/{group_dir}/{self.source_pk}-{self.spectrum_pk}.pkl"
    
    
def set_formal_errors():
    (
        TheCannon.update(
            raw_e_teff=TheCannon.e_teff,
            raw_e_logg=TheCannon.e_logg,
            raw_e_fe_h=TheCannon.e_fe_h,
            raw_e_v_micro=TheCannon.e_v_micro,
            raw_e_v_macro=TheCannon.e_v_macro,
            raw_e_c_fe=TheCannon.e_c_fe,
            raw_e_n_fe=TheCannon.e_n_fe,
            raw_e_o_fe=TheCannon.e_o_fe,
            raw_e_na_fe=TheCannon.e_na_fe,
            raw_e_mg_fe=TheCannon.e_mg_fe,
            raw_e_al_fe=TheCannon.e_al_fe,
            raw_e_si_fe=TheCannon.e_si_fe,
            raw_e_s_fe=TheCannon.e_s_fe,
            raw_e_k_fe=TheCannon.e_k_fe,
            raw_e_ca_fe=TheCannon.e_ca_fe,
            raw_e_ti_fe=TheCannon.e_ti_fe,
            raw_e_v_fe=TheCannon.e_v_fe,
            raw_e_cr_fe=TheCannon.e_cr_fe,
            raw_e_mn_fe=TheCannon.e_mn_fe,
            raw_e_ni_fe=TheCannon.e_ni_fe,
        )
        .execute()
    )
    
def apply_noise_model():
    
    with open(expand_path(f"$MWM_ASTRA/{__version__}/aux/TheCannon_corrections.pkl"), "rb") as fp:
        corrections, reference = pickle.load(fp)

    update_kwds = {}
    for label_name, kwds in corrections.items():
        offset, scale = kwds["offset"], kwds["scale"]
        update_kwds[f"e_{label_name}"] = scale * getattr(TheCannon, f"raw_e_{label_name}") + offset
        
    (
        TheCannon
        .update(**update_kwds)
        .execute()
    )
    
            