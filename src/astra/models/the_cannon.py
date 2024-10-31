import datetime
import pickle
import numpy as np
from peewee import fn
from astra.utils import expand_path
from astra.fields import (
    BitField, PixelArray, BasePixelArrayAccessor, LogLambdaArrayAccessor,
    IntegerField,
    FloatField,
    TextField,
    BooleanField,
)    
from astra.models.pipeline import PipelineOutputModel
from astra.utils import log, expand_path


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
class TrainingSet(PipelineOutputModel):

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


class TrainingSetSpectrum(PipelineOutputModel):

    training_set_pk = ForeignKeyField(TrainingSet)
    spectrum_pk = ForeignKeyField(Spectrum)
    source_pk = ForeignKeyField(Source, null=True)
'''


class TheCannon(PipelineOutputModel):

    """Stellar labels estimated using The Cannon."""
    
    #> Stellar Labels
    teff = FloatField(default=np.nan)
    e_teff = FloatField(default=np.nan)
    logg = FloatField(default=np.nan)
    e_logg = FloatField(default=np.nan)
    fe_h = FloatField(default=np.nan)
    e_fe_h = FloatField(default=np.nan)
    v_micro = FloatField(default=np.nan)
    e_v_micro = FloatField(default=np.nan)
    v_macro = FloatField(default=np.nan)
    e_v_macro = FloatField(default=np.nan)
    
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

    #> Formal uncertainties
    raw_e_teff = FloatField(default=np.nan)
    raw_e_logg = FloatField(default=np.nan)
    raw_e_fe_h = FloatField(default=np.nan)
    raw_e_v_micro = FloatField(default=np.nan)
    raw_e_v_macro = FloatField(default=np.nan)
    raw_e_c_fe = FloatField(default=np.nan)
    raw_e_n_fe = FloatField(default=np.nan)
    raw_e_o_fe = FloatField(default=np.nan)
    raw_e_na_fe = FloatField(default=np.nan)
    raw_e_mg_fe = FloatField(default=np.nan)
    raw_e_al_fe = FloatField(default=np.nan)
    raw_e_si_fe = FloatField(default=np.nan)
    raw_e_s_fe = FloatField(default=np.nan)
    raw_e_k_fe = FloatField(default=np.nan)
    raw_e_ca_fe = FloatField(default=np.nan)
    raw_e_ti_fe = FloatField(default=np.nan)
    raw_e_v_fe = FloatField(default=np.nan)
    raw_e_cr_fe = FloatField(default=np.nan)
    raw_e_mn_fe = FloatField(default=np.nan)
    raw_e_ni_fe = FloatField(default=np.nan)

    #> Metadata
    ier = IntegerField(default=-1)
    nfev = IntegerField(default=-1)
    x0_index = IntegerField(default=-1)
    result_flags = BitField(default=0)
    flag_fitting_failure = result_flags.flag(2**0, help_text="Fitting failure")
    
    #> Spectral data
    wavelength = PixelArray(
        accessor_class=LogLambdaArrayAccessor,
        accessor_kwargs=dict(
            crval=4.179,
            cdelt=6e-6,
            naxis=8575,
        ),
    )
    model_flux = TheCannonPixelArray()
    continuum = TheCannonPixelArray()
    
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
    
            
