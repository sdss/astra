import datetime
import pickle
from peewee import (
    chunked,
    AutoField,
    FloatField,
    TextField,
    ForeignKeyField,
    BooleanField,
    IntegerField,
    DateTimeField
)
from playhouse.hybrid import hybrid_property

from astra import __version__
from astra.utils import expand_path
from astra.models.fields import BitField, PixelArray, BasePixelArrayAccessor, LogLambdaArrayAccessor
from astra.models.base import BaseModel
from astra.models.fields import BitField
from astra.models.source import Source
from astra.models.spectrum import Spectrum
from astra.models.pipeline import PipelineOutputMixin

from astra.glossary import Glossary




class SlamPixelArrayAccessor(BasePixelArrayAccessor):
    
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


class SlamPixelArray(PixelArray):
    
    def __init__(self, ext=None, column_name=None, transform=None, accessor_class=SlamPixelArrayAccessor, help_text=None, **kwargs):
        super(SlamPixelArray, self).__init__(
            ext=ext,
            column_name=column_name,
            transform=transform,
            accessor_class=accessor_class,
            help_text=help_text,
            **kwargs
        )
        


class Slam(BaseModel, PipelineOutputMixin):

    """A result from the 'Stellar Labels Machine'."""

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
    teff = FloatField(null=True, help_text=Glossary.teff)
    e_teff = FloatField(null=True, help_text=Glossary.e_teff)
    logg = FloatField(null=True, help_text=Glossary.logg)
    e_logg = FloatField(null=True, help_text=Glossary.e_logg)
    fe_h = FloatField(null=True, help_text=Glossary.fe_h)
    e_fe_h = FloatField(null=True, help_text=Glossary.e_fe_h)
    
    #> Correlation Coefficients
    rho_teff_logg = FloatField(null=True, help_text=Glossary.rho_teff_logg)
    rho_teff_fe_h = FloatField(null=True, help_text=Glossary.rho_teff_fe_h)
    rho_logg_fe_h = FloatField(null=True, help_text=Glossary.rho_logg_fe_h)

    #> Initial Labels
    initial_teff = FloatField(null=True, help_text=Glossary.initial_teff)
    initial_logg = FloatField(null=True, help_text=Glossary.initial_logg)
    initial_fe_h = FloatField(null=True, help_text=Glossary.initial_fe_h)

    #> Metadata
    success = BooleanField(help_text="Optimizer returned successful value")
    status = IntegerField(help_text="Optimization status")
    optimality = BooleanField(help_text="Optimality condition")
    bad_flag = BooleanField(default=False, help_text="Results are unreliable")
    warn_flag = BooleanField(default=False, help_text="Results may not be reliable")
    result_flags = BitField(default=0, help_text=Glossary.result_flags)        
        
    flag_bad_optimizer_status = result_flags.flag(2**0, help_text="Optimizer status value indicate results may not be reliable")
    flag_teff_outside_bounds = result_flags.flag(2**1, help_text="Teff is outside reliable bounds: (2000, 5000)")
    flag_fe_h_outside_bounds = result_flags.flag(2**2, help_text="[Fe/H] is outside reliable bounds: (-1, 0.5)")        
    
    #> Summary Statistics
    chi2 = FloatField(help_text=Glossary.chi2)
    rchi2 = FloatField(help_text=Glossary.rchi2)
    
    #> Spectral Data
    wavelength = PixelArray(
        accessor_class=LogLambdaArrayAccessor,
        accessor_kwargs=dict(
            crval=3.5523,
            cdelt=1e-4,
            naxis=4648,
        ),
        help_text=Glossary.wavelength
    )
    model_flux = SlamPixelArray(help_text=Glossary.model_flux)
    continuum = SlamPixelArray(help_text=Glossary.continuum)
    
    @property    
    def intermediate_output_path(self):
        return f"$MWM_ASTRA/{self.v_astra}/pipelines/slam/{self.source_pk}-{self.spectrum_pk}.pkl"
        
    
