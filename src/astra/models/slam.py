import pickle
from playhouse.hybrid import hybrid_property

from astra.utils import expand_path
from astra.fields import (
    BitField, PixelArray, BasePixelArrayAccessor, LogLambdaArrayAccessor,
    AutoField,
    FloatField,
    TextField,
    BooleanField,
    IntegerField,
)
from astra.models.pipeline import PipelineOutputModel

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
        


class Slam(PipelineOutputModel):

    """A result from the 'Stellar Labels Machine'."""
    
    #> Stellar Labels
    teff = FloatField(null=True)
    e_teff = FloatField(null=True)
    logg = FloatField(null=True)
    e_logg = FloatField(null=True)
    fe_h = FloatField(null=True)
    e_fe_h = FloatField(null=True)
    fe_h_niu = FloatField(null=True)
    e_fe_h_niu = FloatField(null=True)
    alpha_fe = FloatField(null=True)
    e_alpha_fe = FloatField(null=True)
    
    #> Correlation Coefficients
    rho_teff_logg = FloatField(null=True)
    rho_teff_fe_h = FloatField(null=True)
    rho_teff_fe_h_niu = FloatField(null=True) # 
    rho_teff_alpha_fe = FloatField(null=True) #
    rho_logg_fe_h_niu = FloatField(null=True)
    rho_logg_alpha_fe = FloatField(null=True)
    rho_logg_fe_h = FloatField(null=True)
    rho_fe_h_fe_h_niu = FloatField(null=True)
    rho_fe_h_alpha_fe = FloatField(null=True)

    #> Initial Labels
    initial_teff = FloatField(null=True)
    initial_logg = FloatField(null=True)
    initial_fe_h = FloatField(null=True)
    initial_alpha_fe = FloatField(null=True)
    initial_fe_h_niu = FloatField(null=True)

    #> Metadata
    success = BooleanField()
    status = IntegerField()
    optimality = BooleanField()
    result_flags = BitField(default=0)
    flag_bad_optimizer_status = result_flags.flag(2**0)
    flag_teff_outside_bounds = result_flags.flag(2**1)
    flag_fe_h_outside_bounds = result_flags.flag(2**2)
    flag_outside_photometry_range = result_flags.flag(2**3)
    
    #> Summary Statistics
    chi2 = FloatField()
    rchi2 = FloatField()
    
    #> Spectral Data
    wavelength = PixelArray(
        accessor_class=LogLambdaArrayAccessor,
        accessor_kwargs=dict(
            crval=3.5523,
            cdelt=1e-4,
            naxis=4648,
        ),
    )
    model_flux = SlamPixelArray()
    continuum = SlamPixelArray()
    
    @property    
    def intermediate_output_path(self):
        folders = f"{str(self.spectrum_pk)[-4:-2]}/{str(self.spectrum_pk)[-2:]}"
        return f"$MWM_ASTRA/{self.v_astra}/pipelines/slam/{folders}/{self.spectrum_pk}.pkl"
            
    @hybrid_property
    def flag_warn(self):
        return (
            self.flag_bad_optimizer_status
        |   self.flag_outside_photometry_range
        )
    
    @flag_warn.expression
    def flag_warn(self):    
        return (
            self.flag_bad_optimizer_status
        |   self.flag_outside_photometry_range
        )
    
    @hybrid_property
    def flag_bad(self):
        return (
            self.flag_teff_outside_bounds
        |   self.flag_fe_h_outside_bounds
        |   self.flag_outside_photometry_range
        |   self.flag_bad_optimizer_status
        )        
    
    @flag_bad.expression
    def flag_bad(self):
        return (
            self.flag_teff_outside_bounds
        |   self.flag_fe_h_outside_bounds
        |   self.flag_outside_photometry_range
        |   self.flag_bad_optimizer_status
        )
    
    
def apply_flag_teff_outside_bounds():
    (
        Slam
        .update(result_flags=Slam.flag_teff_outside_bounds.set())
        .where(
            (Slam.teff < 2800) | (Slam.teff > 4500)
        )
        .execute()
    )
    
    
def apply_flag_outside_photometry_range():
    # TODO: put this calculation in at runtime
    
    # Apply flagging criteria as per Zach Way suggestion
    (
        Slam
        .update(result_flags=Slam.flag_outside_photometry_range.set())
        .where(
            (                
                ((Source.bp_mag - Source.rp_mag) >= 3.5)
            |   ((Source.bp_mag - Source.rp_mag) <= 1.5)
            |   (
                    (Source.plx > 0)
                &   (
                        ((Source.g_mag - 5 + 5 * fn.log10(Source.plx)) >= 12)
                    |   ((Source.g_mag - 5 + 5 * fn.log10(Source.plx)) <= 6)
                    )
                )
            )
        )
        .from_(Source)
        .execute()
    )
