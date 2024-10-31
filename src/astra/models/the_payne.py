import datetime
import pickle
from astra.utils import expand_path
from astra.fields import (
    BitField, PixelArray, BasePixelArrayAccessor, LogLambdaArrayAccessor,
    FloatField,
    TextField,
)    
from playhouse.hybrid import hybrid_property
from astra.models.pipeline import PipelineOutputModel

class PaynePixelArrayAccessor(BasePixelArrayAccessor):
    
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




class PaynePixelArray(PixelArray):
    
    def __init__(self, ext=None, column_name=None, transform=None, accessor_class=PaynePixelArrayAccessor, help_text=None, **kwargs):
        super(PaynePixelArray, self).__init__(
            ext=ext,
            column_name=column_name,
            transform=transform,
            accessor_class=accessor_class,
            help_text=help_text,
            **kwargs
        )


class ThePayne(PipelineOutputModel):

    """A result from The Payne."""
        
    #> Stellar Labels
    v_rel = FloatField(null=True)
    teff = FloatField(null=True)
    e_teff = FloatField(null=True)
    logg = FloatField(null=True)
    e_logg = FloatField(null=True)
    v_turb = FloatField(null=True)
    e_v_turb = FloatField(null=True)
    c_h = FloatField(null=True)
    e_c_h = FloatField(null=True)
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
    cu_h = FloatField(null=True)
    e_cu_h = FloatField(null=True)
    ge_h = FloatField(null=True)
    e_ge_h = FloatField(null=True)
    c12_c13 = FloatField(null=True)
    e_c12_c13 = FloatField(null=True)
    v_macro = FloatField(null=True)
    e_v_macro = FloatField(null=True)
    

    #> Summary Statistics
    chi2 = FloatField(null=True)
    reduced_chi2 = FloatField(null=True)
    result_flags = BitField(default=0)
    flag_fitting_failure = result_flags.flag(2**0, "Fitting failure")
    flag_warn_teff = result_flags.flag(2**1, "Teff < 3100 K or Teff > 7900 K")
    flag_warn_logg = result_flags.flag(2**2, "logg < 0.1 or logg > 5.2")
    flag_warn_fe_h = result_flags.flag(2**3, "[Fe/H] > 0.4 or [Fe/H] < -1.4")
    flag_low_snr = result_flags.flag(2**4, "S/N < 70")

    @hybrid_property
    def flag_warn(self):
        return (self.result_flags > 0)

    @flag_warn.expression
    def flag_warn(self):
        return (self.result_flags > 0)

    @hybrid_property
    def flag_bad(self):
        return self.flag_fitting_failure

    @flag_bad.expression
    def flag_bad(self):
        return self.flag_fitting_failure    


    #> Formal uncertainties
    raw_e_teff = FloatField(null=True)
    raw_e_logg = FloatField(null=True)
    raw_e_v_turb = FloatField(null=True)
    raw_e_c_h = FloatField(null=True)
    raw_e_n_h = FloatField(null=True)
    raw_e_o_h = FloatField(null=True)
    raw_e_na_h = FloatField(null=True)
    raw_e_mg_h = FloatField(null=True)
    raw_e_al_h = FloatField(null=True)
    raw_e_si_h = FloatField(null=True)
    raw_e_p_h = FloatField(null=True)
    raw_e_s_h = FloatField(null=True)
    raw_e_k_h = FloatField(null=True)
    raw_e_ca_h = FloatField(null=True)
    raw_e_ti_h = FloatField(null=True)
    raw_e_v_h = FloatField(null=True)
    raw_e_cr_h = FloatField(null=True)
    raw_e_mn_h = FloatField(null=True)
    raw_e_fe_h = FloatField(null=True)
    raw_e_co_h = FloatField(null=True)
    raw_e_ni_h = FloatField(null=True)
    raw_e_cu_h = FloatField(null=True)
    raw_e_ge_h = FloatField(null=True)
    raw_e_c12_c13 = FloatField(null=True)
    raw_e_v_macro = FloatField(null=True)
        
    #> Correlation Coefficients
    rho_teff_logg = FloatField(null=True)
    rho_teff_v_turb = FloatField(null=True)
    rho_teff_c_h = FloatField(null=True)
    rho_teff_n_h = FloatField(null=True)
    rho_teff_o_h = FloatField(null=True)
    rho_teff_na_h = FloatField(null=True)
    rho_teff_mg_h = FloatField(null=True)
    rho_teff_al_h = FloatField(null=True)
    rho_teff_si_h = FloatField(null=True)
    rho_teff_p_h = FloatField(null=True)
    rho_teff_s_h = FloatField(null=True)
    rho_teff_k_h = FloatField(null=True)
    rho_teff_ca_h = FloatField(null=True)
    rho_teff_ti_h = FloatField(null=True)
    rho_teff_v_h = FloatField(null=True)
    rho_teff_cr_h = FloatField(null=True)
    rho_teff_mn_h = FloatField(null=True)
    rho_teff_fe_h = FloatField(null=True)
    rho_teff_co_h = FloatField(null=True)
    rho_teff_ni_h = FloatField(null=True)
    rho_teff_cu_h = FloatField(null=True)
    rho_teff_ge_h = FloatField(null=True)
    rho_teff_c12_c13 = FloatField(null=True)
    rho_teff_v_macro = FloatField(null=True)
    rho_logg_v_turb = FloatField(null=True)
    rho_logg_c_h = FloatField(null=True)
    rho_logg_n_h = FloatField(null=True)
    rho_logg_o_h = FloatField(null=True)
    rho_logg_na_h = FloatField(null=True)
    rho_logg_mg_h = FloatField(null=True)
    rho_logg_al_h = FloatField(null=True)
    rho_logg_si_h = FloatField(null=True)
    rho_logg_p_h = FloatField(null=True)
    rho_logg_s_h = FloatField(null=True)
    rho_logg_k_h = FloatField(null=True)
    rho_logg_ca_h = FloatField(null=True)
    rho_logg_ti_h = FloatField(null=True)
    rho_logg_v_h = FloatField(null=True)
    rho_logg_cr_h = FloatField(null=True)
    rho_logg_mn_h = FloatField(null=True)
    rho_logg_fe_h = FloatField(null=True)
    rho_logg_co_h = FloatField(null=True)
    rho_logg_ni_h = FloatField(null=True)
    rho_logg_cu_h = FloatField(null=True)
    rho_logg_ge_h = FloatField(null=True)
    rho_logg_c12_c13 = FloatField(null=True)
    rho_logg_v_macro = FloatField(null=True)
    rho_v_turb_c_h = FloatField(null=True)
    rho_v_turb_n_h = FloatField(null=True)
    rho_v_turb_o_h = FloatField(null=True)
    rho_v_turb_na_h = FloatField(null=True)
    rho_v_turb_mg_h = FloatField(null=True)
    rho_v_turb_al_h = FloatField(null=True)
    rho_v_turb_si_h = FloatField(null=True)
    rho_v_turb_p_h = FloatField(null=True)
    rho_v_turb_s_h = FloatField(null=True)
    rho_v_turb_k_h = FloatField(null=True)
    rho_v_turb_ca_h = FloatField(null=True)
    rho_v_turb_ti_h = FloatField(null=True)
    rho_v_turb_v_h = FloatField(null=True)
    rho_v_turb_cr_h = FloatField(null=True)
    rho_v_turb_mn_h = FloatField(null=True)
    rho_v_turb_fe_h = FloatField(null=True)
    rho_v_turb_co_h = FloatField(null=True)
    rho_v_turb_ni_h = FloatField(null=True)
    rho_v_turb_cu_h = FloatField(null=True)
    rho_v_turb_ge_h = FloatField(null=True)
    rho_v_turb_c12_c13 = FloatField(null=True)
    rho_v_turb_v_macro = FloatField(null=True)
    rho_c_h_n_h = FloatField(null=True)
    rho_c_h_o_h = FloatField(null=True)
    rho_c_h_na_h = FloatField(null=True)
    rho_c_h_mg_h = FloatField(null=True)
    rho_c_h_al_h = FloatField(null=True)
    rho_c_h_si_h = FloatField(null=True)
    rho_c_h_p_h = FloatField(null=True)
    rho_c_h_s_h = FloatField(null=True)
    rho_c_h_k_h = FloatField(null=True)
    rho_c_h_ca_h = FloatField(null=True)
    rho_c_h_ti_h = FloatField(null=True)
    rho_c_h_v_h = FloatField(null=True)
    rho_c_h_cr_h = FloatField(null=True)
    rho_c_h_mn_h = FloatField(null=True)
    rho_c_h_fe_h = FloatField(null=True)
    rho_c_h_co_h = FloatField(null=True)
    rho_c_h_ni_h = FloatField(null=True)
    rho_c_h_cu_h = FloatField(null=True)
    rho_c_h_ge_h = FloatField(null=True)
    rho_c_h_c12_c13 = FloatField(null=True)
    rho_c_h_v_macro = FloatField(null=True)
    rho_n_h_o_h = FloatField(null=True)
    rho_n_h_na_h = FloatField(null=True)
    rho_n_h_mg_h = FloatField(null=True)
    rho_n_h_al_h = FloatField(null=True)
    rho_n_h_si_h = FloatField(null=True)
    rho_n_h_p_h = FloatField(null=True)
    rho_n_h_s_h = FloatField(null=True)
    rho_n_h_k_h = FloatField(null=True)
    rho_n_h_ca_h = FloatField(null=True)
    rho_n_h_ti_h = FloatField(null=True)
    rho_n_h_v_h = FloatField(null=True)
    rho_n_h_cr_h = FloatField(null=True)
    rho_n_h_mn_h = FloatField(null=True)
    rho_n_h_fe_h = FloatField(null=True)
    rho_n_h_co_h = FloatField(null=True)
    rho_n_h_ni_h = FloatField(null=True)
    rho_n_h_cu_h = FloatField(null=True)
    rho_n_h_ge_h = FloatField(null=True)
    rho_n_h_c12_c13 = FloatField(null=True)
    rho_n_h_v_macro = FloatField(null=True)
    rho_o_h_na_h = FloatField(null=True)
    rho_o_h_mg_h = FloatField(null=True)
    rho_o_h_al_h = FloatField(null=True)
    rho_o_h_si_h = FloatField(null=True)
    rho_o_h_p_h = FloatField(null=True)
    rho_o_h_s_h = FloatField(null=True)
    rho_o_h_k_h = FloatField(null=True)
    rho_o_h_ca_h = FloatField(null=True)
    rho_o_h_ti_h = FloatField(null=True)
    rho_o_h_v_h = FloatField(null=True)
    rho_o_h_cr_h = FloatField(null=True)
    rho_o_h_mn_h = FloatField(null=True)
    rho_o_h_fe_h = FloatField(null=True)
    rho_o_h_co_h = FloatField(null=True)
    rho_o_h_ni_h = FloatField(null=True)
    rho_o_h_cu_h = FloatField(null=True)
    rho_o_h_ge_h = FloatField(null=True)
    rho_o_h_c12_c13 = FloatField(null=True)
    rho_o_h_v_macro = FloatField(null=True)
    rho_na_h_mg_h = FloatField(null=True)
    rho_na_h_al_h = FloatField(null=True)
    rho_na_h_si_h = FloatField(null=True)
    rho_na_h_p_h = FloatField(null=True)
    rho_na_h_s_h = FloatField(null=True)
    rho_na_h_k_h = FloatField(null=True)
    rho_na_h_ca_h = FloatField(null=True)
    rho_na_h_ti_h = FloatField(null=True)
    rho_na_h_v_h = FloatField(null=True)
    rho_na_h_cr_h = FloatField(null=True)
    rho_na_h_mn_h = FloatField(null=True)
    rho_na_h_fe_h = FloatField(null=True)
    rho_na_h_co_h = FloatField(null=True)
    rho_na_h_ni_h = FloatField(null=True)
    rho_na_h_cu_h = FloatField(null=True)
    rho_na_h_ge_h = FloatField(null=True)
    rho_na_h_c12_c13 = FloatField(null=True)
    rho_na_h_v_macro = FloatField(null=True)
    rho_mg_h_al_h = FloatField(null=True)
    rho_mg_h_si_h = FloatField(null=True)
    rho_mg_h_p_h = FloatField(null=True)
    rho_mg_h_s_h = FloatField(null=True)
    rho_mg_h_k_h = FloatField(null=True)
    rho_mg_h_ca_h = FloatField(null=True)
    rho_mg_h_ti_h = FloatField(null=True)
    rho_mg_h_v_h = FloatField(null=True)
    rho_mg_h_cr_h = FloatField(null=True)
    rho_mg_h_mn_h = FloatField(null=True)
    rho_mg_h_fe_h = FloatField(null=True)
    rho_mg_h_co_h = FloatField(null=True)
    rho_mg_h_ni_h = FloatField(null=True)
    rho_mg_h_cu_h = FloatField(null=True)
    rho_mg_h_ge_h = FloatField(null=True)
    rho_mg_h_c12_c13 = FloatField(null=True)
    rho_mg_h_v_macro = FloatField(null=True)
    rho_al_h_si_h = FloatField(null=True)
    rho_al_h_p_h = FloatField(null=True)
    rho_al_h_s_h = FloatField(null=True)
    rho_al_h_k_h = FloatField(null=True)
    rho_al_h_ca_h = FloatField(null=True)
    rho_al_h_ti_h = FloatField(null=True)
    rho_al_h_v_h = FloatField(null=True)
    rho_al_h_cr_h = FloatField(null=True)
    rho_al_h_mn_h = FloatField(null=True)
    rho_al_h_fe_h = FloatField(null=True)
    rho_al_h_co_h = FloatField(null=True)
    rho_al_h_ni_h = FloatField(null=True)
    rho_al_h_cu_h = FloatField(null=True)
    rho_al_h_ge_h = FloatField(null=True)
    rho_al_h_c12_c13 = FloatField(null=True)
    rho_al_h_v_macro = FloatField(null=True)
    rho_si_h_p_h = FloatField(null=True)
    rho_si_h_s_h = FloatField(null=True)
    rho_si_h_k_h = FloatField(null=True)
    rho_si_h_ca_h = FloatField(null=True)
    rho_si_h_ti_h = FloatField(null=True)
    rho_si_h_v_h = FloatField(null=True)
    rho_si_h_cr_h = FloatField(null=True)
    rho_si_h_mn_h = FloatField(null=True)
    rho_si_h_fe_h = FloatField(null=True)
    rho_si_h_co_h = FloatField(null=True)
    rho_si_h_ni_h = FloatField(null=True)
    rho_si_h_cu_h = FloatField(null=True)
    rho_si_h_ge_h = FloatField(null=True)
    rho_si_h_c12_c13 = FloatField(null=True)
    rho_si_h_v_macro = FloatField(null=True)
    rho_p_h_s_h = FloatField(null=True)
    rho_p_h_k_h = FloatField(null=True)
    rho_p_h_ca_h = FloatField(null=True)
    rho_p_h_ti_h = FloatField(null=True)
    rho_p_h_v_h = FloatField(null=True)
    rho_p_h_cr_h = FloatField(null=True)
    rho_p_h_mn_h = FloatField(null=True)
    rho_p_h_fe_h = FloatField(null=True)
    rho_p_h_co_h = FloatField(null=True)
    rho_p_h_ni_h = FloatField(null=True)
    rho_p_h_cu_h = FloatField(null=True)
    rho_p_h_ge_h = FloatField(null=True)
    rho_p_h_c12_c13 = FloatField(null=True)
    rho_p_h_v_macro = FloatField(null=True)
    rho_s_h_k_h = FloatField(null=True)
    rho_s_h_ca_h = FloatField(null=True)
    rho_s_h_ti_h = FloatField(null=True)
    rho_s_h_v_h = FloatField(null=True)
    rho_s_h_cr_h = FloatField(null=True)
    rho_s_h_mn_h = FloatField(null=True)
    rho_s_h_fe_h = FloatField(null=True)
    rho_s_h_co_h = FloatField(null=True)
    rho_s_h_ni_h = FloatField(null=True)
    rho_s_h_cu_h = FloatField(null=True)
    rho_s_h_ge_h = FloatField(null=True)
    rho_s_h_c12_c13 = FloatField(null=True)
    rho_s_h_v_macro = FloatField(null=True)
    rho_k_h_ca_h = FloatField(null=True)
    rho_k_h_ti_h = FloatField(null=True)
    rho_k_h_v_h = FloatField(null=True)
    rho_k_h_cr_h = FloatField(null=True)
    rho_k_h_mn_h = FloatField(null=True)
    rho_k_h_fe_h = FloatField(null=True)
    rho_k_h_co_h = FloatField(null=True)
    rho_k_h_ni_h = FloatField(null=True)
    rho_k_h_cu_h = FloatField(null=True)
    rho_k_h_ge_h = FloatField(null=True)
    rho_k_h_c12_c13 = FloatField(null=True)
    rho_k_h_v_macro = FloatField(null=True)
    rho_ca_h_ti_h = FloatField(null=True)
    rho_ca_h_v_h = FloatField(null=True)
    rho_ca_h_cr_h = FloatField(null=True)
    rho_ca_h_mn_h = FloatField(null=True)
    rho_ca_h_fe_h = FloatField(null=True)
    rho_ca_h_co_h = FloatField(null=True)
    rho_ca_h_ni_h = FloatField(null=True)
    rho_ca_h_cu_h = FloatField(null=True)
    rho_ca_h_ge_h = FloatField(null=True)
    rho_ca_h_c12_c13 = FloatField(null=True)
    rho_ca_h_v_macro = FloatField(null=True)
    rho_ti_h_v_h = FloatField(null=True)
    rho_ti_h_cr_h = FloatField(null=True)
    rho_ti_h_mn_h = FloatField(null=True)
    rho_ti_h_fe_h = FloatField(null=True)
    rho_ti_h_co_h = FloatField(null=True)
    rho_ti_h_ni_h = FloatField(null=True)
    rho_ti_h_cu_h = FloatField(null=True)
    rho_ti_h_ge_h = FloatField(null=True)
    rho_ti_h_c12_c13 = FloatField(null=True)
    rho_ti_h_v_macro = FloatField(null=True)
    rho_v_h_cr_h = FloatField(null=True)
    rho_v_h_mn_h = FloatField(null=True)
    rho_v_h_fe_h = FloatField(null=True)
    rho_v_h_co_h = FloatField(null=True)
    rho_v_h_ni_h = FloatField(null=True)
    rho_v_h_cu_h = FloatField(null=True)
    rho_v_h_ge_h = FloatField(null=True)
    rho_v_h_c12_c13 = FloatField(null=True)
    rho_v_h_v_macro = FloatField(null=True)
    rho_cr_h_mn_h = FloatField(null=True)
    rho_cr_h_fe_h = FloatField(null=True)
    rho_cr_h_co_h = FloatField(null=True)
    rho_cr_h_ni_h = FloatField(null=True)
    rho_cr_h_cu_h = FloatField(null=True)
    rho_cr_h_ge_h = FloatField(null=True)
    rho_cr_h_c12_c13 = FloatField(null=True)
    rho_cr_h_v_macro = FloatField(null=True)
    rho_mn_h_fe_h = FloatField(null=True)
    rho_mn_h_co_h = FloatField(null=True)
    rho_mn_h_ni_h = FloatField(null=True)
    rho_mn_h_cu_h = FloatField(null=True)
    rho_mn_h_ge_h = FloatField(null=True)
    rho_mn_h_c12_c13 = FloatField(null=True)
    rho_mn_h_v_macro = FloatField(null=True)
    rho_fe_h_co_h = FloatField(null=True)
    rho_fe_h_ni_h = FloatField(null=True)
    rho_fe_h_cu_h = FloatField(null=True)
    rho_fe_h_ge_h = FloatField(null=True)
    rho_fe_h_c12_c13 = FloatField(null=True)
    rho_fe_h_v_macro = FloatField(null=True)
    rho_co_h_ni_h = FloatField(null=True)
    rho_co_h_cu_h = FloatField(null=True)
    rho_co_h_ge_h = FloatField(null=True)
    rho_co_h_c12_c13 = FloatField(null=True)
    rho_co_h_v_macro = FloatField(null=True)
    rho_ni_h_cu_h = FloatField(null=True)
    rho_ni_h_ge_h = FloatField(null=True)
    rho_ni_h_c12_c13 = FloatField(null=True)
    rho_ni_h_v_macro = FloatField(null=True)
    rho_cu_h_ge_h = FloatField(null=True)
    rho_cu_h_c12_c13 = FloatField(null=True)
    rho_cu_h_v_macro = FloatField(null=True)
    rho_ge_h_c12_c13 = FloatField(null=True)
    rho_ge_h_v_macro = FloatField(null=True)
    rho_c12_c13_v_macro = FloatField(null=True)
        
    #> Spectral Data
    wavelength = PixelArray(
        accessor_class=LogLambdaArrayAccessor,
        accessor_kwargs=dict(
            crval=4.179,
            cdelt=6e-6,
            naxis=8575,
        ),
    )
    model_flux = PaynePixelArray()
    continuum = PaynePixelArray()
    
    #@property    
    #def intermediate_output_path(self):
    #    folders = f"{str(self.source_pk)[-4:-2]:0>2}/{str(self.source_pk)[-2:]:0>2}"
    #    return f"$MWM_ASTRA/{self.v_astra}/pipelines/ThePayne/intermediate/{folders}/{self.source_pk}-{self.spectrum_pk}.pkl"

    @property
    def intermediate_output_path(self):
        folders = f"{str(self.spectrum_pk)[-4:-2]:0>2}/{str(self.spectrum_pk)[-2:]:0>2}"
        return f"$MWM_ASTRA/{self.v_astra}/pipelines/ThePayne/intermediate/{folders}/{self.spectrum_pk}.pkl"
        
    
    
def set_formal_errors():
    ThePayne.update(raw_e_teff=ThePayne.e_teff).execute()
    ThePayne.update(raw_e_logg=ThePayne.e_logg).execute()
    ThePayne.update(raw_e_v_turb=ThePayne.e_v_turb).execute()
    ThePayne.update(raw_e_c_h=ThePayne.e_c_h).execute()
    ThePayne.update(raw_e_n_h=ThePayne.e_n_h).execute()
    ThePayne.update(raw_e_o_h=ThePayne.e_o_h).execute()
    ThePayne.update(raw_e_na_h=ThePayne.e_na_h).execute()
    ThePayne.update(raw_e_mg_h=ThePayne.e_mg_h).execute()
    ThePayne.update(raw_e_al_h=ThePayne.e_al_h).execute()
    ThePayne.update(raw_e_si_h=ThePayne.e_si_h).execute()
    ThePayne.update(raw_e_p_h=ThePayne.e_p_h).execute()
    ThePayne.update(raw_e_s_h=ThePayne.e_s_h).execute()
    ThePayne.update(raw_e_k_h=ThePayne.e_k_h).execute()
    ThePayne.update(raw_e_ca_h=ThePayne.e_ca_h).execute()
    ThePayne.update(raw_e_ti_h=ThePayne.e_ti_h).execute()
    ThePayne.update(raw_e_v_h=ThePayne.e_v_h).execute()
    ThePayne.update(raw_e_cr_h=ThePayne.e_cr_h).execute()
    ThePayne.update(raw_e_mn_h=ThePayne.e_mn_h).execute()
    ThePayne.update(raw_e_fe_h=ThePayne.e_fe_h).execute()
    ThePayne.update(raw_e_co_h=ThePayne.e_co_h).execute()
    ThePayne.update(raw_e_ni_h=ThePayne.e_ni_h).execute()
    ThePayne.update(raw_e_cu_h=ThePayne.e_cu_h).execute()
    ThePayne.update(raw_e_ge_h=ThePayne.e_ge_h).execute()
    ThePayne.update(raw_e_c12_c13=ThePayne.e_c12_c13).execute()
    ThePayne.update(raw_e_v_macro=ThePayne.e_v_macro).execute()


def apply_noise_model():
    
    with open(expand_path(f"$MWM_ASTRA/{__version__}/aux/ThePayne_corrections.pkl"), "rb") as fp:
        corrections, reference = pickle.load(fp)

    update_kwds = {}
    for label_name, kwds in corrections.items():
        offset, scale = kwds["offset"], kwds["scale"]
        update_kwds[f"e_{label_name}"] = scale * getattr(ThePayne, f"raw_e_{label_name}") + offset
        
    (
        ThePayne
        .update(**update_kwds)
        .execute()
    )
    


def apply_flags():
    (
        ThePayne
        .update(result_flags=ThePayne.flag_warn_teff.set())
        .where(
            (ThePayne.teff < 3100)
        |   (ThePayne.teff > 7900)
        )
        .execute()
    )
    (
        ThePayne
        .update(result_flags=ThePayne.flag_warn_logg.set())
        .where(
            (ThePayne.logg < 0.1)
        |   (ThePayne.logg > 5.2)
        )
        .execute()
    )
    (
        ThePayne
        .update(result_flags=ThePayne.flag_warn_fe_h.set())
        .where(
            (ThePayne.fe_h > 0.4)
        |   (ThePayne.fe_h < -1.4)
        )
        .execute()
    )
    """
    (
        ThePayne
        .update(result_flags=ThePayne.flag_low_snr.set())
        .where(
            (ThePayne.snr < 70)
        )
        .execute()
    )
    """
