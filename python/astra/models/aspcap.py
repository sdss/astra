from peewee import (
    AutoField,
    FloatField,
    TextField,
    ForeignKeyField,
    IntegerField,
    BitField,
)

import numpy as np
from astra import __version__
from astra.models.base import BaseModel
from astra.models.fields import BitField
from astra.models.source import Source
from astra.models.spectrum import Spectrum
from astra.models.pipeline import PipelineOutputMixin
from functools import cached_property

class FerreOutputMixin(PipelineOutputMixin):
        
    @cached_property
    def ferre_flux(self):
        return self._get_input_pixel_array("flux.input")
        
    @cached_property
    def ferre_e_flux(self):
        return self._get_input_pixel_array("e_flux.input")
    

    @cached_property
    def model_flux(self):
        return self._get_output_pixel_array("model_flux.output")
        
    @cached_property
    def rectified_model_flux(self):
        return self._get_output_pixel_array("rectified_model_flux.output")
        
    @cached_property
    def rectified_flux(self):
        return self._get_output_pixel_array("rectified_flux.output")

    def unmask(self, array, fill_value=np.nan):
        from astra.pipelines.ferre.utils import get_apogee_pixel_mask
        mask = get_apogee_pixel_mask()
        unmasked_array = fill_value * np.ones(mask.shape)
        unmasked_array[mask] = array
        return unmasked_array

        
    def _get_input_pixel_array(self, basename):
        return np.loadtxt(
            fname=f"{self.pwd}/{basename}",
            skiprows=int(self.ferre_input_index), 
            max_rows=1,
        )


    def _get_output_pixel_array(self, basename, P=7514):
        from astra.pipelines.ferre.utils import parse_ferre_spectrum_name
        

        assert self.ferre_output_index >= 0

        kwds = dict(
            fname=f"{self.pwd}/{basename}",
            skiprows=int(self.ferre_output_index), 
            max_rows=1,
        )

        name, = np.atleast_1d(np.loadtxt(usecols=(0, ), dtype=str, **kwds))
        array = np.loadtxt(usecols=range(1, 1+P), **kwds)

        meta = parse_ferre_spectrum_name(name)
        assert int(meta["sdss_id"]) == self.sdss_id
        assert int(meta["spectrum_id"]) == self.spectrum_id
        assert int(meta["index"]) == self.ferre_input_index
        return array


class FerreCoarse(BaseModel, FerreOutputMixin):

    sdss_id = ForeignKeyField(Source, index=True, lazy_load=False)
    spectrum_id = ForeignKeyField(Spectrum, index=True, lazy_load=False)
    
    #> Astra Metadata
    task_id = AutoField()
    v_astra = TextField(default=__version__)
    t_elapsed = FloatField(null=True)
    tag = TextField(default="", index=True)

    #> Grid and Working Directory
    pwd = TextField(default="")
    short_grid_name = TextField(default="")
    header_path = TextField(default="")
    
    #> Initial Stellar Parameters
    initial_teff = FloatField(null=True)
    initial_logg = FloatField(null=True)
    initial_m_h = FloatField(null=True)
    initial_log10_v_sini = FloatField(null=True)
    initial_log10_v_micro = FloatField(null=True)
    initial_alpha_m = FloatField(null=True)
    initial_c_m = FloatField(null=True)
    initial_n_m = FloatField(null=True)

    initial_flags = BitField(default=0)
    flag_initial_guess_from_apogeenet = initial_flags.flag(2**0, help_text="Initial guess from APOGEENet")
    flag_initial_guess_from_doppler = initial_flags.flag(2**1, help_text="Initial guess from Doppler (SDSS-V)")
    flag_initial_guess_from_doppler_sdss4 = initial_flags.flag(2**1, help_text="Initial guess from Doppler (SDSS-IV)")
    flag_initial_guess_from_gaia_xp_andrae23 = initial_flags.flag(2**3, help_text="Initial guess from Andrae et al. (2023)")
    flag_initial_guess_from_user = initial_flags.flag(2**2, help_text="Initial guess specified by user")

    #> FERRE Settings
    continuum_order = IntegerField(default=-1)
    continuum_reject = FloatField(null=True)
    interpolation_order = IntegerField(default=-1)
    weight_path = TextField(default="")
    frozen_flags = BitField(default=0)
    f_access = IntegerField(default=-1)
    f_format = IntegerField(default=-1)
    n_threads = IntegerField(default=-1)

    flag_teff_frozen = frozen_flags.flag(2**0, "Effective temperature is frozen")
    flag_logg_frozen = frozen_flags.flag(2**1, "Surface gravity is frozen")
    flag_m_h_frozen = frozen_flags.flag(2**2, "[M/H] is frozen")
    flag_log10_v_sini_frozen = frozen_flags.flag(2**3, "Rotational broadening is frozen")
    flag_log10_v_micro_frozen = frozen_flags.flag(2**4, "Microturbulence is frozen")
    flag_alpha_m_frozen = frozen_flags.flag(2**5, "[alpha/M] is frozen")
    flag_c_m_frozen = frozen_flags.flag(2**6, "[C/M] is frozen")
    flag_n_m_frozen = frozen_flags.flag(2**7, "[N/M] is frozen")

    #> Stellar Parameters
    teff = FloatField(null=True)
    e_teff = FloatField(null=True)
    logg = FloatField(null=True)
    e_logg = FloatField(null=True)
    m_h = FloatField(null=True)
    e_m_h = FloatField(null=True)
    log10_v_sini = FloatField(null=True)
    e_log10_v_sini = FloatField(null=True)
    log10_v_micro = FloatField(null=True)
    e_log10_v_micro = FloatField(null=True)
    alpha_m = FloatField(null=True)
    e_alpha_m = FloatField(null=True)
    c_m = FloatField(null=True)
    e_c_m = FloatField(null=True)
    n_m = FloatField(null=True)
    e_n_m = FloatField(null=True)

    teff_flags = BitField(default=0)
    logg_flags = BitField(default=0)
    m_h_flags = BitField(default=0)
    log10_v_sini_flags = BitField(default=0)
    log10_v_micro_flags = BitField(default=0)
    alpha_m_flags = BitField(default=0)
    c_m_flags = BitField(default=0)
    n_m_flags = BitField(default=0)

    # TODO: flag definitions for each dimension (DRY)
    #> FERRE Access Fields
    ferre_name = TextField(default="")
    ferre_input_index = IntegerField(default=-1)
    ferre_output_index = IntegerField(default=-1)
    ferre_n_obj = IntegerField(default=-1)

    #> Summary Statistics
    chisq = FloatField(null=True)
    ferre_log_snr_sq = FloatField(null=True)
    ferre_log_chisq = FloatField(null=True)
    ferre_frac_phot_data_points = FloatField(default=0)
    ferre_log_penalized_chisq = FloatField(null=True)
    ferre_time_load_grid = FloatField(null=True)
    ferre_time_elapsed = FloatField(null=True)
    ferre_flags = BitField(default=0)

    flag_ferre_fail = ferre_flags.flag(2**0, "FERRE failed")
    flag_missing_model_flux = ferre_flags.flag(2**1, "Missing model fluxes from FERRE")
    flag_potential_ferre_timeout = ferre_flags.flag(2**2, "Potentially impacted by FERRE timeout")
    flag_no_suitable_initial_guess = ferre_flags.flag(2**3, help_text="FERRE not executed because there's no suitable initial guess")



class FerreStellarParameters(BaseModel, FerreOutputMixin):

    sdss_id = ForeignKeyField(Source, index=True, lazy_load=False)
    spectrum_id = ForeignKeyField(Spectrum, index=True, lazy_load=False)
    upstream = ForeignKeyField(FerreCoarse, index=True)

    #> Astra Metadata
    task_id = AutoField()
    v_astra = TextField(default=__version__)
    t_elapsed = FloatField(null=True)
    tag = TextField(default="", index=True)

    #> Grid and Working Directory
    pwd = TextField(default="")
    short_grid_name = TextField(default="")
    header_path = TextField(default="")
    
    #> Initial Stellar Parameters
    initial_teff = FloatField(null=True)
    initial_logg = FloatField(null=True)
    initial_m_h = FloatField(null=True)
    initial_log10_v_sini = FloatField(null=True)
    initial_log10_v_micro = FloatField(null=True)
    initial_alpha_m = FloatField(null=True)
    initial_c_m = FloatField(null=True)
    initial_n_m = FloatField(null=True)

    initial_flags = BitField(default=0)
    flag_initial_guess_from_apogeenet = initial_flags.flag(2**0, help_text="Initial guess from APOGEENet")
    flag_initial_guess_from_doppler = initial_flags.flag(2**1, help_text="Initial guess from Doppler (SDSS-V)")
    flag_initial_guess_from_doppler_sdss4 = initial_flags.flag(2**1, help_text="Initial guess from Doppler (SDSS-IV)")
    flag_initial_guess_from_gaia_xp_andrae23 = initial_flags.flag(2**3, help_text="Initial guess from Andrae et al. (2023)")
    flag_initial_guess_from_user = initial_flags.flag(2**2, help_text="Initial guess specified by user")

    #> FERRE Settings
    continuum_order = IntegerField(default=-1)
    continuum_reject = FloatField(null=True)
    interpolation_order = IntegerField(default=-1)
    weight_path = TextField(default="")
    frozen_flags = BitField(default=0)
    f_access = IntegerField(default=-1)
    f_format = IntegerField(default=-1)
    n_threads = IntegerField(default=-1)

    flag_teff_frozen = frozen_flags.flag(2**0, "Effective temperature is frozen")
    flag_logg_frozen = frozen_flags.flag(2**1, "Surface gravity is frozen")
    flag_m_h_frozen = frozen_flags.flag(2**2, "[M/H] is frozen")
    flag_log10_v_sini_frozen = frozen_flags.flag(2**3, "Rotational broadening is frozen")
    flag_log10_v_micro_frozen = frozen_flags.flag(2**4, "Microturbulence is frozen")
    flag_alpha_m_frozen = frozen_flags.flag(2**5, "[alpha/M] is frozen")
    flag_c_m_frozen = frozen_flags.flag(2**6, "[C/M] is frozen")
    flag_n_m_frozen = frozen_flags.flag(2**7, "[N/M] is frozen")

    #> Stellar Parameters
    teff = FloatField(null=True)
    e_teff = FloatField(null=True)
    logg = FloatField(null=True)
    e_logg = FloatField(null=True)
    m_h = FloatField(null=True)
    e_m_h = FloatField(null=True)
    log10_v_sini = FloatField(null=True)
    e_log10_v_sini = FloatField(null=True)
    log10_v_micro = FloatField(null=True)
    e_log10_v_micro = FloatField(null=True)
    alpha_m = FloatField(null=True)
    e_alpha_m = FloatField(null=True)
    c_m = FloatField(null=True)
    e_c_m = FloatField(null=True)
    n_m = FloatField(null=True)
    e_n_m = FloatField(null=True)

    teff_flags = BitField(default=0)
    logg_flags = BitField(default=0)
    m_h_flags = BitField(default=0)
    log10_v_sini_flags = BitField(default=0)
    log10_v_micro_flags = BitField(default=0)
    alpha_m_flags = BitField(default=0)
    c_m_flags = BitField(default=0)
    n_m_flags = BitField(default=0)

    # TODO: flag definitions for each dimension (DRY)
    #> FERRE Access Fields
    ferre_name = TextField(default="")
    ferre_input_index = IntegerField(default=-1)
    ferre_output_index = IntegerField(default=-1)
    ferre_n_obj = IntegerField(default=-1)

    #> Summary Statistics
    chisq = FloatField(null=True)
    ferre_log_snr_sq = FloatField(null=True)
    ferre_log_chisq = FloatField(null=True)
    ferre_frac_phot_data_points = FloatField(default=0)
    ferre_log_penalized_chisq = FloatField(null=True)
    ferre_time_load_grid = FloatField(null=True)
    ferre_time_elapsed = FloatField(null=True)
    ferre_flags = BitField(default=0)
    
    flag_ferre_fail = ferre_flags.flag(2**0, "FERRE failed")
    flag_missing_model_flux = ferre_flags.flag(2**1, "Missing model fluxes from FERRE")
    flag_potential_ferre_timeout = ferre_flags.flag(2**2, "Potentially impacted by FERRE timeout")
    flag_no_suitable_initial_guess = ferre_flags.flag(2**3, help_text="FERRE not executed because there's no suitable initial guess")



class FerreChemicalAbundances(BaseModel, FerreOutputMixin):

    # TODO: Review this, it's a nearly direct copy from stellar parameters

    sdss_id = ForeignKeyField(Source, index=True, lazy_load=False)
    spectrum_id = ForeignKeyField(Spectrum, index=True, lazy_load=False)
    upstream = ForeignKeyField(FerreStellarParameters, index=True)

    #> Astra Metadata
    task_id = AutoField()
    v_astra = TextField(default=__version__)
    t_elapsed = FloatField(null=True)
    tag = TextField(default="", index=True)

    #> Grid and Working Directory
    pwd = TextField(default="")
    short_grid_name = TextField(default="")
    header_path = TextField(default="")
    
    #> Initial Stellar Parameters
    initial_teff = FloatField(null=True)
    initial_logg = FloatField(null=True)
    initial_m_h = FloatField(null=True)
    initial_log10_v_sini = FloatField(null=True)
    initial_log10_v_micro = FloatField(null=True)
    initial_alpha_m = FloatField(null=True)
    initial_c_m = FloatField(null=True)
    initial_n_m = FloatField(null=True)

    #> FERRE Settings
    continuum_order = IntegerField(default=-1)
    continuum_reject = FloatField(null=True)
    interpolation_order = IntegerField(default=-1)
    weight_path = TextField(default="")
    frozen_flags = BitField(default=0)
    f_access = IntegerField(default=-1)
    f_format = IntegerField(default=-1)
    n_threads = IntegerField(default=-1)

    flag_teff_frozen = frozen_flags.flag(2**0, "Effective temperature is frozen")
    flag_logg_frozen = frozen_flags.flag(2**1, "Surface gravity is frozen")
    flag_m_h_frozen = frozen_flags.flag(2**2, "[M/H] is frozen")
    flag_log10_v_sini_frozen = frozen_flags.flag(2**3, "Rotational broadening is frozen")
    flag_log10_v_micro_frozen = frozen_flags.flag(2**4, "Microturbulence is frozen")
    flag_alpha_m_frozen = frozen_flags.flag(2**5, "[alpha/M] is frozen")
    flag_c_m_frozen = frozen_flags.flag(2**6, "[C/M] is frozen")
    flag_n_m_frozen = frozen_flags.flag(2**7, "[N/M] is frozen")

    #> Stellar Parameters
    teff = FloatField(null=True)
    e_teff = FloatField(null=True)
    logg = FloatField(null=True)
    e_logg = FloatField(null=True)
    m_h = FloatField(null=True)
    e_m_h = FloatField(null=True)
    log10_v_sini = FloatField(null=True)
    e_log10_v_sini = FloatField(null=True)
    log10_v_micro = FloatField(null=True)
    e_log10_v_micro = FloatField(null=True)
    alpha_m = FloatField(null=True)
    e_alpha_m = FloatField(null=True)
    c_m = FloatField(null=True)
    e_c_m = FloatField(null=True)
    n_m = FloatField(null=True)
    e_n_m = FloatField(null=True)

    teff_flags = BitField(default=0)
    logg_flags = BitField(default=0)
    m_h_flags = BitField(default=0)
    log10_v_sini_flags = BitField(default=0)
    log10_v_micro_flags = BitField(default=0)
    alpha_m_flags = BitField(default=0)
    c_m_flags = BitField(default=0)
    n_m_flags = BitField(default=0)

    # TODO: flag definitions for each dimension (DRY)
    #> FERRE Access Fields
    ferre_name = TextField(default="")
    ferre_input_index = IntegerField(default=-1)
    ferre_output_index = IntegerField(default=-1)
    ferre_n_obj = IntegerField(default=-1)

    #> Summary Statistics
    chisq = FloatField(null=True)
    ferre_log_snr_sq = FloatField(null=True)
    ferre_log_chisq = FloatField(null=True)
    ferre_frac_phot_data_points = FloatField(default=0)
    ferre_log_penalized_chisq = FloatField(null=True)
    ferre_time_load_grid = FloatField(null=True)
    ferre_time_elapsed = FloatField(null=True)
    ferre_flags = BitField(default=0)
    
    flag_ferre_fail = ferre_flags.flag(2**0, "FERRE failed")
    flag_missing_model_flux = ferre_flags.flag(2**1, "Missing model fluxes from FERRE")
    flag_potential_ferre_timeout = ferre_flags.flag(2**2, "Potentially impacted by FERRE timeout")
    flag_no_suitable_initial_guess = ferre_flags.flag(2**3, help_text="FERRE not executed because there's no suitable initial guess")



class ASPCAP(BaseModel, PipelineOutputMixin):

    sdss_id = ForeignKeyField(Source, index=True, lazy_load=False)
    spectrum_id = ForeignKeyField(Spectrum, index=True, lazy_load=False)
    ferre_stellar_parameters_id = ForeignKeyField(FerreStellarParameters, index=True, lazy_load=False)

    #> Astra Metadata
    task_id = AutoField()
    v_astra = TextField(default=__version__)
    t_elapsed = FloatField(null=True)
    tag = TextField(default="", index=True)

    short_grid_name = TextField(default="")
    

    #> Stellar Parameters
    teff = FloatField(null=True)
    e_teff = FloatField(null=True)
    logg = FloatField(null=True)
    e_logg = FloatField(null=True)
    v_micro = FloatField(null=True)
    e_v_micro = FloatField(null=True)
    v_sini = FloatField(null=True)
    e_v_sini = FloatField(null=True)
    m_h_atm = FloatField(null=True)
    e_m_h_atm = FloatField(null=True)
    alpha_m_atm = FloatField(null=True)
    e_alpha_m_atm = FloatField(null=True)
    c_m_atm = FloatField(null=True)
    e_c_m_atm = FloatField(null=True)
    n_m_atm = FloatField(null=True)
    e_n_m_atm = FloatField(null=True)

    #> FERRE Settings
    continuum_order = IntegerField(default=-1)
    continuum_reject = FloatField(null=True)
    interpolation_order = IntegerField(default=-1)
    frozen_flags = BitField(default=0)
    f_access = IntegerField(default=-1)
    f_format = IntegerField(default=-1)
    n_threads = IntegerField(default=-1)

    #
    #initial_flags 

