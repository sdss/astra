from peewee import (
    AutoField,
    FloatField,
    TextField,
    ForeignKeyField,
    IntegerField,
    BitField,
    BooleanField,
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

    @cached_property
    def e_rectified_flux(self):
        continuum = self.ferre_flux / self.rectified_flux
        return self.ferre_e_flux / continuum

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
        from astra.pipelines.ferre.utils import parse_ferre_spectrum_name, get_ferre_spectrum_name
        
        #assert self.ferre_input_index >= 0

        kwds = dict(
            fname=f"{self.pwd}/{basename}",
            skiprows=int(self.ferre_input_index), 
            max_rows=1,
        )
        '''
        try:
            name, = np.atleast_1d(np.loadtxt(usecols=(0, ), dtype=str, **kwds))
            array = np.loadtxt(usecols=range(1, 1+P), **kwds)

            meta = parse_ferre_spectrum_name(name)
            if (
                (int(meta["source_id"]) != self.source_id)
            or (int(meta["spectrum_id"]) != self.spectrum_id)
            or (int(meta["index"]) != self.ferre_input_index)
            ):
                raise a
        except:
            del kwds["skiprows"]
            del kwds["max_rows"]

            name = get_ferre_spectrum_name(self.ferre_input_index, self.source_id, self.spectrum_id, self.initial_flags, self.upstream_id)

            index = list(np.loadtxt(usecols=(0, ), dtype=str, **kwds)).index(name)
            self.ferre_output_index = index
            self.save()
            print("saved!")
            kwds["skiprows"] = index
            kwds["max_rows"] = 1

            name, = np.atleast_1d(np.loadtxt(usecols=(0, ), dtype=str, **kwds))
            array = np.loadtxt(usecols=range(1, 1+P), **kwds)

        '''
        name, = np.atleast_1d(np.loadtxt(usecols=(0, ), dtype=str, **kwds))
        array = np.loadtxt(usecols=range(1, 1+P), **kwds)

        meta = parse_ferre_spectrum_name(name)
        assert int(meta["source_id"]) == self.source_id
        assert int(meta["spectrum_id"]) == self.spectrum_id
        assert int(meta["index"]) == self.ferre_input_index
        return array


class FerreCoarse(BaseModel, FerreOutputMixin):

    source_id = ForeignKeyField(Source, index=True, lazy_load=False)
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
    flag_initial_guess_at_grid_center = initial_flags.flag(2**3, help_text="Initial guess from grid center")

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

    # TODO: Is there a way to inherit these or assign these dynamically so we don't repeat ourselves?
    flag_teff_ferre_fail = teff_flags.flag(2**0)
    flag_teff_grid_edge_warn = teff_flags.flag(2**1)
    flag_teff_grid_edge_bad = teff_flags.flag(2**2)
    flag_logg_ferre_fail = logg_flags.flag(2**0)
    flag_logg_grid_edge_warn = logg_flags.flag(2**1)
    flag_logg_grid_edge_bad = logg_flags.flag(2**2)
    flag_m_h_ferre_fail = m_h_flags.flag(2**0)
    flag_m_h_grid_edge_warn = m_h_flags.flag(2**1)
    flag_m_h_grid_edge_bad = m_h_flags.flag(2**2)
    flag_log10_v_sini_ferre_fail = log10_v_sini_flags.flag(2**0)
    flag_log10_v_sini_grid_edge_warn = log10_v_sini_flags.flag(2**1)
    flag_log10_v_sini_grid_edge_bad = log10_v_sini_flags.flag(2**2)
    flag_log10_v_micro_ferre_fail = log10_v_micro_flags.flag(2**0)
    flag_log10_v_micro_grid_edge_warn = log10_v_micro_flags.flag(2**1)
    flag_log10_v_micro_grid_edge_bad = log10_v_micro_flags.flag(2**2)
    flag_alpha_m_ferre_fail = alpha_m_flags.flag(2**0)
    flag_alpha_m_grid_edge_warn = alpha_m_flags.flag(2**1)
    flag_alpha_m_grid_edge_bad = alpha_m_flags.flag(2**2)
    flag_c_m_ferre_fail = c_m_flags.flag(2**0)
    flag_c_m_grid_edge_warn = c_m_flags.flag(2**1)
    flag_c_m_grid_edge_bad = c_m_flags.flag(2**2)
    flag_n_m_ferre_fail = n_m_flags.flag(2**0)
    flag_n_m_grid_edge_warn = n_m_flags.flag(2**1)
    flag_n_m_grid_edge_bad = n_m_flags.flag(2**2)    

    #> FERRE Access Fields
    ferre_name = TextField(default="")
    ferre_input_index = IntegerField(default=-1)
    ferre_output_index = IntegerField(default=-1)
    ferre_n_obj = IntegerField(default=-1)

    #> Summary Statistics
    snr = FloatField(null=True)
    r_chi_sq = FloatField(null=True)
    penalized_r_chi_sq = FloatField(null=True) 
    ferre_log_snr_sq = FloatField(null=True)
    ferre_time_load_grid = FloatField(null=True)
    ferre_time_elapsed = FloatField(null=True)
    ferre_flags = BitField(default=0)

    flag_ferre_fail = ferre_flags.flag(2**0, "FERRE failed")
    flag_missing_model_flux = ferre_flags.flag(2**1, "Missing model fluxes from FERRE")
    flag_potential_ferre_timeout = ferre_flags.flag(2**2, "Potentially impacted by FERRE timeout")
    flag_no_suitable_initial_guess = ferre_flags.flag(2**3, help_text="FERRE not executed because there's no suitable initial guess")
    flag_spectrum_io_error = ferre_flags.flag(2**4, help_text="Error accessing spectrum pixel data")



class FerreStellarParameters(BaseModel, FerreOutputMixin):

    source_id = ForeignKeyField(Source, index=True, lazy_load=False)
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

    # Define flags.
    flag_teff_ferre_fail = teff_flags.flag(2**0)
    flag_teff_grid_edge_warn = teff_flags.flag(2**1)
    flag_teff_grid_edge_bad = teff_flags.flag(2**2)
    flag_logg_ferre_fail = logg_flags.flag(2**0)
    flag_logg_grid_edge_warn = logg_flags.flag(2**1)
    flag_logg_grid_edge_bad = logg_flags.flag(2**2)
    flag_m_h_ferre_fail = m_h_flags.flag(2**0)
    flag_m_h_grid_edge_warn = m_h_flags.flag(2**1)
    flag_m_h_grid_edge_bad = m_h_flags.flag(2**2)
    flag_log10_v_sini_ferre_fail = log10_v_sini_flags.flag(2**0)
    flag_log10_v_sini_grid_edge_warn = log10_v_sini_flags.flag(2**1)
    flag_log10_v_sini_grid_edge_bad = log10_v_sini_flags.flag(2**2)
    flag_log10_v_micro_ferre_fail = log10_v_micro_flags.flag(2**0)
    flag_log10_v_micro_grid_edge_warn = log10_v_micro_flags.flag(2**1)
    flag_log10_v_micro_grid_edge_bad = log10_v_micro_flags.flag(2**2)
    flag_alpha_m_ferre_fail = alpha_m_flags.flag(2**0)
    flag_alpha_m_grid_edge_warn = alpha_m_flags.flag(2**1)
    flag_alpha_m_grid_edge_bad = alpha_m_flags.flag(2**2)
    flag_c_m_ferre_fail = c_m_flags.flag(2**0)
    flag_c_m_grid_edge_warn = c_m_flags.flag(2**1)
    flag_c_m_grid_edge_bad = c_m_flags.flag(2**2)
    flag_n_m_ferre_fail = n_m_flags.flag(2**0)
    flag_n_m_grid_edge_warn = n_m_flags.flag(2**1)
    flag_n_m_grid_edge_bad = n_m_flags.flag(2**2)


    # TODO: flag definitions for each dimension (DRY)
    #> FERRE Access Fields
    ferre_name = TextField(default="")
    ferre_input_index = IntegerField(default=-1)
    ferre_output_index = IntegerField(default=-1)
    ferre_n_obj = IntegerField(default=-1)

    #> Summary Statistics
    snr = FloatField(null=True)
    r_chi_sq = FloatField(null=True)
    penalized_r_chi_sq = FloatField(null=True)
    ferre_log_snr_sq = FloatField(null=True)
    ferre_time_load_grid = FloatField(null=True)
    ferre_time_elapsed = FloatField(null=True)
    ferre_flags = BitField(default=0)
    
    flag_ferre_fail = ferre_flags.flag(2**0, "FERRE failed")
    flag_missing_model_flux = ferre_flags.flag(2**1, "Missing model fluxes from FERRE")
    flag_potential_ferre_timeout = ferre_flags.flag(2**2, "Potentially impacted by FERRE timeout")
    flag_no_suitable_initial_guess = ferre_flags.flag(2**3, help_text="FERRE not executed because there's no suitable initial guess")



class FerreChemicalAbundances(BaseModel, FerreOutputMixin):

    source_id = ForeignKeyField(Source, index=True, lazy_load=False)
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

    initial_flags = BitField(default=0)
    # TODO: Not sure what flag definitions are needed for initial guess.

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

    # Define flags.
    flag_teff_ferre_fail = teff_flags.flag(2**0)
    flag_teff_grid_edge_warn = teff_flags.flag(2**1)
    flag_teff_grid_edge_bad = teff_flags.flag(2**2)
    flag_logg_ferre_fail = logg_flags.flag(2**0)
    flag_logg_grid_edge_warn = logg_flags.flag(2**1)
    flag_logg_grid_edge_bad = logg_flags.flag(2**2)
    flag_m_h_ferre_fail = m_h_flags.flag(2**0)
    flag_m_h_grid_edge_warn = m_h_flags.flag(2**1)
    flag_m_h_grid_edge_bad = m_h_flags.flag(2**2)
    flag_log10_v_sini_ferre_fail = log10_v_sini_flags.flag(2**0)
    flag_log10_v_sini_grid_edge_warn = log10_v_sini_flags.flag(2**1)
    flag_log10_v_sini_grid_edge_bad = log10_v_sini_flags.flag(2**2)
    flag_log10_v_micro_ferre_fail = log10_v_micro_flags.flag(2**0)
    flag_log10_v_micro_grid_edge_warn = log10_v_micro_flags.flag(2**1)
    flag_log10_v_micro_grid_edge_bad = log10_v_micro_flags.flag(2**2)
    flag_alpha_m_ferre_fail = alpha_m_flags.flag(2**0)
    flag_alpha_m_grid_edge_warn = alpha_m_flags.flag(2**1)
    flag_alpha_m_grid_edge_bad = alpha_m_flags.flag(2**2)
    flag_c_m_ferre_fail = c_m_flags.flag(2**0)
    flag_c_m_grid_edge_warn = c_m_flags.flag(2**1)
    flag_c_m_grid_edge_bad = c_m_flags.flag(2**2)
    flag_n_m_ferre_fail = n_m_flags.flag(2**0)
    flag_n_m_grid_edge_warn = n_m_flags.flag(2**1)
    flag_n_m_grid_edge_bad = n_m_flags.flag(2**2)


    # TODO: flag definitions for each dimension (DRY)
    #> FERRE Access Fields
    ferre_name = TextField(default="")
    ferre_input_index = IntegerField(default=-1)
    ferre_output_index = IntegerField(default=-1)
    ferre_n_obj = IntegerField(default=-1)

    #> Summary Statistics
    snr = FloatField(null=True)
    r_chi_sq = FloatField(null=True)
    penalized_r_chi_sq = FloatField(null=True)
    ferre_log_snr_sq = FloatField(null=True)
    ferre_time_load_grid = FloatField(null=True)
    ferre_time_elapsed = FloatField(null=True)
    ferre_flags = BitField(default=0)
    
    flag_ferre_fail = ferre_flags.flag(2**0, "FERRE failed")
    flag_missing_model_flux = ferre_flags.flag(2**1, "Missing model fluxes from FERRE")
    flag_potential_ferre_timeout = ferre_flags.flag(2**2, "Potentially impacted by FERRE timeout")
    flag_no_suitable_initial_guess = ferre_flags.flag(2**3, help_text="FERRE not executed because there's no suitable initial guess")




class ASPCAP(BaseModel, PipelineOutputMixin):

    source_id = ForeignKeyField(Source, index=True, lazy_load=False)
    spectrum_id = ForeignKeyField(Spectrum, index=True, lazy_load=False)

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

    #> Abundances
    al_h = FloatField(null=True)
    e_al_h = FloatField(null=True)
    al_h_flags = BitField(default=0)
    al_h_r_chi_sq = FloatField(null=True)

    c_12_13 = FloatField(null=True)
    e_c_12_13 = FloatField(null=True)
    c_12_13_flags = BitField(default=0)
    c_12_13_r_chi_sq = FloatField(null=True)

    ca_h = FloatField(null=True)
    e_ca_h = FloatField(null=True)
    ca_h_flags = BitField(default=0)
    ca_h_r_chi_sq = FloatField(null=True)
    
    ce_h = FloatField(null=True)
    e_ce_h = FloatField(null=True)
    ce_h_flags = BitField(default=0)
    ce_h_r_chi_sq = FloatField(null=True)
    
    c_1_h = FloatField(null=True)
    e_c_1_h = FloatField(null=True)
    c_1_h_flags = BitField(default=0)
    c_1_h_r_chi_sq = FloatField(null=True)
    
    c_h = FloatField(null=True)
    e_c_h = FloatField(null=True)
    c_h_flags = BitField(default=0)
    c_h_r_chi_sq = FloatField(null=True)
    
    co_h = FloatField(null=True)
    e_co_h = FloatField(null=True)
    co_h_flags = BitField(default=0)
    co_h_r_chi_sq = FloatField(null=True)
    
    cr_h = FloatField(null=True)
    e_cr_h = FloatField(null=True)
    cr_h_flags = BitField(default=0)
    cr_h_r_chi_sq = FloatField(null=True)
    
    cu_h = FloatField(null=True)
    e_cu_h = FloatField(null=True)
    cu_h_flags = BitField(default=0)
    cu_h_r_chi_sq = FloatField(null=True)
    
    fe_h = FloatField(null=True)
    e_fe_h = FloatField(null=True)
    fe_h_flags = BitField(default=0)
    fe_h_r_chi_sq = FloatField(null=True)

    k_h = FloatField(null=True)
    e_k_h = FloatField(null=True)
    k_h_flags = BitField(default=0)
    k_h_r_chi_sq = FloatField(null=True)

    mg_h = FloatField(null=True)
    e_mg_h = FloatField(null=True)
    mg_h_flags = BitField(default=0)
    mg_h_r_chi_sq = FloatField(null=True)

    mn_h = FloatField(null=True)
    e_mn_h = FloatField(null=True)
    mn_h_flags = BitField(default=0)
    mn_h_r_chi_sq = FloatField(null=True)

    na_h = FloatField(null=True)
    e_na_h = FloatField(null=True)
    na_h_flags = BitField(default=0)
    na_h_r_chi_sq = FloatField(null=True)

    nd_h = FloatField(null=True)
    e_nd_h = FloatField(null=True)
    nd_h_flags = BitField(default=0)
    nd_h_r_chi_sq = FloatField(null=True)

    ni_h = FloatField(null=True)
    e_ni_h = FloatField(null=True)
    ni_h_flags = BitField(default=0)
    ni_h_r_chi_sq = FloatField(null=True)

    n_h = FloatField(null=True)
    e_n_h = FloatField(null=True)
    n_h_flags = BitField(default=0)
    n_h_r_chi_sq = FloatField(null=True)

    o_h = FloatField(null=True)
    e_o_h = FloatField(null=True)
    o_h_flags = BitField(default=0)
    o_h_r_chi_sq = FloatField(null=True)

    p_h = FloatField(null=True)
    e_p_h = FloatField(null=True)
    p_h_flags = BitField(default=0)
    p_h_r_chi_sq = FloatField(null=True)

    si_h = FloatField(null=True)
    e_si_h = FloatField(null=True)
    si_h_flags = BitField(default=0)
    si_h_r_chi_sq = FloatField(null=True)

    s_h = FloatField(null=True)
    e_s_h = FloatField(null=True)
    s_h_flags = BitField(default=0)
    s_h_r_chi_sq = FloatField(null=True)

    ti_h = FloatField(null=True)
    e_ti_h = FloatField(null=True)
    ti_h_flags = BitField(default=0)
    ti_h_r_chi_sq = FloatField(null=True)

    ti_2_h = FloatField(null=True)
    e_ti_2_h = FloatField(null=True)
    ti_2_h_flags = BitField(default=0)
    ti_2_h_r_chi_sq = FloatField(null=True)

    v_h = FloatField(null=True)
    e_v_h = FloatField(null=True)
    v_h_flags = BitField(default=0)
    v_h_r_chi_sq = FloatField(null=True)

    # TODO: include the initial parameters from the stellar parameter run? Or the COARSE run?

    #> Sheldon's Fun With Flags
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

    #> Summary Statistics
    snr = FloatField(null=True)
    r_chi_sq = FloatField(null=True)
    ferre_log_snr_sq = FloatField(null=True)
    ferre_flags = BitField(default=0)
    
    flag_ferre_fail = ferre_flags.flag(2**0, "FERRE failed")
    flag_missing_model_flux = ferre_flags.flag(2**1, "Missing model fluxes from FERRE")
    flag_potential_ferre_timeout = ferre_flags.flag(2**2, "Potentially impacted by FERRE timeout")
    flag_no_suitable_initial_guess = ferre_flags.flag(2**3, help_text="FERRE not executed because there's no suitable initial guess")
    flag_teff_grid_edge_warn = ferre_flags.flag(2**4)
    flag_teff_grid_edge_bad = ferre_flags.flag(2**5)
    flag_logg_grid_edge_warn = ferre_flags.flag(2**6)
    flag_logg_grid_edge_bad = ferre_flags.flag(2**7)
    flag_v_micro_grid_edge_warn = ferre_flags.flag(2**8)
    flag_v_micro_grid_edge_bad = ferre_flags.flag(2**9)
    flag_v_sini_grid_edge_warn = ferre_flags.flag(2**10)
    flag_v_sini_grid_edge_bad = ferre_flags.flag(2**11)
    flag_m_h_atm_grid_edge_warn = ferre_flags.flag(2**12)
    flag_m_h_atm_grid_edge_bad = ferre_flags.flag(2**13)
    flag_alpha_m_grid_edge_warn = ferre_flags.flag(2**14)
    flag_alpha_m_grid_edge_bad = ferre_flags.flag(2**15)
    flag_c_m_atm_grid_edge_warn = ferre_flags.flag(2**16)
    flag_c_m_atm_grid_edge_bad = ferre_flags.flag(2**17)
    flag_n_m_atm_grid_edge_warn = ferre_flags.flag(2**18)
    flag_n_m_atm_grid_edge_bad = ferre_flags.flag(2**19)
    
    # TODO: Should we store these here like this, or some othe way?

    #> Task Identifiers
    stellar_parameters_task_id = ForeignKeyField(FerreStellarParameters, null=True, lazy_load=False)
    al_h_task_id = ForeignKeyField(FerreChemicalAbundances, null=True, lazy_load=False)
    c_12_13_task_id = ForeignKeyField(FerreChemicalAbundances, null=True, lazy_load=False)
    ca_h_task_id = ForeignKeyField(FerreChemicalAbundances, null=True, lazy_load=False)
    ce_h_task_id = ForeignKeyField(FerreChemicalAbundances, null=True, lazy_load=False)
    c_1_h_task_id = ForeignKeyField(FerreChemicalAbundances, null=True, lazy_load=False)
    c_h_task_id = ForeignKeyField(FerreChemicalAbundances, null=True, lazy_load=False)
    co_h_task_id = ForeignKeyField(FerreChemicalAbundances, null=True, lazy_load=False)
    cr_h_task_id = ForeignKeyField(FerreChemicalAbundances, null=True, lazy_load=False)
    cu_h_task_id = ForeignKeyField(FerreChemicalAbundances, null=True, lazy_load=False)
    fe_h_task_id = ForeignKeyField(FerreChemicalAbundances, null=True, lazy_load=False)
    k_h_task_id = ForeignKeyField(FerreChemicalAbundances, null=True, lazy_load=False)
    mg_h_task_id = ForeignKeyField(FerreChemicalAbundances, null=True, lazy_load=False)
    mn_h_task_id = ForeignKeyField(FerreChemicalAbundances, null=True, lazy_load=False)
    na_h_task_id = ForeignKeyField(FerreChemicalAbundances, null=True, lazy_load=False)
    nd_h_task_id = ForeignKeyField(FerreChemicalAbundances, null=True, lazy_load=False)
    ni_h_task_id = ForeignKeyField(FerreChemicalAbundances, null=True, lazy_load=False)
    n_h_task_id = ForeignKeyField(FerreChemicalAbundances, null=True, lazy_load=False)
    o_h_task_id = ForeignKeyField(FerreChemicalAbundances, null=True, lazy_load=False)
    p_h_task_id = ForeignKeyField(FerreChemicalAbundances, null=True, lazy_load=False)
    si_h_task_id = ForeignKeyField(FerreChemicalAbundances, null=True, lazy_load=False)
    s_h_task_id = ForeignKeyField(FerreChemicalAbundances, null=True, lazy_load=False)
    ti_h_task_id = ForeignKeyField(FerreChemicalAbundances, null=True, lazy_load=False)
    ti_2_h_task_id = ForeignKeyField(FerreChemicalAbundances, null=True, lazy_load=False)
    v_h_task_id = ForeignKeyField(FerreChemicalAbundances, null=True, lazy_load=False)


    #> Raw (Uncalibrated) Quantities
    calibrated = BooleanField(default=False)
    raw_teff = FloatField(null=True)
    raw_e_teff = FloatField(null=True)
    raw_logg = FloatField(null=True)
    raw_e_logg = FloatField(null=True)
    raw_v_micro = FloatField(null=True)
    raw_e_v_micro = FloatField(null=True)
    raw_v_sini = FloatField(null=True)
    raw_e_v_sini = FloatField(null=True)
    raw_m_h_atm = FloatField(null=True)
    raw_e_m_h_atm = FloatField(null=True)
    raw_alpha_m_atm = FloatField(null=True)
    raw_e_alpha_m_atm = FloatField(null=True)
    raw_c_m_atm = FloatField(null=True)
    raw_e_c_m_atm = FloatField(null=True)
    raw_n_m_atm = FloatField(null=True)
    raw_e_n_m_atm = FloatField(null=True)
    raw_al_h = FloatField(null=True)
    raw_e_al_h = FloatField(null=True)
    raw_c_12_13 = FloatField(null=True)
    raw_e_c_12_13 = FloatField(null=True)
    raw_ca_h = FloatField(null=True)
    raw_e_ca_h = FloatField(null=True)    
    raw_ce_h = FloatField(null=True)
    raw_e_ce_h = FloatField(null=True)    
    raw_c_1_h = FloatField(null=True)
    raw_e_c_1_h = FloatField(null=True)    
    raw_c_h = FloatField(null=True)
    raw_e_c_h = FloatField(null=True)    
    raw_co_h = FloatField(null=True)
    raw_e_co_h = FloatField(null=True)    
    raw_cr_h = FloatField(null=True)
    raw_e_cr_h = FloatField(null=True)    
    raw_cu_h = FloatField(null=True)
    raw_e_cu_h = FloatField(null=True)    
    raw_fe_h = FloatField(null=True)
    raw_e_fe_h = FloatField(null=True)
    raw_k_h = FloatField(null=True)
    raw_e_k_h = FloatField(null=True)
    raw_mg_h = FloatField(null=True)
    raw_e_mg_h = FloatField(null=True)
    raw_mn_h = FloatField(null=True)
    raw_e_mn_h = FloatField(null=True)
    raw_na_h = FloatField(null=True)
    raw_e_na_h = FloatField(null=True)
    raw_nd_h = FloatField(null=True)
    raw_e_nd_h = FloatField(null=True)
    raw_ni_h = FloatField(null=True)
    raw_e_ni_h = FloatField(null=True)
    raw_n_h = FloatField(null=True)
    raw_e_n_h = FloatField(null=True)
    raw_o_h = FloatField(null=True)
    raw_e_o_h = FloatField(null=True)
    raw_p_h = FloatField(null=True)
    raw_e_p_h = FloatField(null=True)
    raw_si_h = FloatField(null=True)
    raw_e_si_h = FloatField(null=True)
    raw_s_h = FloatField(null=True)
    raw_e_s_h = FloatField(null=True)
    raw_ti_h = FloatField(null=True)
    raw_e_ti_h = FloatField(null=True)
    raw_ti_2_h = FloatField(null=True)
    raw_e_ti_2_h = FloatField(null=True)
    raw_v_h = FloatField(null=True)
    raw_e_v_h = FloatField(null=True)
    