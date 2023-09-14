from peewee import (
    AutoField,
    FloatField,
    TextField,
    ForeignKeyField,
    IntegerField,
    BitField,
    DateTimeField,
    BooleanField,
)

import datetime
import numpy as np
from functools import cached_property

from astra import __version__
from astra.models.base import BaseModel
from astra.models.fields import BitField, PixelArray, BasePixelArrayAccessor
from astra.models.source import Source
from astra.models.spectrum import Spectrum
from astra.models.pipeline import PipelineOutputMixin
from astra.glossary import Glossary


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
                (int(meta["source_pk"]) != self.source_pk)
            or (int(meta["spectrum_pk"]) != self.spectrum_pk)
            or (int(meta["index"]) != self.ferre_input_index)
            ):
                raise a
        except:
            del kwds["skiprows"]
            del kwds["max_rows"]

            name = get_ferre_spectrum_name(self.ferre_input_index, self.source_pk, self.spectrum_pk, self.initial_flags, self.upstream_id)

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
        assert int(meta["source_pk"]) == self.source_pk
        assert int(meta["spectrum_pk"]) == self.spectrum_pk
        assert int(meta["index"]) == self.ferre_input_index
        return array


class FerreCoarse(BaseModel, FerreOutputMixin):

    source_pk = ForeignKeyField(Source, index=True, lazy_load=False)
    spectrum_pk = ForeignKeyField(Spectrum, index=True, lazy_load=False)
    
    #> Astra Metadata
    task_pk = AutoField()
    v_astra = TextField(default=__version__)
    created = DateTimeField(default=datetime.datetime.now)
    t_elapsed = FloatField(null=True)
    t_overhead = FloatField(null=True)
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
    continuum_order = IntegerField(default=-1, null=True)
    continuum_reject = FloatField(null=True)
    continuum_flag = IntegerField(default=0, null=True)
    continuum_observations_flag = IntegerField(default=0, null=True)
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
    rchi2 = FloatField(null=True)
    penalized_rchi2 = FloatField(null=True) 
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

    source_pk = ForeignKeyField(Source, index=True, lazy_load=False)
    spectrum_pk = ForeignKeyField(Spectrum, index=True, lazy_load=False)
    upstream = ForeignKeyField(FerreCoarse, index=True)

    #> Astra Metadata
    task_pk = AutoField()
    v_astra = TextField(default=__version__)
    created = DateTimeField(default=datetime.datetime.now)
    t_elapsed = FloatField(null=True)
    t_overhead = FloatField(null=True)
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
    continuum_flag = IntegerField(default=0, null=True)
    continuum_observations_flag = IntegerField(default=0, null=True)
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
    rchi2 = FloatField(null=True)
    penalized_rchi2 = FloatField(null=True)
    ferre_log_snr_sq = FloatField(null=True)
    ferre_time_load_grid = FloatField(null=True)
    ferre_time_elapsed = FloatField(null=True)
    ferre_flags = BitField(default=0)
    
    flag_ferre_fail = ferre_flags.flag(2**0, "FERRE failed")
    flag_missing_model_flux = ferre_flags.flag(2**1, "Missing model fluxes from FERRE")
    flag_potential_ferre_timeout = ferre_flags.flag(2**2, "Potentially impacted by FERRE timeout")
    flag_no_suitable_initial_guess = ferre_flags.flag(2**3, help_text="FERRE not executed because there's no suitable initial guess")



class FerreChemicalAbundances(BaseModel, FerreOutputMixin):

    source_pk = ForeignKeyField(Source, index=True, lazy_load=False)
    spectrum_pk = ForeignKeyField(Spectrum, index=True, lazy_load=False)
    upstream = ForeignKeyField(FerreStellarParameters, index=True)

    #> Astra Metadata
    task_pk = AutoField()
    v_astra = TextField(default=__version__)
    created = DateTimeField(default=datetime.datetime.now)
    t_elapsed = FloatField(null=True)
    t_overhead = FloatField(null=True)
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
    continuum_flag = IntegerField(default=0, null=True)
    continuum_observations_flag = IntegerField(default=0, null=True)
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
    rchi2 = FloatField(null=True)
    penalized_rchi2 = FloatField(null=True)
    ferre_log_snr_sq = FloatField(null=True)
    ferre_time_load_grid = FloatField(null=True)
    ferre_time_elapsed = FloatField(null=True)
    ferre_flags = BitField(default=0)
    
    flag_ferre_fail = ferre_flags.flag(2**0, "FERRE failed")
    flag_missing_model_flux = ferre_flags.flag(2**1, "Missing model fluxes from FERRE")
    flag_potential_ferre_timeout = ferre_flags.flag(2**2, "Potentially impacted by FERRE timeout")
    flag_no_suitable_initial_guess = ferre_flags.flag(2**3, help_text="FERRE not executed because there's no suitable initial guess")



class UpstreamFerrePixelArrayAccessor(BasePixelArrayAccessor):
    
    def __get__(self, instance, instance_type=None):
        if instance is not None:
            try:
                return instance.__pixel_data__[self.name]
            except (AttributeError, KeyError):
                # Load them all.
                instance.__pixel_data__ = {}

                # TODO: THIS IS SO SLOW AND SUCH A HACK
                upstream = FerreStellarParameters.get(instance.stellar_parameters_task_pk)
                instance.__pixel_data__.setdefault("model_flux", upstream.unmask(upstream.model_flux))
                instance.__pixel_data__.setdefault("continuum", upstream.unmask(upstream.ferre_flux / upstream.rectified_flux))
                
                return instance.__pixel_data__[self.name]

        return self.field



class MG_HFerrePixelArrayAccessor(BasePixelArrayAccessor):
    
    def __get__(self, instance, instance_type=None):
        if instance is not None:
            try:
                return instance.__pixel_data__[self.name]
            except (AttributeError, KeyError):
                # Load them all.
                instance.__pixel_data__ = {}

                # TODO: THIS IS SO SLOW AND SUCH A HACK
                upstream = FerreChemicalAbundances.get(instance.ti_h_task_pk)
                try:
                    instance.__pixel_data__.setdefault("model_flux", upstream.unmask(upstream.model_flux))
                    instance.__pixel_data__.setdefault("continuum", upstream.unmask(upstream.ferre_flux / upstream.rectified_flux))
                except:
                    instance.__pixel_data__["model_flux"] = np.nan * np.ones(8575)
                    instance.__pixel_data__["continuum"] = np.nan * np.ones(8575)

                return instance.__pixel_data__[self.name]

        return self.field


class ASPCAP(BaseModel, PipelineOutputMixin):

    """ APOGEE Stellar Parameter and Chemical Abundances Pipeline (ASPCAP) """

    source_pk = ForeignKeyField(Source, index=True, lazy_load=False)
    spectrum_pk = ForeignKeyField(Spectrum, index=True, lazy_load=False, help_text=Glossary.spectrum_pk)    
    
    #> Astra Metadata
    task_pk = AutoField(help_text=Glossary.task_pk)
    v_astra = TextField(default=__version__, help_text=Glossary.v_astra)
    created = DateTimeField(default=datetime.datetime.now) # TODO: ADD
    t_elapsed = FloatField(null=True, help_text=Glossary.t_elapsed)
    t_overhead = FloatField(null=True)
    tag = TextField(default="", index=True, help_text=Glossary.tag)
    
    #> Spectral Data
    model_flux = PixelArray(accessor_class=UpstreamFerrePixelArrayAccessor, pixels=8575)
    continuum = PixelArray(accessor_class=UpstreamFerrePixelArrayAccessor, pixels=8575)    
    # TODO: Add other model fluxes
    #ti_hmodel_flux = PixelArray(accessor_class=MG_HFerrePixelArrayAccessor)

    #> Stellar Parameters
    teff = FloatField(null=True, help_text=Glossary.teff)
    e_teff = FloatField(null=True, help_text=Glossary.e_teff)
    logg = FloatField(null=True, help_text=Glossary.logg)
    e_logg = FloatField(null=True, help_text=Glossary.e_logg)
    v_micro = FloatField(null=True, help_text=Glossary.v_micro)
    e_v_micro = FloatField(null=True, help_text=Glossary.e_v_micro)
    v_sini = FloatField(null=True, help_text=Glossary.v_sini)
    e_v_sini = FloatField(null=True, help_text=Glossary.e_v_sini)
    m_h_atm = FloatField(null=True, help_text=Glossary.m_h_atm)
    e_m_h_atm = FloatField(null=True, help_text=Glossary.e_m_h_atm)
    alpha_m_atm = FloatField(null=True, help_text=Glossary.alpha_m_atm)
    e_alpha_m_atm = FloatField(null=True, help_text=Glossary.e_alpha_m_atm)
    c_m_atm = FloatField(null=True, help_text=Glossary.c_m_atm)
    e_c_m_atm = FloatField(null=True, help_text=Glossary.e_c_m_atm)
    n_m_atm = FloatField(null=True, help_text=Glossary.n_m_atm)
    e_n_m_atm = FloatField(null=True, help_text=Glossary.e_n_m_atm)

    #> Chemical Abundances
    al_h = FloatField(null=True, help_text=Glossary.al_h)
    e_al_h = FloatField(null=True, help_text=Glossary.e_al_h)
    al_h_flags = BitField(default=0, help_text=Glossary.al_h_flags)
    al_h_rchi2 = FloatField(null=True, help_text=Glossary.al_h_rchi2)

    c_12_13 = FloatField(null=True, help_text=Glossary.c_12_13)
    e_c_12_13 = FloatField(null=True, help_text=Glossary.e_c_12_13)
    c_12_13_flags = BitField(default=0, help_text=Glossary.c_12_13_flags)
    c_12_13_rchi2 = FloatField(null=True, help_text=Glossary.c_12_13_rchi2)

    ca_h = FloatField(null=True, help_text=Glossary.ca_h)
    e_ca_h = FloatField(null=True, help_text=Glossary.e_ca_h)
    ca_h_flags = BitField(default=0, help_text=Glossary.ca_h_flags)
    ca_h_rchi2 = FloatField(null=True, help_text=Glossary.ca_h_rchi2)
    
    ce_h = FloatField(null=True, help_text=Glossary.ce_h)
    e_ce_h = FloatField(null=True, help_text=Glossary.e_ce_h)
    ce_h_flags = BitField(default=0, help_text=Glossary.ce_h_flags)
    ce_h_rchi2 = FloatField(null=True, help_text=Glossary.ce_h_rchi2)
    
    c_1_h = FloatField(null=True, help_text=Glossary.c_1_h)
    e_c_1_h = FloatField(null=True, help_text=Glossary.e_c_1_h)
    c_1_h_flags = BitField(default=0, help_text=Glossary.c_1_h_flags)
    c_1_h_rchi2 = FloatField(null=True, help_text=Glossary.c_1_h_rchi2)
    
    c_h = FloatField(null=True, help_text=Glossary.c_h)
    e_c_h = FloatField(null=True, help_text=Glossary.e_c_h)
    c_h_flags = BitField(default=0, help_text=Glossary.c_h_flags)
    c_h_rchi2 = FloatField(null=True, help_text=Glossary.c_h_rchi2)
    
    co_h = FloatField(null=True, help_text=Glossary.co_h)
    e_co_h = FloatField(null=True, help_text=Glossary.e_co_h)
    co_h_flags = BitField(default=0, help_text=Glossary.co_h_flags)
    co_h_rchi2 = FloatField(null=True, help_text=Glossary.co_h_rchi2)
    
    cr_h = FloatField(null=True, help_text=Glossary.cr_h)
    e_cr_h = FloatField(null=True, help_text=Glossary.e_cr_h)
    cr_h_flags = BitField(default=0, help_text=Glossary.cr_h_flags)
    cr_h_rchi2 = FloatField(null=True, help_text=Glossary.cr_h_rchi2)
    
    cu_h = FloatField(null=True, help_text=Glossary.cu_h)
    e_cu_h = FloatField(null=True, help_text=Glossary.e_cu_h)
    cu_h_flags = BitField(default=0, help_text=Glossary.cu_h_flags)
    cu_h_rchi2 = FloatField(null=True, help_text=Glossary.cu_h_rchi2)
    
    fe_h = FloatField(null=True, help_text=Glossary.fe_h)
    e_fe_h = FloatField(null=True, help_text=Glossary.e_fe_h)
    fe_h_flags = BitField(default=0, help_text=Glossary.fe_h_flags)
    fe_h_rchi2 = FloatField(null=True, help_text=Glossary.fe_h_rchi2)

    k_h = FloatField(null=True, help_text=Glossary.k_h)
    e_k_h = FloatField(null=True, help_text=Glossary.e_k_h)
    k_h_flags = BitField(default=0, help_text=Glossary.k_h_flags)
    k_h_rchi2 = FloatField(null=True, help_text=Glossary.k_h_rchi2)

    mg_h = FloatField(null=True, help_text=Glossary.mg_h)
    e_mg_h = FloatField(null=True, help_text=Glossary.e_mg_h)
    mg_h_flags = BitField(default=0, help_text=Glossary.mg_h_flags)
    mg_h_rchi2 = FloatField(null=True, help_text=Glossary.mg_h_rchi2)

    mn_h = FloatField(null=True, help_text=Glossary.mn_h)
    e_mn_h = FloatField(null=True, help_text=Glossary.e_mn_h)
    mn_h_flags = BitField(default=0, help_text=Glossary.mn_h_flags)
    mn_h_rchi2 = FloatField(null=True, help_text=Glossary.mn_h_rchi2)

    na_h = FloatField(null=True, help_text=Glossary.na_h)
    e_na_h = FloatField(null=True, help_text=Glossary.e_na_h)
    na_h_flags = BitField(default=0, help_text=Glossary.na_h_flags)
    na_h_rchi2 = FloatField(null=True, help_text=Glossary.na_h_rchi2)

    nd_h = FloatField(null=True, help_text=Glossary.nd_h)
    e_nd_h = FloatField(null=True, help_text=Glossary.e_nd_h)
    nd_h_flags = BitField(default=0, help_text=Glossary.nd_h_flags)
    nd_h_rchi2 = FloatField(null=True, help_text=Glossary.nd_h_rchi2)

    ni_h = FloatField(null=True, help_text=Glossary.ni_h)
    e_ni_h = FloatField(null=True, help_text=Glossary.e_ni_h)
    ni_h_flags = BitField(default=0, help_text=Glossary.ni_h_flags)
    ni_h_rchi2 = FloatField(null=True, help_text=Glossary.ni_h_rchi2)

    n_h = FloatField(null=True, help_text=Glossary.n_h)
    e_n_h = FloatField(null=True, help_text=Glossary.e_n_h)
    n_h_flags = BitField(default=0, help_text=Glossary.n_h_flags)
    n_h_rchi2 = FloatField(null=True, help_text=Glossary.n_h_rchi2)

    o_h = FloatField(null=True, help_text=Glossary.o_h)
    e_o_h = FloatField(null=True, help_text=Glossary.e_o_h)
    o_h_flags = BitField(default=0, help_text=Glossary.o_h_flags)
    o_h_rchi2 = FloatField(null=True, help_text=Glossary.o_h_rchi2)

    p_h = FloatField(null=True, help_text=Glossary.p_h)
    e_p_h = FloatField(null=True, help_text=Glossary.e_p_h)
    p_h_flags = BitField(default=0, help_text=Glossary.p_h_flags)
    p_h_rchi2 = FloatField(null=True, help_text=Glossary.p_h_rchi2)

    si_h = FloatField(null=True, help_text=Glossary.si_h)
    e_si_h = FloatField(null=True, help_text=Glossary.e_si_h)
    si_h_flags = BitField(default=0, help_text=Glossary.si_h_flags)
    si_h_rchi2 = FloatField(null=True, help_text=Glossary.si_h_rchi2)

    s_h = FloatField(null=True, help_text=Glossary.s_h)
    e_s_h = FloatField(null=True, help_text=Glossary.e_s_h)
    s_h_flags = BitField(default=0, help_text=Glossary.s_h_flags)
    s_h_rchi2 = FloatField(null=True, help_text=Glossary.s_h_rchi2)

    ti_h = FloatField(null=True, help_text=Glossary.ti_h)
    e_ti_h = FloatField(null=True, help_text=Glossary.e_ti_h)
    ti_h_flags = BitField(default=0, help_text=Glossary.ti_h_flags)
    ti_h_rchi2 = FloatField(null=True, help_text=Glossary.ti_h_rchi2)

    ti_2_h = FloatField(null=True, help_text=Glossary.ti_2_h)
    e_ti_2_h = FloatField(null=True, help_text=Glossary.e_ti_2_h)
    ti_2_h_flags = BitField(default=0, help_text=Glossary.ti_2_h_flags)
    ti_2_h_rchi2 = FloatField(null=True, help_text=Glossary.ti_2_h_rchi2)

    v_h = FloatField(null=True, help_text=Glossary.v_h)
    e_v_h = FloatField(null=True, help_text=Glossary.e_v_h)
    v_h_flags = BitField(default=0, help_text=Glossary.v_h_flags)
    v_h_rchi2 = FloatField(null=True, help_text=Glossary.v_h_rchi2)

    # TODO: include the initial parameters from the stellar parameter run? Or the COARSE run?

    #> FERRE Settings
    short_grid_name = TextField(default="", help_text="Short name describing the FERRE grid used")
    continuum_order = IntegerField(default=-1, help_text="Continuum order used in FERRE")
    continuum_reject = FloatField(null=True, help_text="Tolerance for FERRE to reject continuum points")
    #continuum_flag = IntegerField(default=0, null=True)
    #continuum_observations_flag = IntegerField(default=0, null=True)
    interpolation_order = IntegerField(default=-1, help_text="Interpolation order used by FERRE")
    initial_flags = BitField(default=0, help_text=Glossary.initial_flags)
    flag_initial_guess_from_apogeenet = initial_flags.flag(2**0, help_text="Initial guess from APOGEENet")
    flag_initial_guess_from_doppler = initial_flags.flag(2**1, help_text="Initial guess from Doppler (SDSS-V)")
    flag_initial_guess_from_doppler_sdss4 = initial_flags.flag(2**1, help_text="Initial guess from Doppler (SDSS-IV)")
    flag_initial_guess_from_gaia_xp_andrae23 = initial_flags.flag(2**3, help_text="Initial guess from Andrae et al. (2023)")
    flag_initial_guess_from_user = initial_flags.flag(2**2, help_text="Initial guess specified by user")

    #> Summary Statistics
    snr = FloatField(null=True, help_text=Glossary.snr)
    rchi2 = FloatField(null=True, help_text=Glossary.rchi2)
    ferre_log_snr_sq = FloatField(null=True, help_text="FERRE-reported log10(snr**2)")
    ferre_flags = BitField(default=0, help_text="Flags indicating FERRE issues")
    
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

    #> Task Primary Keys
    stellar_parameters_task_pk = ForeignKeyField(FerreStellarParameters, null=True, lazy_load=False, help_text="Task primary key for stellar parameters")
    al_h_task_pk = ForeignKeyField(FerreChemicalAbundances, null=True, lazy_load=False, help_text="Task primary key for [Al/H]")
    c_12_13_task_pk = ForeignKeyField(FerreChemicalAbundances, null=True, lazy_load=False, help_text="Task primary key for C12/C13")
    ca_h_task_pk = ForeignKeyField(FerreChemicalAbundances, null=True, lazy_load=False, help_text="Task primary key for [Ca/H]")
    ce_h_task_pk = ForeignKeyField(FerreChemicalAbundances, null=True, lazy_load=False, help_text="Task primary key for [Ce/H]")
    c_1_h_task_pk = ForeignKeyField(FerreChemicalAbundances, null=True, lazy_load=False, help_text="Task primary key for [C 1/H]")
    c_h_task_pk = ForeignKeyField(FerreChemicalAbundances, null=True, lazy_load=False, help_text="Task primary key for [C/H]")
    co_h_task_pk = ForeignKeyField(FerreChemicalAbundances, null=True, lazy_load=False, help_text="Task primary key for [Co/H]")
    cr_h_task_pk = ForeignKeyField(FerreChemicalAbundances, null=True, lazy_load=False, help_text="Task primary key for [Cr/H]")
    cu_h_task_pk = ForeignKeyField(FerreChemicalAbundances, null=True, lazy_load=False, help_text="Task primary key for [Cu/H]")
    fe_h_task_pk = ForeignKeyField(FerreChemicalAbundances, null=True, lazy_load=False, help_text="Task primary key for [Fe/H]")
    k_h_task_pk = ForeignKeyField(FerreChemicalAbundances, null=True, lazy_load=False, help_text="Task primary key for [K/H]")
    mg_h_task_pk = ForeignKeyField(FerreChemicalAbundances, null=True, lazy_load=False, help_text="Task primary key for [Mg/H]")
    mn_h_task_pk = ForeignKeyField(FerreChemicalAbundances, null=True, lazy_load=False, help_text="Task primary key for [Mn/H]")
    na_h_task_pk = ForeignKeyField(FerreChemicalAbundances, null=True, lazy_load=False, help_text="Task primary key for [Na/H]")
    nd_h_task_pk = ForeignKeyField(FerreChemicalAbundances, null=True, lazy_load=False, help_text="Task primary key for [Nd/H]")
    ni_h_task_pk = ForeignKeyField(FerreChemicalAbundances, null=True, lazy_load=False, help_text="Task primary key for [Ni/H]")
    n_h_task_pk = ForeignKeyField(FerreChemicalAbundances, null=True, lazy_load=False, help_text="Task primary key for [N/H]")
    o_h_task_pk = ForeignKeyField(FerreChemicalAbundances, null=True, lazy_load=False, help_text="Task primary key for [O/H]")
    p_h_task_pk = ForeignKeyField(FerreChemicalAbundances, null=True, lazy_load=False, help_text="Task primary key for [P/H]")
    si_h_task_pk = ForeignKeyField(FerreChemicalAbundances, null=True, lazy_load=False, help_text="Task primary key for [Si/H]")
    s_h_task_pk = ForeignKeyField(FerreChemicalAbundances, null=True, lazy_load=False, help_text="Task primary key for [S/H]")
    ti_h_task_pk = ForeignKeyField(FerreChemicalAbundances, null=True, lazy_load=False, help_text="Task primary key for [Ti/H]")
    ti_2_h_task_pk = ForeignKeyField(FerreChemicalAbundances, null=True, lazy_load=False, help_text="Task primary key for [Ti 2/H]")
    v_h_task_pk = ForeignKeyField(FerreChemicalAbundances, null=True, lazy_load=False, help_text="Task primary key for [V/H]")


    #> Raw (Uncalibrated) Quantities
    calibrated = BooleanField(default=False, help_text=Glossary.calibrated)
    raw_teff = FloatField(null=True, help_text=Glossary.raw_teff)
    raw_e_teff = FloatField(null=True, help_text=Glossary.raw_e_teff)
    raw_e_sys_teff = FloatField(null=True, help_text=Glossary.raw_e_sys_teff)
    raw_logg = FloatField(null=True, help_text=Glossary.raw_logg)
    raw_e_logg = FloatField(null=True, help_text=Glossary.raw_e_logg)
    raw_e_sys_logg = FloatField(null=True, help_text=Glossary.raw_e_sys_logg)
    raw_v_micro = FloatField(null=True, help_text=Glossary.raw_v_micro)
    raw_e_v_micro = FloatField(null=True, help_text=Glossary.raw_e_v_micro)
    raw_e_sys_v_micro = FloatField(null=True, help_text=Glossary.raw_e_sys_v_micro)
    raw_v_sini = FloatField(null=True, help_text=Glossary.raw_v_sini)
    raw_e_v_sini = FloatField(null=True, help_text=Glossary.raw_e_v_sini)
    raw_e_sys_v_sini = FloatField(null=True, help_text=Glossary.raw_e_sys_v_sini)
    raw_m_h_atm = FloatField(null=True, help_text=Glossary.raw_m_h_atm)
    raw_e_m_h_atm = FloatField(null=True, help_text=Glossary.raw_e_m_h_atm)
    raw_e_sys_m_h_atm = FloatField(null=True, help_text=Glossary.raw_e_sys_m_h_atm)
    raw_alpha_m_atm = FloatField(null=True, help_text=Glossary.raw_alpha_m_atm)
    raw_e_alpha_m_atm = FloatField(null=True, help_text=Glossary.raw_e_alpha_m_atm)
    raw_e_sys_alpha_m_atm = FloatField(null=True, help_text=Glossary.raw_e_sys_alpha_m_atm)
    raw_c_m_atm = FloatField(null=True, help_text=Glossary.raw_c_m_atm)
    raw_e_c_m_atm = FloatField(null=True, help_text=Glossary.raw_e_c_m_atm)
    raw_e_sys_c_m_atm = FloatField(null=True, help_text=Glossary.raw_e_sys_c_m_atm)
    raw_n_m_atm = FloatField(null=True, help_text=Glossary.raw_n_m_atm)
    raw_e_n_m_atm = FloatField(null=True, help_text=Glossary.raw_e_n_m_atm)
    raw_e_sys_n_m_atm = FloatField(null=True, help_text=Glossary.raw_e_sys_n_m_atm)
    raw_al_h = FloatField(null=True, help_text=Glossary.raw_al_h)
    raw_e_al_h = FloatField(null=True, help_text=Glossary.raw_e_al_h)
    raw_e_sys_al_h = FloatField(null=True, help_text=Glossary.raw_e_sys_al_h)
    raw_c_12_13 = FloatField(null=True, help_text=Glossary.raw_c_12_13)
    raw_e_c_12_13 = FloatField(null=True, help_text=Glossary.raw_e_c_12_13)
    raw_e_sys_c_12_13 = FloatField(null=True, help_text=Glossary.raw_e_sys_c_12_13)
    raw_ca_h = FloatField(null=True, help_text=Glossary.raw_ca_h)
    raw_e_ca_h = FloatField(null=True, help_text=Glossary.raw_e_ca_h)
    raw_e_sys_ca_h = FloatField(null=True, help_text=Glossary.raw_e_sys_ca_h)
    raw_ce_h = FloatField(null=True, help_text=Glossary.raw_ce_h)
    raw_e_ce_h = FloatField(null=True, help_text=Glossary.raw_e_ce_h)
    raw_e_sys_ce_h = FloatField(null=True, help_text=Glossary.raw_e_sys_ce_h)
    raw_c_1_h = FloatField(null=True, help_text=Glossary.raw_c_1_h)
    raw_e_c_1_h = FloatField(null=True, help_text=Glossary.raw_e_c_1_h)
    raw_e_sys_c_1_h = FloatField(null=True, help_text=Glossary.raw_e_sys_c_1_h)
    raw_c_h = FloatField(null=True, help_text=Glossary.raw_c_h)
    raw_e_c_h = FloatField(null=True, help_text=Glossary.raw_e_c_h)
    raw_e_sys_c_h = FloatField(null=True, help_text=Glossary.raw_e_sys_c_h)
    raw_co_h = FloatField(null=True, help_text=Glossary.raw_co_h)
    raw_e_co_h = FloatField(null=True, help_text=Glossary.raw_e_co_h)
    raw_e_sys_co_h = FloatField(null=True, help_text=Glossary.raw_e_sys_co_h)
    raw_cr_h = FloatField(null=True, help_text=Glossary.raw_cr_h)
    raw_e_cr_h = FloatField(null=True, help_text=Glossary.raw_e_cr_h)
    raw_e_sys_cr_h = FloatField(null=True, help_text=Glossary.raw_e_sys_cr_h)
    raw_cu_h = FloatField(null=True, help_text=Glossary.raw_cu_h)
    raw_e_cu_h = FloatField(null=True, help_text=Glossary.raw_e_cu_h) 
    raw_e_sys_cu_h = FloatField(null=True, help_text=Glossary.raw_e_sys_cu_h)
    raw_fe_h = FloatField(null=True, help_text=Glossary.raw_fe_h)
    raw_e_fe_h = FloatField(null=True, help_text=Glossary.raw_e_fe_h)
    raw_e_sys_fe_h = FloatField(null=True, help_text=Glossary.raw_e_sys_fe_h)
    raw_k_h = FloatField(null=True, help_text=Glossary.raw_k_h)
    raw_e_k_h = FloatField(null=True, help_text=Glossary.raw_e_k_h)
    raw_e_sys_k_h = FloatField(null=True, help_text=Glossary.raw_e_sys_k_h)
    raw_mg_h = FloatField(null=True, help_text=Glossary.raw_mg_h)
    raw_e_mg_h = FloatField(null=True, help_text=Glossary.raw_e_mg_h)
    raw_e_sys_mg_h = FloatField(null=True, help_text=Glossary.raw_e_sys_mg_h)
    raw_mn_h = FloatField(null=True, help_text=Glossary.raw_mn_h)
    raw_e_mn_h = FloatField(null=True, help_text=Glossary.raw_e_mn_h)
    raw_e_sys_mn_h = FloatField(null=True, help_text=Glossary.raw_e_sys_mn_h)
    raw_na_h = FloatField(null=True, help_text=Glossary.raw_na_h)
    raw_e_na_h = FloatField(null=True, help_text=Glossary.raw_e_na_h)
    raw_e_sys_na_h = FloatField(null=True, help_text=Glossary.raw_e_sys_na_h)
    raw_nd_h = FloatField(null=True, help_text=Glossary.raw_nd_h)
    raw_e_nd_h = FloatField(null=True, help_text=Glossary.raw_e_nd_h)
    raw_e_sys_nd_h = FloatField(null=True, help_text=Glossary.raw_e_sys_nd_h)
    raw_ni_h = FloatField(null=True, help_text=Glossary.raw_ni_h)
    raw_e_ni_h = FloatField(null=True, help_text=Glossary.raw_e_ni_h)
    raw_e_sys_ni_h = FloatField(null=True, help_text=Glossary.raw_e_sys_ni_h)
    raw_n_h = FloatField(null=True, help_text=Glossary.raw_n_h)
    raw_e_n_h = FloatField(null=True, help_text=Glossary.raw_e_n_h)
    raw_e_sys_n_h = FloatField(null=True, help_text=Glossary.raw_e_sys_n_h)
    raw_o_h = FloatField(null=True, help_text=Glossary.raw_o_h)
    raw_e_o_h = FloatField(null=True, help_text=Glossary.raw_e_o_h)
    raw_e_sys_o_h = FloatField(null=True, help_text=Glossary.raw_e_sys_o_h)
    raw_p_h = FloatField(null=True, help_text=Glossary.raw_p_h)
    raw_e_p_h = FloatField(null=True, help_text=Glossary.raw_e_p_h)
    raw_e_sys_p_h = FloatField(null=True, help_text=Glossary.raw_e_sys_p_h)
    raw_si_h = FloatField(null=True, help_text=Glossary.raw_si_h)
    raw_e_si_h = FloatField(null=True, help_text=Glossary.raw_e_si_h)
    raw_e_sys_si_h = FloatField(null=True, help_text=Glossary.raw_e_sys_si_h)
    raw_s_h = FloatField(null=True, help_text=Glossary.raw_s_h)
    raw_e_s_h = FloatField(null=True, help_text=Glossary.raw_e_s_h)
    raw_e_sys_s_h = FloatField(null=True, help_text=Glossary.raw_e_sys_s_h)
    raw_ti_h = FloatField(null=True, help_text=Glossary.raw_ti_h)
    raw_e_ti_h = FloatField(null=True, help_text=Glossary.raw_e_ti_h)
    raw_e_sys_ti_h = FloatField(null=True, help_text=Glossary.raw_e_sys_ti_h)
    raw_ti_2_h = FloatField(null=True, help_text=Glossary.raw_ti_2_h)
    raw_e_ti_2_h = FloatField(null=True, help_text=Glossary.raw_e_ti_2_h)
    raw_e_sys_ti_2_h = FloatField(null=True, help_text=Glossary.raw_e_sys_ti_2_h)
    raw_v_h = FloatField(null=True, help_text=Glossary.raw_v_h)
    raw_e_v_h = FloatField(null=True, help_text=Glossary.raw_e_v_h)
    raw_e_sys_v_h = FloatField(null=True, help_text=Glossary.raw_e_sys_v_h)
    