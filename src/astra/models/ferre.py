
import datetime
import numpy as np
from astra import __version__
from astra.models.base import BaseModel
from astra.fields import (
    AutoField,
    BitField,
    FloatField,
    TextField,
    ForeignKeyField,
    IntegerField,
    BitField,
    DateTimeField,
    BooleanField,
)
from astra.models.source import Source
from astra.models.spectrum import Spectrum
from functools import cached_property

from astra.pipelines.ferre.utils import (get_apogee_pixel_mask, parse_ferre_spectrum_name)

APOGEE_FERRE_MASK = get_apogee_pixel_mask()


class FerreOutputMixin(BaseModel):
        
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
        unmasked_array = fill_value * np.ones(APOGEE_FERRE_MASK.shape)
        unmasked_array[APOGEE_FERRE_MASK] = array
        return unmasked_array

        
    def _get_input_pixel_array(self, basename):
        return np.loadtxt(
            fname=f"{self.pwd}/{basename}",
            skiprows=int(self.ferre_input_index), 
            max_rows=1,
        )


    def _get_output_pixel_array(self, basename, P=7514):
        
        #assert self.ferre_input_index >= 0

        kwds = dict(
            fname=f"{self.pwd}/{basename}",
            skiprows=int(self.ferre_output_index), 
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
    modified = DateTimeField(default=datetime.datetime.now)
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
    flag_initial_guess_from_doppler_sdss4 = initial_flags.flag(2**2, help_text="Initial guess from Doppler (SDSS-IV)")
    flag_initial_guess_from_gaia_xp_andrae_2023 = initial_flags.flag(2**3, help_text="Initial guess from Andrae et al. (2023)")
    flag_initial_guess_from_gaia_xp_zhang_2023 = initial_flags.flag(2**4, "Initial guess from Zhang, Green & Rix (2023)")
    flag_initial_guess_from_user = initial_flags.flag(2**5, help_text="Initial guess specified by user")
    flag_initial_guess_at_grid_center = initial_flags.flag(2**6, help_text="Initial guess from grid center")

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

    flag_teff_frozen = frozen_flags.flag(2**0, help_text="Effective temperature is frozen")
    flag_logg_frozen = frozen_flags.flag(2**1, help_text="Surface gravity is frozen")
    flag_m_h_frozen = frozen_flags.flag(2**2, help_text="[M/H] is frozen")
    flag_log10_v_sini_frozen = frozen_flags.flag(2**3, help_text="Rotational broadening is frozen")
    flag_log10_v_micro_frozen = frozen_flags.flag(2**4, help_text="Microturbulence is frozen")
    flag_alpha_m_frozen = frozen_flags.flag(2**5, help_text="[alpha/M] is frozen")
    flag_c_m_frozen = frozen_flags.flag(2**6, help_text="[C/M] is frozen")
    flag_n_m_frozen = frozen_flags.flag(2**7, help_text="[N/M] is frozen")

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

    flag_ferre_fail = ferre_flags.flag(2**0, help_text="FERRE failed")
    flag_missing_model_flux = ferre_flags.flag(2**1, help_text="Missing model fluxes from FERRE")
    flag_potential_ferre_timeout = ferre_flags.flag(2**2, help_text="Potentially impacted by FERRE timeout")
    flag_no_suitable_initial_guess = ferre_flags.flag(2**3, help_text="FERRE not executed because there's no suitable initial guess")
    flag_spectrum_io_error = ferre_flags.flag(2**4, help_text="Error accessing spectrum pixel data")
    flag_teff_grid_edge_warn = ferre_flags.flag(2**5)
    flag_teff_grid_edge_bad = ferre_flags.flag(2**6)
    flag_logg_grid_edge_warn = ferre_flags.flag(2**7)
    flag_logg_grid_edge_bad = ferre_flags.flag(2**8)
    flag_v_micro_grid_edge_warn = ferre_flags.flag(2**9)
    flag_v_micro_grid_edge_bad = ferre_flags.flag(2**10)
    flag_v_sini_grid_edge_warn = ferre_flags.flag(2**11)
    flag_v_sini_grid_edge_bad = ferre_flags.flag(2**12)
    flag_m_h_atm_grid_edge_warn = ferre_flags.flag(2**13)
    flag_m_h_atm_grid_edge_bad = ferre_flags.flag(2**14)
    flag_alpha_m_grid_edge_warn = ferre_flags.flag(2**15)
    flag_alpha_m_grid_edge_bad = ferre_flags.flag(2**16)
    flag_c_m_atm_grid_edge_warn = ferre_flags.flag(2**17)
    flag_c_m_atm_grid_edge_bad = ferre_flags.flag(2**18)
    flag_n_m_atm_grid_edge_warn = ferre_flags.flag(2**19)
    flag_n_m_atm_grid_edge_bad = ferre_flags.flag(2**20)    



    '''
    class Meta:
        # To prevent post-processing tasks ingesting the same results many times
        indexes = (
            (
                (
                    "pwd",
                    "ferre_name",
                ),
                True,
            ),
        )  
    '''          

class FerreStellarParameters(BaseModel, FerreOutputMixin):

    source_pk = ForeignKeyField(Source, index=True, lazy_load=False)
    spectrum_pk = ForeignKeyField(Spectrum, index=True, lazy_load=False)
    upstream = ForeignKeyField(FerreCoarse, column_name="upstream_pk", index=True)

    #> Astra Metadata
    task_pk = AutoField()
    v_astra = TextField(default=__version__)
    created = DateTimeField(default=datetime.datetime.now)
    modified = DateTimeField(default=datetime.datetime.now)
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

    flag_teff_frozen = frozen_flags.flag(2**0, help_text="Effective temperature is frozen")
    flag_logg_frozen = frozen_flags.flag(2**1, help_text="Surface gravity is frozen")
    flag_m_h_frozen = frozen_flags.flag(2**2, help_text="[M/H] is frozen")
    flag_log10_v_sini_frozen = frozen_flags.flag(2**3, help_text="Rotational broadening is frozen")
    flag_log10_v_micro_frozen = frozen_flags.flag(2**4, help_text="Microturbulence is frozen")
    flag_alpha_m_frozen = frozen_flags.flag(2**5, help_text="[alpha/M] is frozen")
    flag_c_m_frozen = frozen_flags.flag(2**6, help_text="[C/M] is frozen")
    flag_n_m_frozen = frozen_flags.flag(2**7, help_text="[N/M] is frozen")

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

    flag_ferre_fail = ferre_flags.flag(2**0, help_text="FERRE failed")
    flag_missing_model_flux = ferre_flags.flag(2**1, help_text="Missing model fluxes from FERRE")
    flag_potential_ferre_timeout = ferre_flags.flag(2**2, help_text="Potentially impacted by FERRE timeout")
    flag_no_suitable_initial_guess = ferre_flags.flag(2**3, help_text="FERRE not executed because there's no suitable initial guess")
    flag_spectrum_io_error = ferre_flags.flag(2**4, help_text="Error accessing spectrum pixel data")
    flag_teff_grid_edge_warn = ferre_flags.flag(2**5)
    flag_teff_grid_edge_bad = ferre_flags.flag(2**6)
    flag_logg_grid_edge_warn = ferre_flags.flag(2**7)
    flag_logg_grid_edge_bad = ferre_flags.flag(2**8)
    flag_v_micro_grid_edge_warn = ferre_flags.flag(2**9)
    flag_v_micro_grid_edge_bad = ferre_flags.flag(2**10)
    flag_v_sini_grid_edge_warn = ferre_flags.flag(2**11)
    flag_v_sini_grid_edge_bad = ferre_flags.flag(2**12)
    flag_m_h_atm_grid_edge_warn = ferre_flags.flag(2**13)
    flag_m_h_atm_grid_edge_bad = ferre_flags.flag(2**14)
    flag_alpha_m_grid_edge_warn = ferre_flags.flag(2**15)
    flag_alpha_m_grid_edge_bad = ferre_flags.flag(2**16)
    flag_c_m_atm_grid_edge_warn = ferre_flags.flag(2**17)
    flag_c_m_atm_grid_edge_bad = ferre_flags.flag(2**18)
    flag_n_m_atm_grid_edge_warn = ferre_flags.flag(2**19)
    flag_n_m_atm_grid_edge_bad = ferre_flags.flag(2**20)    




    '''
    class Meta:
        # To prevent post-processing tasks ingesting the same results many times
        indexes = (
            (
                (
                    "pwd",
                    "ferre_name",
                ),
                True,
            ),
        )            
    '''


class FerreChemicalAbundances(BaseModel, FerreOutputMixin):

    # We need to ovverride these from the FerreOutputMixin class because the `flux` and `e_flux`
    # arrays are in the parent directory.
    @cached_property
    def ferre_flux(self):
        return self._get_input_pixel_array("../flux.input")
        
    @cached_property
    def ferre_e_flux(self):
        return self._get_input_pixel_array("../e_flux.input")
    
    source_pk = ForeignKeyField(Source, index=True, lazy_load=False)
    spectrum_pk = ForeignKeyField(Spectrum, index=True, lazy_load=False)
    upstream = ForeignKeyField(FerreStellarParameters, column_name="upstream_pk", index=True)

    #> Astra Metadata
    task_pk = AutoField()
    v_astra = TextField(default=__version__)
    created = DateTimeField(default=datetime.datetime.now)
    modified = DateTimeField(default=datetime.datetime.now)
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

    flag_teff_frozen = frozen_flags.flag(2**0, help_text="Effective temperature is frozen")
    flag_logg_frozen = frozen_flags.flag(2**1, help_text="Surface gravity is frozen")
    flag_m_h_frozen = frozen_flags.flag(2**2, help_text="[M/H] is frozen")
    flag_log10_v_sini_frozen = frozen_flags.flag(2**3, help_text="Rotational broadening is frozen")
    flag_log10_v_micro_frozen = frozen_flags.flag(2**4, help_text="Microturbulence is frozen")
    flag_alpha_m_frozen = frozen_flags.flag(2**5, help_text="[alpha/M] is frozen")
    flag_c_m_frozen = frozen_flags.flag(2**6, help_text="[C/M] is frozen")
    flag_n_m_frozen = frozen_flags.flag(2**7, help_text="[N/M] is frozen")

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
    
    flag_ferre_fail = ferre_flags.flag(2**0, help_text="FERRE failed")
    flag_missing_model_flux = ferre_flags.flag(2**1, help_text="Missing model fluxes from FERRE")
    flag_potential_ferre_timeout = ferre_flags.flag(2**2, help_text="Potentially impacted by FERRE timeout")
    flag_no_suitable_initial_guess = ferre_flags.flag(2**3, help_text="FERRE not executed because there's no suitable initial guess")
    flag_spectrum_io_error = ferre_flags.flag(2**4, help_text="Error accessing spectrum pixel data")
    flag_teff_grid_edge_warn = ferre_flags.flag(2**5)
    flag_teff_grid_edge_bad = ferre_flags.flag(2**6)
    flag_logg_grid_edge_warn = ferre_flags.flag(2**7)
    flag_logg_grid_edge_bad = ferre_flags.flag(2**8)
    flag_v_micro_grid_edge_warn = ferre_flags.flag(2**9)
    flag_v_micro_grid_edge_bad = ferre_flags.flag(2**10)
    flag_v_sini_grid_edge_warn = ferre_flags.flag(2**11)
    flag_v_sini_grid_edge_bad = ferre_flags.flag(2**12)
    flag_m_h_atm_grid_edge_warn = ferre_flags.flag(2**13)
    flag_m_h_atm_grid_edge_bad = ferre_flags.flag(2**14)
    flag_alpha_m_grid_edge_warn = ferre_flags.flag(2**15)
    flag_alpha_m_grid_edge_bad = ferre_flags.flag(2**16)
    flag_c_m_atm_grid_edge_warn = ferre_flags.flag(2**17)
    flag_c_m_atm_grid_edge_bad = ferre_flags.flag(2**18)
    flag_n_m_atm_grid_edge_warn = ferre_flags.flag(2**19)
    flag_n_m_atm_grid_edge_bad = ferre_flags.flag(2**20)    


    '''
    class Meta:
        # To prevent post-processing tasks ingesting the same results many times
        indexes = (
            (
                (
                    "pwd",
                    "ferre_name",
                ),
                True,
            ),
        )            
    '''
