
import os
import numpy as np

from peewee import (
    AutoField,
    FloatField,
    BooleanField,
    DateTimeField,
    BigIntegerField,
    IntegerField,
    TextField,
    Model,
    ForeignKeyField,
    Node,
    Field,
    BigBitField,
)
from astropy.io import fits
from playhouse.hybrid import hybrid_property
from playhouse.sqlite_ext import SqliteExtDatabase, JSONField

from astra import __version__
from astra.models.base import BaseModel, Source, UniqueSpectrum
from astra.models.glossary import Glossary
from astra.models.fields import PixelArray, BitField
from astra.utils import expand_path

class PipelineOutputMixin:

    def dump_to_fits(self, filename, overwrite=False):
        """
        Dump the pipeline results to a FITS file.

        :param filename:
            The filename to dump to.
        
        :param overwrite:
            Overwrite the file if it already exists?
        """

        raise NotImplementedError



class ApogeeNet(BaseModel, PipelineOutputMixin):

    #: Source information.
    source = ForeignKeyField(Source, index=True)

    #: Metadata
    spectrum_id = ForeignKeyField(UniqueSpectrum, index=True, lazy_load=False)
    v_astra = TextField(default=__version__)
    t_elapsed = FloatField(null=True)

    #: Stellar parameters
    teff = FloatField()
    e_teff = FloatField(help_text="Error on effective temperature [K]")
    logg = FloatField(help_text="Surface gravity [log10(cm/s^2)]")
    e_logg = FloatField(help_text="Error on surface gravity [log10(cm/s^2)]")
    fe_h = FloatField(help_text="Metallicity [dex]")
    e_fe_h = FloatField(help_text="Error on metallicity [dex]")
    teff_sample_median = FloatField()
    logg_sample_median = FloatField()
    fe_h_sample_median = FloatField()
    bitmask_flag = BitField()

    #: Flag definitions
    flag_teff_unreliable = bitmask_flag.flag(1, help_text="Effective temperature is unreliable")
    flag_logg_unreliable = bitmask_flag.flag(2, help_text="Surface gravity is unreliable")
    flag_fe_h_unreliable = bitmask_flag.flag(4, help_text="Metallicity is unreliable")
    
    flag_e_teff_unreliable = bitmask_flag.flag(8)
    flag_e_logg_unreliable = bitmask_flag.flag(16)
    flag_e_fe_h_unreliable = bitmask_flag.flag(32)

    flag_e_teff_large = bitmask_flag.flag(64)
    flag_e_logg_large = bitmask_flag.flag(128)
    flag_e_fe_h_large = bitmask_flag.flag(256)
    flag_missing_photometry = bitmask_flag.flag(512)
    flag_result_unreliable = bitmask_flag.flag(1024)
    
    @hybrid_property
    def warn_flag(self):
        return (
            self.flag_e_teff_large |
            self.flag_e_logg_large |
            self.flag_e_fe_h_large |
            self.flag_missing_photometry
        )

    @warn_flag.expression
    def warn_flag(self):
        return (
            self.flag_e_teff_large |
            self.flag_e_logg_large |
            self.flag_e_fe_h_large |
            self.flag_missing_photometry
        )

    @hybrid_property
    def bad_flag(self):
        return (
            self.flag_result_unreliable |
            self.flag_teff_unreliable |
            self.flag_logg_unreliable |
            self.flag_fe_h_unreliable |
            self.flag_e_teff_unreliable |
            self.flag_e_logg_unreliable |
            self.flag_e_fe_h_unreliable
        )

    @bad_flag.expression
    def bad_flag(self):
        return (
            self.flag_result_unreliable |
            self.flag_teff_unreliable |
            self.flag_logg_unreliable |
            self.flag_fe_h_unreliable |
            self.flag_e_teff_unreliable |
            self.flag_e_logg_unreliable |
            self.flag_e_fe_h_unreliable
        )
    

    def apply_flags(self, meta=None):
        """
        Set flags for the pipeline outputs, given the metadata used.

        :param meta:
            A seven-length array containing the following metadata:
                - parallax
                - g_mag
                - bp_mag
                - rp_mag
                - j_mag
                - h_mag
                - k_mag
        """
    
        if self.fe_h > 0.5 or self.fe_h < -2 or np.log10(self.teff) > 3.82:
            self.flag_fe_h_unreliable = True

        if self.logg < -1.5 or self.logg > 6:
            self.flag_logg_unreliable = True
        
        if np.log10(self.teff) < 3.1 or np.log10(self.teff) > 4.7:
            self.flag_teff_unreliable = True

        if self.fe_h_sample_median > 0.5 or self.fe_h_sample_median < -2 or np.log10(self.teff_sample_median) > 3.82:
            self.flag_e_fe_h_unreliable = True
        
        if self.logg_sample_median < -1.5 or self.logg_sample_median > 6:
            self.flag_e_logg_unreliable = True
        
        if np.log10(self.teff) < 3.1 or np.log10(self.teff) > 4.7:
            self.flag_e_teff_unreliable = True

        if self.e_logg > 0.3:
            self.flag_e_logg_large = True
        if self.e_fe_h > 0.5:
            self.flag_e_fe_h_large = True
        if np.log10(self.e_teff) > 2.7:
            self.flag_e_teff_large = True

        if meta is None or not np.all(np.isfinite(meta)):
            self.flag_missing_photometry = True

        if meta is not None:
            plx, g_mag, bp_mag, rp_mag, j_mag, h_mag, k_mag = meta   
            is_bad = ((rp_mag - k_mag) > 2.3) & ((h_mag - 5 * np.log10(1000 / plx) + 5) > 6)
            if is_bad:
                self.flag_result_unreliable = True
            
        return None



class FerreCoarse(BaseModel, PipelineOutputMixin):
    
    id = AutoField(primary_key=True)
    source = ForeignKeyField(Source, index=True)
    spectrum_id = ForeignKeyField(UniqueSpectrum, index=True, lazy_load=False)

    pwd = TextField(default="")
    short_grid_name = TextField(default="")
    header_path = TextField(default="")
    
    initial_teff = FloatField(default=np.nan)
    initial_logg = FloatField(default=np.nan)
    initial_m_h = FloatField(default=np.nan)
    initial_log10_v_sini = FloatField(default=np.nan)
    initial_log10_v_micro = FloatField(default=np.nan)
    initial_alpha_m = FloatField(default=np.nan)
    initial_c_m = FloatField(default=np.nan)
    initial_n_m = FloatField(default=np.nan)

    initial_flags = BitField(default=0)
    flag_initial_guess_from_apogeenet = initial_flags.flag(1)
    flag_initial_guess_from_doppler = initial_flags.flag(2)
    flag_initial_guess_from_user = initial_flags.flag(4)
    flag_initial_guess_from_gaia_xp_andrae23 = initial_flags.flag(8)

    frozen_flags = BitField(default=0)
    flag_teff_frozen = frozen_flags.flag(1)
    flag_logg_frozen = frozen_flags.flag(2)
    flag_m_h_frozen = frozen_flags.flag(4)
    flag_log10_v_sini_frozen = frozen_flags.flag(8)
    flag_log10_v_micro_frozen = frozen_flags.flag(16)
    flag_alpha_m_frozen = frozen_flags.flag(32)
    flag_c_m_frozen = frozen_flags.flag(64)
    flag_n_m_frozen = frozen_flags.flag(128)

    continuum_order = IntegerField(default=-1)
    continuum_reject = FloatField(default=np.nan)
    interpolation_order = IntegerField(default=-1)
    weight_path = TextField(default="")

    # Now the outputs
    teff = FloatField(default=np.nan)
    logg = FloatField(default=np.nan)
    m_h = FloatField(default=np.nan)
    log10_v_sini = FloatField(default=np.nan)
    log10_v_micro = FloatField(default=np.nan)
    alpha_m = FloatField(default=np.nan)
    c_m = FloatField(default=np.nan)
    n_m = FloatField(default=np.nan)

    e_teff = FloatField(default=np.nan)
    e_logg = FloatField(default=np.nan)
    e_m_h = FloatField(default=np.nan)
    e_log10_v_sini = FloatField(default=np.nan)
    e_log10_v_micro = FloatField(default=np.nan)
    e_alpha_m = FloatField(default=np.nan)
    e_c_m = FloatField(default=np.nan)
    e_n_m = FloatField(default=np.nan)

    teff_flags = BitField(default=0)
    logg_flags = BitField(default=0)
    m_h_flags = BitField(default=0)
    log10_v_sini_flags = BitField(default=0)
    log10_v_micro_flags = BitField(default=0)
    alpha_m_flags = BitField(default=0)
    c_m_flags = BitField(default=0)
    n_m_flags = BitField(default=0)

    # TODO: flag grid edge bad/warn
    flag_teff_ferre_fail = teff_flags.flag(1)
    flag_teff_grid_edge_bad = teff_flags.flag(2)
    flag_teff_grid_edge_warn = teff_flags.flag(4)


    chisq = FloatField(default=np.nan)
    ferre_log_snr_sq = FloatField(default=np.nan)
    ferre_log_chisq = FloatField(default=np.nan)
    ferre_frac_phot_data_points = FloatField(default=0)
    ferre_log_penalized_chisq = FloatField(default=np.nan)
    
    f_access = IntegerField(default=-1)
    f_format = IntegerField(default=-1)
    n_threads = IntegerField(default=-1)

    ferre_name = TextField(default="")
    ferre_index = IntegerField(default=-1)
    ferre_n_obj = IntegerField(default=-1)
    ferre_time_load_grid = FloatField(default=np.nan)
    ferre_time_elapsed = FloatField(default=np.nan)

    ferre_flags = BitField(default=0)
    flag_ferre_fail = ferre_flags.flag(1)
    flag_missing_model_flux = ferre_flags.flag(2)
    flag_potential_ferre_timeout = ferre_flags.flag(4)
    flag_no_suitable_initial_guess = ferre_flags.flag(8)

    '''
    class Meta:
        indexes = (
            (
                (
                    "short_grid_name", 
                    "spectrum_id", 
                    "initial_teff",
                    "initial_logg",
                    "initial_m_h",
                ), 
                True
            ),
        )
    '''




    # TODO: A helper function until I have implemented the PixelAccessor classes for FERRE outputs
    def _get_pixel_array_from_file_with_name(self, basename, P=7514):
        path = expand_path(f"{self.pwd}/{basename}")
        names = np.loadtxt(path, usecols=(0, ), dtype=str)
        index = np.where(names == self.ferre_name)[0][0]
        skiprows = index if index > 0 else 0
        assert np.atleast_1d(np.loadtxt(path, usecols=(0, ), dtype=str, skiprows=skiprows, max_rows=1))[0] == self.ferre_name
        return np.loadtxt(
            path, 
            usecols=range(1, 1 + P), 
            dtype=float, 
            skiprows=skiprows,
            max_rows=1
        )
        
    def _get_input_pixel_array(self, basename):
        path = expand_path(f"{self.pwd}/{basename}")
        return np.loadtxt(path, dtype=float, skiprows=self.ferre_index, max_rows=1)



def make_plot(item, mode="reflect"):
    from scipy.ndimage.filters import median_filter
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(15, 4), sharex=True)

    # the model flux is the best-fitting model flux that has been rectified by FERRE.
    ferre_flux = item._get_input_pixel_array("flux.input")
    model_flux = item._get_pixel_array_from_file_with_name("model_flux.output")

    # the rectified flux is that rectification, applied to the observed flux
    rectified_flux = item._get_pixel_array_from_file_with_name("rectified_flux.output")

    continuum_applied_by_ferre = ferre_flux / rectified_flux

    median_filtered = median_filter(rectified_flux / model_flux, [151], mode=mode, cval=0.0)

    axes[0].plot(rectified_flux, c='k')
    axes[0].plot(model_flux, c='tab:red')
    axes[0].plot(median_filtered, c="tab:blue")

    axes[1].plot(rectified_flux * continuum_applied_by_ferre, c='k')
    axes[1].plot(model_flux * continuum_applied_by_ferre, c="tab:red")
    axes[1].plot(ferre_flux / median_filtered, c="tab:blue")
    #ax.plot(median_filtered, c="tab:orange")
    #ax.plot(continuum_applied_by_ferre, c="tab:orange")
    fig.savefig("tmp.png", dpi=600)
    return fig, axes[0]




class FerreStellarParameters(BaseModel, PipelineOutputMixin):
    
    #: Identifiers
    id = AutoField(primary_key=True)
    source = ForeignKeyField(Source, index=True)
    spectrum_id = ForeignKeyField(UniqueSpectrum, index=True, lazy_load=False)
    coarse = ForeignKeyField(
        FerreCoarse,
        index=True,
        help_text=Glossary.coarse_id
    )

    pwd = TextField(default="")
    short_grid_name = TextField(default="")
    header_path = TextField(default="")

    initial_teff = FloatField(default=np.nan)
    initial_logg = FloatField(default=np.nan)
    initial_m_h = FloatField(default=np.nan)
    initial_log10_v_sini = FloatField(default=np.nan)
    initial_log10_v_micro = FloatField(default=np.nan)
    initial_alpha_m = FloatField(default=np.nan)
    initial_c_m = FloatField(default=np.nan)
    initial_n_m = FloatField(default=np.nan)
    initial_flags = BitField(default=0)
    flag_initial_guess_from_apogeenet = initial_flags.flag(1)
    flag_initial_guess_from_doppler = initial_flags.flag(2)
    flag_initial_guess_from_user = initial_flags.flag(4)
    flag_initial_guess_from_gaia_xp_andrae23 = initial_flags.flag(8)

    frozen_flags = BitField(default=0)
    flag_teff_frozen = frozen_flags.flag(1)
    flag_logg_frozen = frozen_flags.flag(2)
    flag_m_h_frozen = frozen_flags.flag(4)
    flag_log10_v_sini_frozen = frozen_flags.flag(8)
    flag_log10_v_micro_frozen = frozen_flags.flag(16)
    flag_alpha_m_frozen = frozen_flags.flag(32)
    flag_c_m_frozen = frozen_flags.flag(64)
    flag_n_m_frozen = frozen_flags.flag(128)

    continuum_order = IntegerField(default=-1)
    continuum_reject = FloatField(default=np.nan)
    interpolation_order = IntegerField(default=-1)
    weight_path = TextField(default="")

    # Now the outputs
    teff = FloatField(default=np.nan)
    logg = FloatField(default=np.nan)
    m_h = FloatField(default=np.nan)
    log10_v_sini = FloatField(default=np.nan)
    log10_v_micro = FloatField(default=np.nan)
    alpha_m = FloatField(default=np.nan)
    c_m = FloatField(default=np.nan)
    n_m = FloatField(default=np.nan)

    e_teff = FloatField(default=np.nan)
    e_logg = FloatField(default=np.nan)
    e_m_h = FloatField(default=np.nan)
    e_log10_v_sini = FloatField(default=np.nan)
    e_log10_v_micro = FloatField(default=np.nan)
    e_alpha_m = FloatField(default=np.nan)
    e_c_m = FloatField(default=np.nan)
    e_n_m = FloatField(default=np.nan)

    teff_flags = BitField(default=0)
    logg_flags = BitField(default=0)
    m_h_flags = BitField(default=0)
    log10_v_sini_flags = BitField(default=0)
    log10_v_micro_flags = BitField(default=0)
    alpha_m_flags = BitField(default=0)
    c_m_flags = BitField(default=0)
    n_m_flags = BitField(default=0)

    # TODO: flag grid edge bad/warn
    flag_teff_ferre_fail = teff_flags.flag(1)
    flag_teff_grid_edge_bad = teff_flags.flag(2)
    flag_teff_grid_edge_warn = teff_flags.flag(4)


    chisq = FloatField(default=np.nan)
    ferre_log_snr_sq = FloatField(default=np.nan)
    ferre_log_chisq = FloatField(default=np.nan)
    ferre_frac_phot_data_points = FloatField(default=0)
    ferre_log_penalized_chisq = FloatField(default=np.nan)
    
    f_access = IntegerField(default=-1)
    f_format = IntegerField(default=-1)
    n_threads = IntegerField(default=-1)

    ferre_n_obj = IntegerField(default=-1)
    ferre_time_load_grid = FloatField(default=np.nan)
    ferre_time_elapsed = FloatField(default=np.nan)

    ferre_name = TextField(default="")
    ferre_flags = BitField(default=0)
    flag_ferre_fail = ferre_flags.flag(1)
    flag_missing_model_flux = ferre_flags.flag(2)
    flag_potential_ferre_timeout = ferre_flags.flag(4)
    flag_no_suitable_initial_guess = ferre_flags.flag(8)

    '''
    class Meta:
        indexes = (
            (
                (
                    "short_grid_name", 
                    "spectrum_id", 
                    "initial_teff",
                    "initial_logg",
                    "initial_m_h",
                ), 
                True
            ),
        )
    '''

    # TODO: A helper function until I have implemented the PixelAccessor classes for FERRE outputs
    def _get_pixel_array_from_file_with_name(self, basename, P=7514):
        path = expand_path(f"{self.pwd}/{basename}")
        names = np.loadtxt(path, usecols=(0, ), dtype=str)
        index = np.where(names == self.ferre_name)[0][0]
        skiprows = index - 1 if index > 0 else 0
        return np.loadtxt(
            path, 
            usecols=range(1, 1 + P), 
            dtype=float, 
            skiprows=skiprows,
            max_rows=1
        )
        
    





class Aspcap(BaseModel):

    source = ForeignKeyField(Source, index=True)
    spectrum_id = ForeignKeyField(UniqueSpectrum, index=True, lazy_load=False)

    teff = FloatField()
    logg = FloatField()
    m_h = FloatField()
    v_sini = FloatField()
    v_micro = FloatField()
    alpha_m = FloatField()
    c_m = FloatField()
    n_m = FloatField()

    model_flux = PixelArray()
    
    #al_h = FloatField()
    #e_al_h = FloatField()
    #model_flux_al_h = PixelArray()


    


'''

class FerreStellarParameters(BaseModel):

    source = ForeignKeyField(Source, index=True)
    spectrum = ForeignKeyField(Spectrum, index=True, lazy_load=False)

    pwd = TextField()
    short_grid_name = TextField()
    

    initial_guess_source? = ForeignKeyField(FerreCoarse)
    initial_teff
    initial_logg
    ...

    frozen_teff
    frozen_logg
    frozen_metals
    ...

    teff
    logg
    m_h

    bitmask

    same ferre-related inputs


class FerreAbundances(BaseModel):

    source = ForeignKeyField(Source, index=True)
    spectrum = ForeignKeyField(Spectrum, index=True, lazy_load=False)

    pwd = TextField()
    short_grid_name = TextField()

    element


'''
