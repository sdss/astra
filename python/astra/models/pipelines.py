
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
    
    source = ForeignKeyField(Source, index=True)
    spectrum_id = ForeignKeyField(UniqueSpectrum, index=True, lazy_load=False)

    pwd = TextField()
    grid_name = TextField()
    
    initial_teff = FloatField()
    initial_logg = FloatField()
    initial_m_h = FloatField()
    initial_v_sini = FloatField()
    initial_v_micro = FloatField()
    initial_alpha_m = FloatField()
    initial_c_m = FloatField()
    initial_n_m = FloatField()

    initial_flags = BitField()
    flag_initial_guess_from_apogeenet = initial_flags.flag(1)
    flag_initial_guess_from_doppler = initial_flags.flag(2)
    flag_initial_guess_from_user = initial_flags.flag(4)
    flag_initial_guess_from_gaia_xp_andrae23 = initial_flags.flag(8)

    frozen_flags = BitField()
    flag_teff_frozen = frozen_flags.flag(1)
    flag_logg_frozen = frozen_flags.flag(2)
    flag_m_h_frozen = frozen_flags.flag(4)
    flag_v_sini_frozen = frozen_flags.flag(8)
    flag_v_micro_frozen = frozen_flags.flag(16)
    flag_alpha_m_frozen = frozen_flags.flag(32)
    flag_c_m_frozen = frozen_flags.flag(64)
    flag_n_m_frozen = frozen_flags.flag(128)


    interpolation_order = IntegerField()
    weight_path = TextField()

    #continuum_flag
    #continuum_order
    #continuum_segment
    #continuum_reject
    #continuum_observations_flag

    # Now the outputs
    teff = FloatField()
    logg = FloatField()
    m_h = FloatField()
    v_sini = FloatField()
    v_micro = FloatField()
    alpha_m = FloatField()
    c_m = FloatField()
    n_m = FloatField()

    e_teff = FloatField()
    e_logg = FloatField()
    e_m_h = FloatField()
    e_v_sini = FloatField()
    e_v_micro = FloatField()
    e_alpha_m = FloatField()
    e_c_m = FloatField()
    e_n_m = FloatField()

    teff_flags = BitField()
    logg_flags = BitField()
    m_h_flags = BitField()
    v_sini_flags = BitField()
    v_micro_flags = BitField()
    alpha_m_flags = BitField()
    c_m_flags = BitField()
    n_m_flags = BitField()
    
    chisq = FloatField()
    ferre_chisq = FloatField()
    ferre_penalized_chisq = FloatField()
    
    f_access = IntegerField()
    f_format = IntegerField()
    n_threads = IntegerField()

    ferre_n_obj = IntegerField()
    ferre_load_grid_time = FloatField()
    ferre_time_elapsed = FloatField()
    ferre_flags = BitField()
    flag_ferre_timeout = ferre_flags.flag(1)

    class Meta:
        indexes = (
            (
                (
                    "grid_name", 
                    "spectrum_id", 
                    "initial_teff",
                    "initial_logg",
                    "initial_m_h",
                ), 
                True
            ),
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
    grid_name = TextField()
    

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
    grid_name = TextField()

    element


'''
