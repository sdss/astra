import numpy as np
from inspect import currentframe
from peewee import (
    AutoField,
    FloatField,
    TextField,
    ForeignKeyField,
    BitField,
)
from playhouse.hybrid import hybrid_property

from astra import __version__
from astra.models.base import BaseModel, Source, UniqueSpectrum
from astra.models.glossary import Glossary
from astra.models.fields import BitField
from astra.models.pipeline import PipelineOutputMixin



class ApogeeNet(BaseModel, PipelineOutputMixin):

    """A result from the APOGEENet pipeline."""

    sdss_id = ForeignKeyField(Source, index=True, help_text=Glossary.sdss_id)
    spectrum_id = ForeignKeyField(UniqueSpectrum, index=True, lazy_load=False, help_text=Glossary.spectrum_id)

    #> Astra Metadata
    task_id = AutoField()
    v_astra = TextField(default=__version__, help_text=Glossary.v_astra)
    t_elapsed = FloatField(null=True, help_text=Glossary.t_elapsed)
    tag = TextField(default="", index=True, help_text=Glossary.tag)

    #> Stellar Parameters
    teff = FloatField(help_text=Glossary.teff)
    e_teff = FloatField(help_text=Glossary.e_teff)
    logg = FloatField(help_text=Glossary.logg)
    e_logg = FloatField(help_text=Glossary.e_logg)
    fe_h = FloatField(help_text=Glossary.fe_h)
    e_fe_h = FloatField(help_text=Glossary.e_fe_h)
    teff_sample_median = FloatField(help_text=Glossary.teff_sample_median)
    logg_sample_median = FloatField(help_text=Glossary.logg_sample_median)
    fe_h_sample_median = FloatField(help_text=Glossary.fe_h_sample_median)
    result_flags = BitField(default=0, help_text="Bit flags for the stellar parameters")

    #> Flag definitions
    flag_teff_unreliable = result_flags.flag(2**0, help_text="Effective temperature is unreliable")
    flag_logg_unreliable = result_flags.flag(2**1, help_text="Surface gravity is unreliable")
    flag_fe_h_unreliable = result_flags.flag(2**2, help_text="Metallicity is unreliable")
    
    flag_e_teff_unreliable = result_flags.flag(2**3, help_text="Error on effective temperature is unreliable")
    flag_e_logg_unreliable = result_flags.flag(2**4, help_text="Error on surface gravity is unreliable")
    flag_e_fe_h_unreliable = result_flags.flag(2**5, help_text="Error on metallicity is unreliable")

    flag_e_teff_large = result_flags.flag(2**6, help_text="Error on effective temperature is large")
    flag_e_logg_large = result_flags.flag(2**7, help_text="Error on surface gravity is large")
    flag_e_fe_h_large = result_flags.flag(2**8, help_text="Error on metallicity is large")
    flag_missing_photometry = result_flags.flag(2**9, help_text="Missing photometry")
    flag_result_unreliable = result_flags.flag(2**10, help_text="Stellar parameters are knowingly unreliable")

    @hybrid_property
    def flag_warn(self):
        return (
            self.flag_e_teff_large |
            self.flag_e_logg_large |
            self.flag_e_fe_h_large |
            self.flag_missing_photometry
        )

    @flag_warn.expression
    def flag_warn(self):
        return (
            self.flag_e_teff_large |
            self.flag_e_logg_large |
            self.flag_e_fe_h_large |
            self.flag_missing_photometry
        )

    @hybrid_property
    def flag_bad(self):
        return (
            self.flag_result_unreliable |
            self.flag_teff_unreliable |
            self.flag_logg_unreliable |
            self.flag_fe_h_unreliable |
            self.flag_e_teff_unreliable |
            self.flag_e_logg_unreliable |
            self.flag_e_fe_h_unreliable
        )

    @flag_bad.expression
    def flag_bad(self):
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