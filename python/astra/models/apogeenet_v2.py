import numpy as np
import datetime
from peewee import (
    AutoField,
    FloatField,
    TextField,
    ForeignKeyField,
    BitField,
    DateTimeField
)
from playhouse.hybrid import hybrid_property

from astra import __version__
from astra.models.base import BaseModel
from astra.models.fields import BitField
from astra.models.source import Source
from astra.models.spectrum import Spectrum
from astra.models.pipeline import PipelineOutputMixin

from astra.glossary import Glossary

class ApogeeNetV2(BaseModel, PipelineOutputMixin):

    """A result from the APOGEENet (v2) pipeline."""

    source_pk = ForeignKeyField(Source, null=True, index=True, lazy_load=False, help_text=Glossary.source_pk)
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
    
    #> Stellar Parameters
    teff = FloatField(null=True, help_text=Glossary.teff)
    e_teff = FloatField(null=True, help_text=Glossary.e_teff)
    logg = FloatField(null=True, help_text=Glossary.logg)
    e_logg = FloatField(null=True, help_text=Glossary.e_logg)
    fe_h = FloatField(null=True, help_text=Glossary.fe_h)
    e_fe_h = FloatField(null=True, help_text=Glossary.e_fe_h)

    teff_sample_median = FloatField(null=True, help_text="Median effective temperature of many draws [K]")
    logg_sample_median = FloatField(null=True, help_text="Median surface gravity of many draws")
    fe_h_sample_median = FloatField(null=True, help_text="Median metallicity of many draws [dex]")
    result_flags = BitField(default=0, help_text=Glossary.result_flags)

    #> Flag definitions
    flag_teff_unreliable = result_flags.flag(2**0, help_text="Effective temperature is unreliable because it is outside the range 3.1 < log10(TEFF) < 4.7")
    flag_logg_unreliable = result_flags.flag(2**1, help_text="Surface gravity is unreliable because it is outside the range -1.5 < logg < 6")
    flag_fe_h_unreliable = result_flags.flag(2**2, help_text="Metallicity is unreliable because it is outside the range -2 < [Fe/H] < 0.5 or log10(TEFF) > 3.82")
    
    flag_e_teff_unreliable = result_flags.flag(2**3, help_text="Error on effective temperature is unreliable")
    flag_e_logg_unreliable = result_flags.flag(2**4, help_text="Error on surface gravity is unreliable")
    flag_e_fe_h_unreliable = result_flags.flag(2**5, help_text="Error on metallicity is unreliabl")

    flag_e_teff_large = result_flags.flag(2**6, help_text="Error on effective temperature is large")
    flag_e_logg_large = result_flags.flag(2**7, help_text="Error on surface gravity is large")
    flag_e_fe_h_large = result_flags.flag(2**8, help_text="Error on metallicity is large")
    flag_missing_photometry = result_flags.flag(2**9, help_text="Missing photometry")
    flag_result_unreliable = result_flags.flag(2**10, help_text="Stellar parameters are knowingly unreliable")
    flag_no_result = result_flags.flag(2**11, help_text="Exception raised when loading spectra")

    #> Formal error values
    raw_e_teff = FloatField(null=True, help_text=Glossary.raw_e_teff)
    raw_e_logg = FloatField(null=True, help_text=Glossary.raw_e_logg)
    raw_e_fe_h = FloatField(null=True, help_text=Glossary.raw_e_fe_h)

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
            self.flag_e_fe_h_unreliable |
            self.flag_no_result
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
            self.flag_e_fe_h_unreliable |
            self.flag_no_result
        )
    

    def apply_flags(self, meta=None, missing_photometry=False):
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
        if np.log10(self.raw_e_teff) > 2.7:
            self.flag_e_teff_large = True

        if meta is None or not np.all(np.isfinite(meta)) or missing_photometry:
            self.flag_missing_photometry = True

        if meta is not None:
            plx, g_mag, bp_mag, rp_mag, j_mag, h_mag, k_mag = meta   
            is_bad = ((rp_mag - k_mag) > 2.3) & ((h_mag - 5 * np.log10(1000 / plx) + 5) > 6)
            if is_bad:
                self.flag_result_unreliable = True
            
        return None


def apply_noise_model():
    
    (
        ApogeeNetV2
        .update(e_teff=1.25 * ApogeeNetV2.raw_e_teff + 5)
        .execute()            
    )
    (
        ApogeeNetV2
        .update(e_logg=1.50 * ApogeeNetV2.raw_e_logg + 1e-2)
        .execute()
    )
    (
        ApogeeNetV2
        .update(e_fe_h=1.25 * ApogeeNetV2.raw_e_fe_h + 1e-2)
        .execute()
    )