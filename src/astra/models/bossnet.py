from playhouse.hybrid import hybrid_property
from astra.fields import (FloatField, BitField)
from astra.models.pipeline import PipelineOutputModel

class BossNet(PipelineOutputModel):

    """A result from the BOSSNet pipeline."""
    
    #> Stellar Parameters
    teff = FloatField(null=True)
    e_teff = FloatField(null=True)
    logg = FloatField(null=True)
    e_logg = FloatField(null=True)
    fe_h = FloatField(null=True)
    e_fe_h = FloatField(null=True)
    result_flags = BitField(default=0)
    flag_runtime_exception = result_flags.flag(2**0, help_text="Exception occurred at runtime")
    flag_unreliable_teff = result_flags.flag(2**1, help_text="`teff` is outside the range of 1700 K and 100,000 K")
    flag_unreliable_logg = result_flags.flag(2**2, help_text="`logg` is outside the range of -1 and 10")
    flag_unreliable_fe_h = result_flags.flag(2**3, help_text="`teff` < 3200 K or `logg` > 5 or `fe_h` is outside the range of -4 and 2")
    flag_suspicious_fe_h = result_flags.flag(2**4, help_text="[Fe/H] below `teff` < 3900 K and with 3 < `logg` < 6 may be suspicious")
    
    bn_v_r = FloatField(null=True)
    e_bn_v_r = FloatField(null=True)
    
    @hybrid_property
    def flag_warn(self):
        return (
            self.flag_unreliable_fe_h
        |   self.flag_suspicious_fe_h
        )
    
    @flag_warn.expression
    def flag_warn(self):
        return (
            self.flag_unreliable_fe_h
        |   self.flag_suspicious_fe_h
        )

    @hybrid_property
    def flag_bad(self):
        return (
            self.flag_unreliable_teff
        |   self.flag_unreliable_logg
        |   self.flag_unreliable_fe_h
        |   self.flag_runtime_exception
    )


    @flag_bad.expression
    def flag_bad(self):
        return (
            self.flag_unreliable_teff
        |   self.flag_unreliable_logg
        |   self.flag_unreliable_fe_h
        |   self.flag_runtime_exception
    )
    
    @classmethod
    def from_spectrum(cls, spectrum, **kwargs):
        """
        Create a new instance of this model from a Spectrum instance.

        :param spectrum:
            The spectrum instance.
        """
        kwds = kwargs.copy()
        teff = kwargs.get("teff", None)
        if teff is not None:
            kwds["flag_unreliable_teff"] = ((teff < 1700) | (teff > 100_000))
        else:
            kwds["flag_runtime_exception"] = True
        
        logg = kwargs.get("logg", None)
        if logg is not None:
            kwds["flag_unreliable_logg"] = ((logg < -1) | (logg > 10))
        
        fe_h = kwargs.get("fe_h", None)
        if fe_h is not None and logg is not None and teff is not None:
            kwds["flag_unreliable_fe_h"] = (
                    (teff < 3200)
                |   (logg > 5)
                |   (fe_h < -4)
                |   (fe_h > 2)
            )

        if teff is not None and logg is not None:
            kwds["flag_suspicious_fe_h"] = (
                (teff < 3900)
            &   (6 > logg > 3)
            )

        return super().from_spectrum(spectrum, **kwds)
