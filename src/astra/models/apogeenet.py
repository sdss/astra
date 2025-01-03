from playhouse.hybrid import hybrid_property
from astra.fields import (FloatField, BitField)
from astra.models.pipeline import PipelineOutputModel
    
class ApogeeNet(PipelineOutputModel): 

    """A result from the APOGEENet (version 3) pipeline."""
    
    #> Stellar Parameters
    teff = FloatField(null=True)
    e_teff = FloatField(null=True)
    logg = FloatField(null=True)
    e_logg = FloatField(null=True)
    fe_h = FloatField(null=True)
    e_fe_h = FloatField(null=True)
    result_flags = BitField(default=0)

    # Putting the flag definitions here until we decide on making them consistent across pipelines.
    flag_runtime_exception = result_flags.flag(2**0, help_text="Exception raised during runtime")
    flag_unreliable_teff = result_flags.flag(2**1, help_text="`teff` is outside the range of 1700 K and 100,000 K")
    flag_unreliable_logg = result_flags.flag(2**2, help_text="`logg` is outside the range of -1 and 6")
    flag_unreliable_fe_h = result_flags.flag(2**3, help_text="`teff` < 3200 K or `logg` > 5 or `fe_h` is outside the range of -4 and 2")

    #> Formal (raw) uncertainties
    raw_e_teff = FloatField(null=True)
    raw_e_logg = FloatField(null=True)
    raw_e_fe_h = FloatField(null=True)

    @hybrid_property
    def flag_warn(self):
        return self.flag_unreliable_fe_h
    
    @flag_warn.expression
    def flag_warn(self):
        return self.flag_unreliable_fe_h
    
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
    def from_spectrum(cls, spectrum, teff=None, logg=None, fe_h=None, **kwargs):
        """
        Create a new instance of this model from a Spectrum instance.

        :param spectrum:
            The spectrum instance.
        """
        kwds = kwargs.copy()
        for key in ("teff", "logg", "fe_h"):
            
            kwds.setdefault(f"raw_e_{key}", kwargs.get(f"e_{key}"))
            
        kwds.update(
            teff=teff,
            logg=logg,
            fe_h=fe_h,
            flag_unreliable_teff=(teff is not None and not (1700 < teff < 100_000)),
            flag_unreliable_logg=(logg is not None and not (-1 < logg < 10)),
            flag_unreliable_fe_h=(
                    (fe_h is not None and not (-4 < fe_h < 2))
                or  (teff is not None and teff < 3200)
                or  (logg is not None and logg > 5)
            )
        )
        return super().from_spectrum(spectrum, **kwds)




    

def apply_noise_model():

    (
        ApogeeNet
        .update(
            e_teff=1.25 * ApogeeNet.raw_e_teff + 10,
            e_logg=1.25 * ApogeeNet.raw_e_logg + 1e-2,
            e_fe_h=ApogeeNet.raw_e_fe_h + 1e-2       
        )
        .where(ApogeeNet.v_astra == __version__)
        .execute()
    )
