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

# TODO: Move this to happen at runtime
def apply_result_flags():
    (
        ApogeeNet
        .update(
            raw_e_teff=ApogeeNet.e_teff,
            raw_e_logg=ApogeeNet.e_logg,
            raw_e_fe_h=ApogeeNet.e_fe_h
        )
        .where(
            ApogeeNet.raw_e_teff.is_null()
        &   (ApogeeNet.v_astra == __version__)
        )
        .execute()
    )    
    (
        ApogeeNet
        .update(result_flags=ApogeeNet.flag_unreliable_teff.set())
        .where(
            (ApogeeNet.teff < 1700) | (ApogeeNet.teff > 100_000)
        )
        .where(ApogeeNet.v_astra == __version__)
        .execute()
    )    
    (
        ApogeeNet
        .update(result_flags=ApogeeNet.flag_unreliable_logg.set())
        .where(
            (ApogeeNet.logg < -1) | (ApogeeNet.logg > 10)
        )
        .where(ApogeeNet.v_astra == __version__)
        .execute()
    )    
    (
        ApogeeNet
        .update(result_flags=ApogeeNet.flag_unreliable_fe_h.set())
        .where(
            (ApogeeNet.teff < 3200)
        |   (ApogeeNet.logg > 5)
        |   (ApogeeNet.fe_h < -4)
        |   (ApogeeNet.fe_h > 2)
        )
        .where(ApogeeNet.v_astra == __version__)
        .execute()
    )
    

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
