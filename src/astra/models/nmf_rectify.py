from astra.fields import (ArrayField, FloatField, IntegerField, BitField)
from astra.models.pipeline import PipelineOutputModel

class NMFRectify(PipelineOutputModel):
    
    #> Continuum Fitting
    log10_W = ArrayField(FloatField, null=True, help_text="log10(W) NMF coefficients to compute spectra")
    continuum_theta = ArrayField(FloatField, null=True, help_text="Continuum coefficients")
    L = FloatField(help_text="Sinusoidal length scale for continuum")
    deg = IntegerField(help_text="Sinusoidal degree for continuum")
    rchi2 = FloatField(null=True, help_text=Glossary.rchi2)
    joint_rchi2 = FloatField(null=True, help_text="Joint reduced chi^2 from simultaneous fit")
    nmf_flags = BitField(default=0, help_text="NMF Continuum method flags") #TODO: rename as nmf_flags
    flag_initialised_from_small_w = nmf_flags.flag(2**0)

    flag_could_not_read_spectrum = nmf_flags.flag(2**3)
    flag_runtime_exception = nmf_flags.flag(2**4)
