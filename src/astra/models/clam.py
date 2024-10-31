import numpy as np
import datetime
from astra.fields import (
    AutoField, FloatField, TextField, BitField, ForeignKeyField, BitField, DateTimeField
)
from astra.models.source import Source
from astra.models.spectrum import Spectrum
from astra.models.pipeline import PipelineOutputModel

class Clam(PipelineOutputModel):

    """A result from the Clam pipeline."""

    #> Stellar Labels
    teff = FloatField(null=True)
    e_teff = FloatField(null=True)
    logg = FloatField(null=True)
    e_logg = FloatField(null=True)
    m_h = FloatField(null=True)
    e_m_h = FloatField(null=True)
    n_m = FloatField(null=True)
    e_n_m = FloatField(null=True)
    c_m = FloatField(null=True)
    e_c_m = FloatField(null=True)
    v_micro = FloatField(null=True)
    e_v_micro = FloatField(null=True)
    v_sini = FloatField(null=True)
    e_v_sini = FloatField(null=True)

    #> Initial position
    initial_teff = FloatField(null=True)
    initial_logg = FloatField(null=True)
    initial_m_h = FloatField(null=True)
    initial_n_m = FloatField(null=True)
    initial_c_m = FloatField(null=True)
    initial_v_micro = FloatField(null=True)
    initial_v_sini = FloatField(null=True)

    #> Summary Statistics
    rchi2 = FloatField(null=True)
    result_flags = BitField(default=0, help_text="Flags describing the results")
    flag_spectrum_io_error = result_flags.flag(2**0, help_text="Spectrum I/O error")
    flag_runtime_error = result_flags.flag(2**1, help_text="Runtime error")
