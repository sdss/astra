import datetime
import numpy as np
from astra import __version__
from astra.fields import (AutoField, BitField, FloatField, TextField, ForeignKeyField, DateTimeField)
from astra.models.base import BaseModel
from astra.models.source import Source
from astra.models.spectrum import Spectrum
from astra.models.pipeline import PipelineOutputMixin
from playhouse.hybrid import hybrid_property

class AstroNNdist(BaseModel, PipelineOutputMixin):

    """A result from the AstroNN distance pipeline."""

    source_pk = ForeignKeyField(Source, null=True, index=True, lazy_load=False)
    spectrum_pk = ForeignKeyField(Spectrum, index=True, lazy_load=False)
    
    #> Astra Metadata
    task_pk = AutoField()
    v_astra = TextField(default=__version__)
    created = DateTimeField(default=datetime.datetime.now)
    modified = DateTimeField(default=datetime.datetime.now)
    t_elapsed = FloatField(null=True)
    t_overhead = FloatField(null=True)
    tag = TextField(default="", index=True)

    #> Stellar Labels
    k_mag = FloatField(null=True)
    ebv = FloatField(null=True)
    A_k_mag = FloatField(null=True)
    L_fakemag = FloatField(null=True)
    e_L_fakemag = FloatField(null=True)
    dist = FloatField(null=True)
    e_dist = FloatField(null=True)

    #> Summary Statistics
    result_flags = BitField(default=0)

    #> Flag definitions
    flag_fakemag_unreliable = result_flags.flag(2**0, help_text="Predicted (fake) Ks-band absolute luminosity is unreliable (fakemag_err / fakemag >= 0.2)")
    flag_missing_photometry = result_flags.flag(2**1, help_text="Missing Ks-band apparent magnitude")
    flag_missing_extinction = result_flags.flag(2**2, help_text="Missing extinction")
    flag_no_result = result_flags.flag(2**11, help_text="Exception raised when loading spectra")

    @hybrid_property
    def flag_bad(self):
        return (
            self.flag_fakemag_unreliable
        |   self.flag_missing_photometry
        |   self.flag_no_result
    )
    
    @flag_bad.expression
    def flag_bad(self):
        return (
            self.flag_fakemag_unreliable
        |   self.flag_missing_photometry
        |   self.flag_no_result
    )

    @hybrid_property
    def flag_warn(self):
        return self.flag_missing_extinction
    
    @flag_warn.expression
    def flag_warn(self):
        return self.flag_missing_extinction

    class Meta:
        table_name = "astro_nn_dist"



    def apply_flags(self, meta=None, missing_photometry=False, missing_extinction=False):
        """
        Set flags for the pipeline outputs, given the metadata used.

        :param meta:
            A three-length array containing the following metadata:
                - k_mag
                - ebv
                - a_k_mag
        """

        if self.L_fakemag_err >= self.L_fakemag * 0.2:
            self.flag_fakemag_unreliable = True

        if meta is None or missing_photometry:
            self.flag_missing_photometry = True

        if meta is None or missing_extinction:
            self.flag_missing_extinction = True
            
        return None

print("WARNING: TODO: apply_flags needs properly integrating for astronn_dist")