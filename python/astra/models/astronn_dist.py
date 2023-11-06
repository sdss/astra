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

from astra import __version__
from astra.models.base import BaseModel
from astra.models.fields import BitField
from astra.models.source import Source
from astra.models.spectrum import Spectrum
from astra.models.pipeline import PipelineOutputMixin

from astra.glossary import Glossary

class AstroNNdist(BaseModel, PipelineOutputMixin):

    """A result from the AstroNN distance pipeline."""

    source_pk = ForeignKeyField(Source, null=True, index=True, lazy_load=False)
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

    #> Stellar Labels
    k_mag = FloatField(null=True, help_text="2MASS K-band apparent magnitude")
    ebv = FloatField(null=True, help_text="E(B-V) reddening")
    a_k_mag = FloatField(null=True, help_text="K-band extinction")
    fake_k_mag = FloatField(null=True, help_text="Predicted (fake) K-band absolute magnitude")
    fake_k_mag_err = FloatField(null=True, help_text="Prediected (fake) K-band absolute magnitude error")
    dist = FloatField(null=True, help_text="Heliocentric distance [pc]")
    dist_err = FloatField(null=True, help_text="Heliocentric distance error [pc]")

    #> Summary Statistics
    result_flags = BitField(default=0, help_text=Glossary.result_flags)

    #> Flag definitions
    flag_fakemag_unreliable = result_flags.flag(2**0, help_text="Predicted (fake) K-band absolute photometry is unreliable (fakemag_err / fakemag >= 0.2)")
    flag_missing_photometry = result_flags.flag(2**1, help_text="Missing Ks-band apparent magnitude")
    flag_missing_extinction = result_flags.flag(2**2, help_text="Missing extinction")
    flag_no_result = result_flags.flag(2**11, help_text="Exception raised when loading spectra")


    def apply_flags(self, meta=None, missing_photometry=False, missing_extinction=False):
        """
        Set flags for the pipeline outputs, given the metadata used.

        :param meta:
            A three-length array containing the following metadata:
                - k_mag
                - ebv
                - a_k_mag
        """

        if self.fake_k_mag_err >= self.fake_k_mag * 0.2:
            self.flag_fakemag_unreliable = True

        if meta is None or missing_photometry:
            self.flag_missing_photometry = True

        if meta is None or missing_extinction:
            self.flag_missing_extinction = True
            
        return None
