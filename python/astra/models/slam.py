from peewee import (
    AutoField,
    FloatField,
    TextField,
    ForeignKeyField,
    BooleanField,
    IntegerField
)
from playhouse.hybrid import hybrid_property

from astra import __version__
from astra.models.base import BaseModel
from astra.models.fields import BitField
from astra.models.source import Source
from astra.models.spectrum import Spectrum
from astra.models.pipeline import PipelineOutputMixin


class Slam(BaseModel, PipelineOutputMixin):

    """A result from the 'Stellar Labels Machine'."""

    sdss_id = ForeignKeyField(Source, index=True)
    spectrum_id = ForeignKeyField(Spectrum, index=True, lazy_load=False)
    
    #> Astra Metadata
    task_id = AutoField()
    v_astra = TextField(default=__version__)
    t_elapsed = FloatField(null=True)
    tag = TextField(default="", index=True)
    
    #> Stellar Labels
    teff = FloatField(null=True)
    e_teff = FloatField(null=True)
    logg = FloatField(null=True)
    e_logg = FloatField(null=True)
    fe_h = FloatField(null=True)
    e_fe_h = FloatField(null=True)
    
    #> Correlation Coefficients
    rho_teff_logg = FloatField(null=True)
    rho_teff_fe_h = FloatField(null=True)
    rho_logg_fe_h = FloatField(null=True)

    #> Initial Labels
    initial_teff = FloatField(null=True)
    initial_logg = FloatField(null=True)
    initial_fe_h = FloatField(null=True)

    #> Metadata
    success = BooleanField()
    status = IntegerField()
    optimality = BooleanField()
    result_flags = BitField(default=0)

    #> Summary Statistics
    snr = FloatField()
    chi_sq = FloatField()
    reduced_chi_sq = FloatField()
    

    @hybrid_property
    def flag_warn(self):
        return (self.status > 0) & (self.status != 2)

    @flag_warn.expression
    def flag_warn(self):
        return (self.status > 0) & (self.status != 2)

    @hybrid_property
    def flag_bad(self):
        return (self.status < 0)

    @flag_bad.expression
    def flag_bad(self):
        return (self.status < 0)