import datetime
from peewee import (
    AutoField,
    FloatField,
    TextField,
    ForeignKeyField,
    BooleanField,
    IntegerField,
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


class Slam(BaseModel, PipelineOutputMixin):

    """A result from the 'Stellar Labels Machine'."""

    sdss_id = ForeignKeyField(Source, null=True, index=True, lazy_load=False, help_text=Glossary.sdss_id)
    spectrum_id = ForeignKeyField(Spectrum, index=True, lazy_load=False, help_text=Glossary.spectrum_id)
    
    #> Astra Metadata
    task_id = AutoField(help_text=Glossary.task_id)
    v_astra = TextField(default=__version__, help_text=Glossary.v_astra)
    created = DateTimeField(default=datetime.datetime.now, help_text=Glossary.created)
    t_elapsed = FloatField(null=True, help_text=Glossary.t_elapsed)
    t_overhead = FloatField(null=True, help_text=Glossary.t_overhead)
    tag = TextField(default="", index=True, help_text=Glossary.tag)
    
    #> Stellar Labels
    teff = FloatField(null=True, help_text=Glossary.teff)
    e_teff = FloatField(null=True, help_text=Glossary.e_teff)
    logg = FloatField(null=True, help_text=Glossary.logg)
    e_logg = FloatField(null=True, help_text=Glossary.e_logg)
    fe_h = FloatField(null=True, help_text=Glossary.fe_h)
    e_fe_h = FloatField(null=True, help_text=Glossary.e_fe_h)
    
    #> Correlation Coefficients
    rho_teff_logg = FloatField(null=True, help_text=Glossary.rho_teff_logg)
    rho_teff_fe_h = FloatField(null=True, help_text=Glossary.rho_teff_fe_h)
    rho_logg_fe_h = FloatField(null=True, help_text=Glossary.rho_logg_fe_h)

    #> Initial Labels
    initial_teff = FloatField(null=True, help_text=Glossary.initial_teff)
    initial_logg = FloatField(null=True, help_text=Glossary.initial_logg)
    initial_fe_h = FloatField(null=True, help_text=Glossary.initial_fe_h)

    #> Metadata
    success = BooleanField(help_text="Optimizer returned successful value")
    status = IntegerField(help_text="Optimization status")
    optimality = BooleanField(help_text="Optimality condition")
    result_flags = BitField(default=0, help_text=Glossary.result_flags)

    #> Summary Statistics
    snr = FloatField(help_text=Glossary.snr)
    chi2 = FloatField(help_text=Glossary.chi2)
    rchi2 = FloatField(help_text=Glossary.rchi2)
    

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