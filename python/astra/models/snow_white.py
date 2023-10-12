from peewee import (
    AutoField,
    FloatField,
    TextField,
    ForeignKeyField,
    IntegerField,
    DateTimeField
)
import datetime
from astra import __version__
from astra.models.base import BaseModel
from astra.models.fields import BitField
from astra.models.source import Source
from astra.models.spectrum import Spectrum
from astra.models.pipeline import PipelineOutputMixin
from astra.models.fields import PixelArray, BitField, LogLambdaArrayAccessor

from astra.glossary import Glossary

from playhouse.postgres_ext import ArrayField


class SnowWhite(BaseModel, PipelineOutputMixin):

    """A result from the white-dwarf pipeline, affectionally known as Snow White."""

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

    #> Classification Probabilities
    classification = TextField(null=True)
    p_cv = FloatField(null=True)
    p_da = FloatField(null=True)
    p_dab = FloatField(null=True)
    p_dabz = FloatField(null=True)
    p_dah = FloatField(null=True)
    p_dahe = FloatField(null=True)
    p_dao = FloatField(null=True)
    p_daz = FloatField(null=True)
    p_da_ms = FloatField(null=True)
    p_db = FloatField(null=True)
    p_dba = FloatField(null=True)
    p_dbaz = FloatField(null=True)
    p_dbh = FloatField(null=True)
    p_dbz = FloatField(null=True)
    p_db_ms = FloatField(null=True)
    p_dc = FloatField(null=True)
    p_dc_ms = FloatField(null=True)
    p_do = FloatField(null=True)
    p_dq = FloatField(null=True)
    p_dqz = FloatField(null=True)
    p_dqpec = FloatField(null=True)
    p_dz = FloatField(null=True)
    p_dza = FloatField(null=True)
    p_dzb = FloatField(null=True)
    p_dzba = FloatField(null=True)
    p_mwd = FloatField(null=True)
    p_hotdq = FloatField(null=True)    

    #> Stellar Parameters
    teff = FloatField(null=True, help_text=Glossary.teff)
    e_teff = FloatField(null=True, help_text=Glossary.e_teff)
    logg = FloatField(null=True, help_text=Glossary.logg)
    e_logg = FloatField(null=True, help_text=Glossary.e_logg)
    v_rel = FloatField(null=True, help_text="Relative velocity used in stellar parameter fit [km/s]")
    
    #> Metadata
    result_flags = BitField(default=0)

    #> Spectral Data
    wavelength = PixelArray(
        accessor_class=LogLambdaArrayAccessor,
        accessor_kwargs=dict(
            crval=3.5523,
            cdelt=1e-4,
            naxis=4648,
        ),
        help_text=Glossary.wavelength
    )    
    model_flux = PixelArray(
        ext=1,
        help_text=Glossary.model_flux,
    )

    @property
    def path(self):
        return f"$MWM_ASTRA/{__version__}/pipelines/snow_white/{self.spectrum_pk}.fits"



