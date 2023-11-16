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
    classification = TextField(null=True, help_text="Classification")
    p_cv = FloatField(null=True, help_text="Cataclysmic variable probability")
    p_da = FloatField(null=True, help_text="DA-type white dwarf probability")
    p_dab = FloatField(null=True, help_text="DAB-type white dwarf probability")
    p_dabz = FloatField(null=True, help_text="DABZ-type white dwarf probability")
    p_dah = FloatField(null=True, help_text="DA (H)-type white dwarf probability")
    p_dahe = FloatField(null=True, help_text="DA (He)-type white dwarf probability")
    p_dao = FloatField(null=True, help_text="DAO-type white dwarf probability")
    p_daz = FloatField(null=True, help_text="DAZ-type white dwarf probability")
    p_da_ms = FloatField(null=True, help_text="DA-MS binary probability")
    p_db = FloatField(null=True, help_text="DB-type white dwarf probability")
    p_dba = FloatField(null=True, help_text="DBA-type white dwarf probability")
    p_dbaz = FloatField(null=True, help_text="DBAZ-type white dwarf probability")
    p_dbh = FloatField(null=True, help_text="DB (H)-type white dwarf probability")
    p_dbz = FloatField(null=True, help_text="DBZ-type white dwarf probability")
    p_db_ms = FloatField(null=True, help_text="DB-MS binary probability")
    p_dc = FloatField(null=True, help_text="DC-type white dwarf probability")
    p_dc_ms = FloatField(null=True, help_text="DC-MS binary probability")
    p_do = FloatField(null=True, help_text="DO-type white dwarf probability")
    p_dq = FloatField(null=True, help_text="DQ-type white dwarf probability")
    p_dqz = FloatField(null=True, help_text="DQZ-type white dwarf probability")
    p_dqpec = FloatField(null=True, help_text="DQ Peculiar-type white dwarf probability")
    p_dz = FloatField(null=True, help_text="DZ-type white dwarf probability")
    p_dza = FloatField(null=True, help_text="DZA-type white dwarf probability")
    p_dzb = FloatField(null=True, help_text="DZB-type white dwarf probability")
    p_dzba = FloatField(null=True, help_text="DZBA-type white dwarf probability")
    p_mwd = FloatField(null=True, help_text="Main sequence star probability")
    p_hotdq = FloatField(null=True, help_text="Hot DQ-type white dwarf probability") 

    #> Stellar Parameters
    teff = FloatField(null=True, help_text=Glossary.teff)
    e_teff = FloatField(null=True, help_text=Glossary.e_teff)
    logg = FloatField(null=True, help_text=Glossary.logg)
    e_logg = FloatField(null=True, help_text=Glossary.e_logg)
    v_rel = FloatField(null=True, help_text="Relative velocity used in stellar parameter fit [km/s]")
    
    #> Formal uncertainties
    raw_e_teff = FloatField(null=True, help_text=Glossary.raw_e_teff)
    raw_e_logg = FloatField(null=True, help_text=Glossary.raw_e_logg)
    
    #> Metadata
    result_flags = BitField(default=0, help_text="Result flags")

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

    # TODO: have a consistent name for this, like intermediate_output_path (e.g., see SLAM model)
    @property
    def path(self):
        return f"$MWM_ASTRA/{__version__}/pipelines/snow_white/{self.spectrum_pk}.fits"




def apply_noise_model():
    
    (
        SnowWhite
        .update(e_teff=1.5 * SnowWhite.raw_e_teff + 100)
        .execute()
    )
    (
        SnowWhite
        .update(e_logg=2 * SnowWhite.raw_e_logg + 5e-2)
        .execute()
    )