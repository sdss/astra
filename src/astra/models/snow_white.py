import datetime
from astropy.io import fits
from astra import __version__
from astra.utils import expand_path
from astra.models.base import BaseModel
from astra.fields import (
    BitField,
    AutoField,
    ArrayField,
    FloatField,
    TextField,
    ForeignKeyField,
    IntegerField,
    DateTimeField,
    PixelArray, BitField, LogLambdaArrayAccessor,
    BasePixelArrayAccessor
)    
from astra.models.source import Source
from astra.models.spectrum import Spectrum

class IntermediatePixelArrayAccessor(BasePixelArrayAccessor):
    
    def __get__(self, instance, instance_type=None):
        if instance is not None:
            try:
                return instance.__pixel_data__[self.name]
            except (AttributeError, KeyError):
                # Load them all.
                instance.__pixel_data__ = {}

                with fits.open(expand_path(instance.intermediate_output_path)) as image:
                    model_flux = image[1].data
                
                instance.__pixel_data__.setdefault("model_flux", model_flux)
                
                return instance.__pixel_data__[self.name]

        return self.field




class IntermediatePixelArray(PixelArray):
    
    def __init__(self, ext=None, column_name=None, transform=None, accessor_class=IntermediatePixelArrayAccessor, help_text=None, **kwargs):
        super(IntermediatePixelArray, self).__init__(
            ext=ext,
            column_name=column_name,
            transform=transform,
            accessor_class=accessor_class,
            help_text=help_text,
            **kwargs
        )


class SnowWhite(BaseModel):

    """A result from the white-dwarf pipeline, affectionally known as Snow White."""

    source_pk = ForeignKeyField(Source, null=True, index=True, lazy_load=False)
    spectrum_pk = ForeignKeyField(
        Spectrum, 
        index=True, 
        lazy_load=False,
    )
    
    #> Astra Metadata
    task_pk = AutoField()
    v_astra = TextField(default=__version__)
    created = DateTimeField(default=datetime.datetime.now)
    modified = DateTimeField(default=datetime.datetime.now)
    t_elapsed = FloatField(null=True)
    t_overhead = FloatField(null=True)
    tag = TextField(default="", index=True)

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
    teff = FloatField(null=True)
    e_teff = FloatField(null=True)
    logg = FloatField(null=True)
    e_logg = FloatField(null=True)
    v_rel = FloatField(null=True, help_text="Relative velocity used in stellar parameter fit [km/s]")
    
    #> Formal uncertainties
    raw_e_teff = FloatField(null=True)
    raw_e_logg = FloatField(null=True)
    
    #> Metadata
    result_flags = BitField(default=0)
    flag_low_snr = result_flags.flag(2**0, help_text="Results are suspect because S/N <= 8")
    flag_unconverged = result_flags.flag(2**1, help_text="Fit did not converge")
    flag_teff_grid_edge_bad = result_flags.flag(2**2, help_text="TEFF is edge of grid")
    flag_logg_grid_edge_bad = result_flags.flag(2**3, help_text="LOGG is edge of grid")
    flag_no_flux = result_flags.flag(2**4, help_text="Spectrum has no flux")

    #> Spectral Data
    wavelength = PixelArray(
        accessor_class=LogLambdaArrayAccessor,
        accessor_kwargs=dict(
            crval=3.5523,
            cdelt=1e-4,
            naxis=4648,
        ),
    )    
    model_flux = IntermediatePixelArray(ext=1)

    @property
    def intermediate_output_path(self):
        folders = f"{str(self.spectrum_pk)[-4:-2]:0>2}/{str(self.spectrum_pk)[-2:]:0>2}"
        return f"$MWM_ASTRA/{self.v_astra}/pipelines/snow_white/{folders}/{self.spectrum_pk}.fits"



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
