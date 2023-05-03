from peewee import (
    AutoField,
    FloatField,
    TextField,
    ForeignKeyField,
    IntegerField
)

from astra import __version__
from astra.models.base import BaseModel
from astra.models.fields import BitField
from astra.models.source import Source
from astra.models.spectrum import Spectrum
from astra.models.pipeline import PipelineOutputMixin


class SnowWhite(BaseModel, PipelineOutputMixin):

    """A result from the white-dwarf pipeline, affectionally known as Snow White."""

    sdss_id = ForeignKeyField(Source, index=True)
    spectrum_id = ForeignKeyField(Spectrum, index=True, lazy_load=False)
    
    #> Astra Metadata
    task_id = AutoField()
    v_astra = TextField(default=__version__)
    t_elapsed = FloatField(null=True)
    tag = TextField(default="", index=True)
    
    # Snow White might deliver one or more of:
    # - line ratios;
    # - WD type (based on line ratios);
    # - stellar parameters (if the type is DA)

    #> Task Parameters
    polyfit_order = IntegerField()
    
    #> Line ratios
    ratio_3880_hydrogen_line = FloatField(null=True, help_text="Line ratio for hydrogen line near 3880 A")
    ratio_3975_hydrogen_line = FloatField(null=True, help_text="Line ratio for hydrogen line near 3975 A")
    ratio_4102_hydrogen_line = FloatField(null=True, help_text="Line ratio for hydrogen line near 4102 A")
    ratio_4340_hydrogen_line = FloatField(null=True, help_text="Line ratio for hydrogen line near 4340 A")
    ratio_4860_hydrogen_line = FloatField(null=True, help_text="Line ratio for hydrogen line near 4860 A")
    ratio_6560_hydrogen_line = FloatField(null=True, help_text="Line ratio for hydrogen line near 6560 A")
    ratio_3892_helium_line = FloatField(null=True, help_text="Line ratio for helium line near 3892 A")
    ratio_3965_helium_line = FloatField(null=True, help_text="Line ratio for helium line near 3965 A")
    ratio_4023_helium_line = FloatField(null=True, help_text="Line ratio for helium line near 4023 A")
    ratio_4125_helium_line = FloatField(null=True, help_text="Line ratio for helium line near 4125 A")
    ratio_4390_helium_line = FloatField(null=True, help_text="Line ratio for helium line near 4390 A")
    ratio_4468_helium_line = FloatField(null=True, help_text="Line ratio for helium line near 4468 A")
    ratio_4715_helium_line = FloatField(null=True, help_text="Line ratio for helium line near 4715 A")
    ratio_4925_helium_line = FloatField(null=True, help_text="Line ratio for helium line near 4925 A")
    ratio_5015_helium_line = FloatField(null=True, help_text="Line ratio for helium line near 5015 A")
    ratio_5875_helium_line = FloatField(null=True, help_text="Line ratio for helium line near 5875 A")
    ratio_6685_helium_line = FloatField(null=True, help_text="Line ratio for helium line near 6685 A")
    ratio_7070_helium_line = FloatField(null=True, help_text="Line ratio for helium line near 7070 A")
    ratio_7282_helium_line = FloatField(null=True, help_text="Line ratio for helium line near 7282 A")
    ratio_4675_molecular_carbon_band = FloatField(null=True, help_text="Line ratio for molecular carbon band near 4675 A")
    ratio_5080_molecular_carbon_band = FloatField(null=True, help_text="Line ratio for molecular carbon band near 5080 A")
    ratio_3932_ca_k_line = FloatField(null=True, help_text="Line ratio for Ca K line near 3932 A")
    ratio_3968_ca_h_line = FloatField(null=True, help_text="Line ratio for Ca H line near 3968 A")

    #> Classification
    wd_type = TextField(null=True)

    #> Stellar Parameters
    teff = FloatField(null=True)
    e_teff = FloatField(null=True)
    logg = FloatField(null=True)
    e_logg = FloatField(null=True)
    v_rel = FloatField(null=True, help_text="Relative radial velocity used in stellar parameter fit [km/s]")
    
    #> Metadata
    conditioned_on_parallax = FloatField(null=True, help_text="Parallax used in stellar parameter fit [mas]")
    conditioned_on_phot_g_mean_mag = FloatField(null=True, help_text="G mag used in stellar parameter fit [mag]")    
    result_flags = BitField(default=0)

    #> Summary Statistics
    snr = FloatField(null=True)
    chi_sq = FloatField(null=True)
    reduced_chi_sq = FloatField(null=True)