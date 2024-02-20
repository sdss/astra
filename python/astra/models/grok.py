import datetime
import numpy as np
from peewee import (
    AutoField,
    FloatField,
    TextField,
    ForeignKeyField,
    IntegerField,
    BitField,
    DateTimeField
)

from astra import __version__
from astra.models.base import BaseModel
from astra.models.fields import BitField, PixelArray, LogLambdaArrayAccessor, PixelArrayAccessorHDF
from astra.models.source import Source
from astra.models.spectrum import Spectrum
from astra.models.pipeline import PipelineOutputMixin
from astra.glossary import Glossary
from astra.pipelines.ferre.utils import get_apogee_pixel_mask

APOGEE_FERRE_MASK = get_apogee_pixel_mask()

def unmask(array, fill_value=np.nan):
    unmasked_array = fill_value * np.ones(APOGEE_FERRE_MASK.shape)
    unmasked_array[APOGEE_FERRE_MASK] = array
    return unmasked_array

class Grok(BaseModel, PipelineOutputMixin):

    """A result from the Grok pipeline."""

    source_pk = ForeignKeyField(Source, null=True, index=True, lazy_load=False, help_text=Glossary.source_pk)
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

    #> Coarse estimates
    coarse_teff = FloatField(null=True, help_text="Coarse estimate of effective temperature [K]")
    coarse_logg = FloatField(null=True, help_text="Coarse estimate of surface gravity [dex]")
    coarse_c_m = FloatField(null=True, help_text="Coarse estimate of [C/M] [dex]")
    coarse_m_h = FloatField(null=True, help_text="Coarse estimate of [M/H] [dex]")
    coarse_n_m = FloatField(null=True, help_text="Coarse estimate of [N/M] [dex]")
    coarse_v_micro = FloatField(null=True, help_text="Coarse estimate of microturbulence [km/s]")
    coarse_v_sini = FloatField(null=True, help_text="Coarse estimate of v sini [km/s]")
    coarse_chi2 = FloatField(null=True, help_text="\chi2 value of coarse estimate")

    #> Stellar labels
    teff = FloatField(null=True, help_text=Glossary.teff)
    e_teff = FloatField(null=True, help_text=Glossary.e_teff)
    logg = FloatField(null=True, help_text=Glossary.logg)
    e_logg = FloatField(null=True, help_text=Glossary.e_logg)
    v_micro = FloatField(null=True, help_text=Glossary.v_micro)
    e_v_micro = FloatField(null=True, help_text=Glossary.e_v_micro)
    m_h = FloatField(null=True, help_text=Glossary.m_h)
    e_m_h = FloatField(null=True, help_text=Glossary.e_m_h)
    c_m = FloatField(null=True, help_text=Glossary.c_m)
    e_c_m = FloatField(null=True, help_text=Glossary.e_c_m)
    n_m = FloatField(null=True, help_text=Glossary.n_m)
    e_n_m = FloatField(null=True, help_text=Glossary.e_n_m)
    v_sini = FloatField(null=True, help_text=Glossary.v_sini)
    e_v_sini = FloatField(null=True, help_text=Glossary.e_v_sini)
    
    #> Chemical abundances
    c_h = FloatField(null=True, help_text="Carbon abundance [dex]")
    e_c_h = FloatField(null=True, help_text="Error on carbon abundance [dex]")
    n_h = FloatField(null=True, help_text="Nitrogen abundance [dex]")
    e_n_h = FloatField(null=True,  help_text="Error on nitrogen abundance [dex]")
    o_h = FloatField(null=True, help_text="Oxygen abundance [dex]")
    e_o_h = FloatField(null=True, help_text="Error on oxygen abundance [dex]")
    na_h = FloatField(null=True, help_text="Sodium abundance [dex]")
    e_na_h = FloatField(null=True,  help_text="Error on sodium abundance [dex]")
    mg_h = FloatField(null=True, help_text="Magnesium abundance [dex]")
    e_mg_h = FloatField(null=True, help_text="Error on magnesium abundance [dex]")
    al_h = FloatField(null=True, help_text="Aluminum abundance [dex]")
    e_al_h = FloatField(null=True, help_text="Error on aluminum abundance [dex]")
    si_h = FloatField(null=True, help_text="Silicon abundance [dex]")
    e_si_h = FloatField(null=True, help_text="Error on silicon abundance [dex]")
    p_h = FloatField(null=True, help_text="Phosphorus abundance [dex]")
    e_p_h = FloatField(null=True, help_text="Error on phosphorus abundance [dex]")
    s_h = FloatField(null=True, help_text="Sulfur abundance [dex]")
    e_s_h = FloatField(null=True, help_text="Error on sulfur abundance [dex]")
    k_h = FloatField(null=True, help_text="Potassium abundance [dex]")
    e_k_h = FloatField(null=True, help_text="Error on potassium abundance [dex]")
    ca_h = FloatField(null=True, help_text="Calcium abundance [dex]")
    e_ca_h = FloatField(null=True,  help_text="Error on calcium abundance [dex]")
    ti_h = FloatField(null=True, help_text="Titanium abundance [dex]")
    e_ti_h = FloatField(null=True, help_text="Error on titanium abundance [dex]")
    v_h = FloatField(null=True, help_text="Vanadium abundance [dex]")
    e_v_h = FloatField(null=True, help_text="Error on vanadium abundance [dex]")
    cr_h = FloatField(null=True, help_text="Chromium abundance [dex]")
    e_cr_h = FloatField(null=True, help_text="Error on chromium abundance [dex]")
    mn_h = FloatField(null=True, help_text="Manganese abundance [dex]")
    e_mn_h = FloatField(null=True, help_text="Error on manganese abundance [dex]")
    fe_h = FloatField(null=True, help_text="Iron abundance [dex]")
    e_fe_h = FloatField(null=True, help_text="Error on iron abundance [dex]")
    co_h = FloatField(null=True, help_text="Cobalt abundance [dex]")
    e_co_h = FloatField(null=True, help_text="Error on cobalt abundance [dex]")
    ni_h = FloatField(null=True, help_text="Nickel abundance [dex]")
    e_ni_h = FloatField(null=True, help_text="Error on nickel abundance [dex]")

    #> Path references
    #output_path = TextField(help_text="Path to output file")
    #row_index = IntegerField(help_text="Index of result in output file")
    
    #> Summary Statistics
    chi2 = FloatField(null=True, help_text=Glossary.chi2)
    #rchi2 = FloatField(null=True, help_text=Glossary.rchi2)
    result_flags = BitField(default=0, help_text="Flags describing the results")
    flag_runtime_failure = result_flags.flag(2**0, help_text="Runtime failure")

    #> Spectral data
    wavelength = PixelArray(
        accessor_class=LogLambdaArrayAccessor,
        accessor_kwargs=dict(
            crval=4.179,
            cdelt=6e-6,
            naxis=8575,
        ),
        help_text=Glossary.wavelength
    )
    ##TODO: Probably don't want this accessing `.path`...
    model_flux = PixelArray(
        help_text=Glossary.model_flux, 
        accessor_class=PixelArrayAccessorHDF, 
        column_name="model_spectra",
        transform=unmask
    )
    ## TODO: should we be including the continuum somehow? because `model_flux` here is rectified, i think.