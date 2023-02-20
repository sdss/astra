import numpy as np
import os
from peewee import FloatField, TextField
from typing import Iterable
from astropy.nddata import InverseVariance
from functools import cache

from astra.base import task
from astra.database.astradb import DataProduct, SDSSOutput
from astra.utils import expand_path, executable
from astra.tools.spectrum import SpectrumList


@cache
def read_template_fluxes_and_types(template_list):
    with open(expand_path(template_list), "r") as fp:
        template_list = list(map(str.strip, fp.readlines()))

    template_flux = list(map(read_and_resample_template, template_list))
    template_type = list(map(get_template_type, template_list))
    return (template_flux, template_type)


class MDwarfTypeOutput(SDSSOutput):

    continuum = FloatField(null=True)
    spectral_type = TextField()
    sub_type = FloatField()
    chi_sq = FloatField()


@task
def classify_m_dwarf(
    data_product: DataProduct,
    template_list: str = "$MWM_ASTRA/component_data/M-dwarf-templates/template.list",
    continuum_method: str = "astra.tools.continuum.Scalar",
    continuum_kwargs: dict = dict(mask=[(0, 7495), (7505, 11_000)])
) -> Iterable[MDwarfTypeOutput]:
    """
    Classify a single M dwarf using APOGEENet.
    """

    template_flux, template_type = read_template_fluxes_and_types(template_list)

    f_continuum = executable(continuum_method)(**continuum_kwargs)    

    for spectrum in SpectrumList.read(data_product.path):

        continuum = f_continuum.fit(spectrum)(spectrum)
        flux = spectrum.flux.value / continuum
        ivar = spectrum.uncertainty.represent_as(InverseVariance).array

        chi_sq = np.nansum((flux - template_flux)**2 * ivar, axis=1)
        index = np.argmin(chi_sq)
        spectral_type, sub_type = template_type[index]

                            
        yield MDwarfTypeOutput(
            data_product=data_product,
            spectrum=spectrum,
            continuum=continuum.flatten()[0],
            spectral_type=spectral_type,
            sub_type=sub_type,
            chi_sq=chi_sq[index],
        )
        


def get_template_type(path):
    _, spectral_type, sub_type = os.path.basename(path).split("_")
    sub_type = sub_type[:-4]
    return spectral_type, sub_type


def read_and_resample_template(path):
    log_wl, flux = np.loadtxt(
        expand_path(path), 
        skiprows=1, 
        delimiter=",",
        usecols=(1, 2)
    ).T
    common_wavelength = 10**(3.5523 + 0.0001 * np.arange(4648))
    # Interpolate to the BOSS wavelength grid
    return np.interp(common_wavelength, 10**log_wl, flux, left=np.nan, right=np.nan)
