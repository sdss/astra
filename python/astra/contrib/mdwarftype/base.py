import numpy as np
import os
import multiprocessing as mp
from peewee import FloatField, TextField, IntegerField
from typing import Iterable
from astropy.nddata import InverseVariance
from itertools import cycle
from functools import cache
from tqdm import tqdm

from astra.base import task
from astra.database.astradb import DataProduct, SDSSOutput, _get_sdss_metadata
from astra.utils import log, flatten, expand_path, executable
from astra.tools.spectrum import SpectrumList
from astra.tools.spectrum.utils import spectrum_overlaps
from astropy import units as u


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
    bitmask_flag = IntegerField(default=0)






@task
def classify_m_dwarf(
    data_product: Iterable[DataProduct],
    template_list: str = "$MWM_ASTRA/component_data/M-dwarf-templates/template.list",
    continuum_method: str = "astra.tools.continuum.Scalar",
    continuum_kwargs: dict = dict(mask=[(0, 7495), (7505, 11_000)])
) -> Iterable[MDwarfTypeOutput]:
    """
    Classify a single M dwarf using APOGEENet.
    """

    template_flux, template_type = read_template_fluxes_and_types(template_list)

    N_workers = 32
    q_output = mp.Queue()
    args = (q_output, template_flux, template_type, continuum_method, continuum_kwargs)

    qps = []
    for i in range(N_workers):
        q = mp.Queue()
        p = mp.Process(target=_gimli, args=(q, *args))
        p.start()
        qps.append((q, p))
        
    # Distribute the work.
    for (q, p), data_product in zip(cycle(qps), flatten(data_product)):
        q.put(data_product)

    # Tell the workers they're done.
    for (q, p) in qps:
        q.put(None)

    N = 0
    while True:
        try:
            result = q_output.get(timeout=60)
        except:
            # Let's assume we're done.
            break
        else:
            if result is None:
                N += 1
                if N == N_workers:
                    break
            else:
                yield MDwarfTypeOutput(**result)

    for q, p in qps:
        p.join()
    



def _classify_m_dwarf(data_product, template_flux, template_type, f_continuum):
    for spectrum in SpectrumList.read(data_product.path):
        if not spectrum_overlaps(spectrum, 5_000 * u.Angstrom):
            # Skip non-BOSS spectra.
            continue
        
        continuum = f_continuum.fit(spectrum)(spectrum)
        flux = spectrum.flux.value / continuum
        ivar = spectrum.uncertainty.represent_as(InverseVariance).array
        chi_sqs = np.nansum((flux - template_flux)**2 * ivar, axis=1)
        index = np.argmin(chi_sqs)
        chi_sq = chi_sqs[index]
        continuum_level = continuum.flatten()[0]
        spectral_type, sub_type = template_type[index]

        bitmask_flag = 1 if spectral_type == "K5.0" else 0

        kwds = dict(
            data_product=data_product,
            continuum=continuum_level,
            spectral_type=spectral_type,
            sub_type=sub_type,
            chi_sq=chi_sq,
            bitmask_flag=bitmask_flag
        )
        # a Spectrum1D can't be serialized.
        kwds.update(_get_sdss_metadata(data_product, spectrum))
        yield kwds



def _gimli(q_input, q_output, template_flux, template_type, continuum_method, continuum_kwargs):

    f_continuum = executable(continuum_method)(**continuum_kwargs)    

    while True:
        data_product = q_input.get()
        if data_product is None:
            break

        for result in _classify_m_dwarf(data_product, template_flux, template_type, f_continuum):
            q_output.put(result)
    
    print(f"Worker done")
    q_output.put(None)
    return None


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


if __name__ == "__main__":

    from astra.database.astradb import DataProduct
    from astra.sdss.datamodels.mwm import MWMSourceStatus

    q = (
        DataProduct
        .select()
        .join(MWMSourceStatus, on=(DataProduct.source_id == MWMSourceStatus.source_id))
        .where(
            (DataProduct.filetype == "mwmStar")
        &   (MWMSourceStatus.num_boss_apo_visits_in_stack > 0)
        )
        .limit(100)
    )


    foo = list(classify_m_dwarf(list(q)))



