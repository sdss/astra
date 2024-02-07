
import os
import h5py as h5
import subprocess
from tqdm import tqdm
from datetime import datetime
from peewee import JOIN, ModelSelect, chunked
from tempfile import mkdtemp
from time import sleep
from typing import Optional, Iterable, Union

from astra import task, __version__
from astra.utils import log, expand_path

from astra.models import ASPCAP
from astra.models.grok import Grok
from astra.models.mwm import (ApogeeCombinedSpectrum, ApogeeRestFrameVisitSpectrum)

from astra.pipelines.ferre.utils import get_apogee_pixel_mask
import numpy as np

APOGEE_FERRE_MASK = get_apogee_pixel_mask()

def unmask(array, fill_value=np.nan):
    unmasked_array = fill_value * np.ones(APOGEE_FERRE_MASK.shape)
    unmasked_array[APOGEE_FERRE_MASK] = array
    return unmasked_array


@task
def grok(
    spectra: Optional[Iterable[Union[ApogeeCombinedSpectrum, ApogeeRestFrameVisitSpectrum]]] = (
        ApogeeCombinedSpectrum
        .select()
        .distinct(ApogeeCombinedSpectrum.spectrum_pk)
        .join(Grok, JOIN.LEFT_OUTER, on=(ApogeeCombinedSpectrum.spectrum_pk == Grok.spectrum_pk))
        .switch(ApogeeCombinedSpectrum)
        .join(ASPCAP, on=(ApogeeCombinedSpectrum.source_pk == ASPCAP.source_pk))
        .where(
            Grok.spectrum_pk.is_null()
        &   (6000 >= ASPCAP.teff) & (ASPCAP.teff >= 5000)
        &   (5 >= ASPCAP.logg) & (ASPCAP.logg >= 3)
        &   (0.5 >= ASPCAP.m_h_atm) & (ASPCAP.m_h_atm >= -1)
        )        
    ),
    page=None,
    limit=None,
    n_jobs=32,
    **kwargs
) -> Iterable[Grok]:
    """
    
    Note: currently the default is to find things that ASPCAP says are in the range of the grid, and to only use Combined (mwmStar) spectra
    """
    
    if limit is not None and isinstance(spectra, ModelSelect):
        if page is not None:
            spectra = spectra.paginate(page, limit)
        else:
            spectra = spectra.limit(limit)
    
    
    # create a temporary working directory/path
    pwd = create_pwd()

    get_input_path = lambda chunk: f"{pwd}/input_{chunk}.list"
    get_output_path = lambda chunk: f"{pwd}/output_{chunk}.h5"
    
    n_spectra = limit or len(spectra)
    n_jobs = min(n_jobs, n_spectra)
    
    chunk_size = int(np.ceil(n_spectra / n_jobs))
    
    chunks = list(chunked(spectra, chunk_size))
    n_jobs = len(chunks)
    log.info(f"Running grok on {n_spectra} spectra with {n_jobs} jobs ({chunk_size} chunk size)")
    
    for i, chunk in enumerate(chunks):
        with open(get_input_path(i), "w") as fp:
            fp.write(prepare_input_list(chunk))

    env = os.environ.copy()
    env.setdefault("GROK_GRID_FILE", expand_path("$MWM_ASTRA/pipelines/Grok/dense_grid_2023_12.h5"))

    processes = []
    for i in range(n_jobs):
        print(get_input_path(i), get_output_path(i))
        process = subprocess.Popen(
            ["julia", "run_grok.jl", get_input_path(i), get_output_path(i)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd="/uufs/chpc.utah.edu/common/home/u6020307/astra/python/astra/pipelines/grok/",
            env=env,
        )
        processes.append(process)
    
    
    # wait for processes to finish
    with tqdm(total=n_spectra, unit="spectra") as pb:
        while True:
                
            for i, process in enumerate(processes):
                code = process.poll()
                if code is not None:
                    processes.pop(i)
                    
                    output_path = process.args[-1]
                    index = int(output_path.split("_")[-1].split(".")[0])

                    if code > 0:
                        for spectrum in chunks[index]:    
                            yield Grok(
                                spectrum_pk=spectrum.spectrum_pk,
                                source_pk=spectrum.source_pk,
                                output_path=output_path,
                                flag_runtime_failure=True
                            )
                            pb.update()                        
                    else:                        
                        for result in post_process_grok(chunks[index], output_path):
                            yield result
                            pb.update()
                    break
            else:
                sleep(1)
                
            if len(processes) == 0:
                break
            
    log.info("Done")
    

def post_process_grok(spectra, output_path):
    # parse the output file
    elements = ("mg_h", "na_h", "al_h") # TODO: these should either be given to protogrok, or parsed from it, or something
    #label_names = ("teff", "logg", "v_micro", "m_h")
    label_names = (
        "teff", 
        "logg", 
        #"v_micro", 
        "m_h",
        "v_sini"
    )

    with h5.File(output_path, "r") as fp:
        for i, spectrum in enumerate(spectra):
            
            kwds = dict(
                spectrum_pk=spectrum.spectrum_pk,
                source_pk=spectrum.source_pk,
                output_path=output_path,
                row_index=i,
                t_elapsed=fp["runtimes"][i],
            )
            # Add best node
            kwds.update(dict(zip(
                [f"grid_{label_name}" for label_name in label_names],
                fp["best_nodes"][i]
            )))
            # Stellar parameters
            kwds.update(dict(zip(label_names, fp["stellar_params"][i])))
            kwds.update(dict(zip(elements, fp["detailed_abundances"][i])))
            
            # Make a figure.
            '''
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(12, 2))
            ax.plot(spectrum.wavelength, spectrum.flux / spectrum.continuum, c='k')
            ax.plot(spectrum.wavelength, unmask(fp["model_spectra"][i]), c="tab:red")
            val = f"{spectrum.source_pk:0>4.0f}"
            
            output_path = f"$MWM_ASTRA/{__version__}/pipelines/Grok/{val[-4:-2]}/{val[-2:]}/{val}.png"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            fig.savefig(output_path, dpi=300)
            del fig
            plt.close("all")
            '''
            
            yield Grok(**kwds)
            
    
def create_pwd():
    today = datetime.now().strftime("%Y-%m-%d")
    parent_dir = expand_path(f"$MWM_ASTRA/{__version__}/pipelines/Grok/")
    os.makedirs(parent_dir, exist_ok=True)
    return mkdtemp(prefix=f"{today}-", dir=parent_dir)

def prepare_input_list(spectra):
    lines = []
    for spectrum in spectra:
        hdu = 3 if spectrum.telescope.lower().startswith("apo") else 4
        lines.append(f"{hdu} {expand_path(spectrum.path)}")
    return "\n".join(lines)


if __name__ == "__main__":
    
    from astra.models import Source
    from astra.models.mwm import ApogeeCombinedSpectrum
    from astra.pipelines.grok import grok
    
    #sol = Source.get(sdss4_apogee_id="VESTA")
    
    spectrum = ApogeeCombinedSpectrum.get()
    
    from astra.models import ASPCAP
    
    spectra = list(
        ApogeeCombinedSpectrum
        .select()
        .join(ASPCAP, on=(ApogeeCombinedSpectrum.source_pk == ASPCAP.source_pk))
        .where(
            (6000 >= ASPCAP.teff) 
        &   (ASPCAP.teff >= 5000)
        &   (5 >= ASPCAP.logg)
        &   (ASPCAP.logg >= 3)
        &   (ASPCAP.m_h_atm >= -1)
        &   (ASPCAP.m_h_atm <= 0.5)
        )
        .limit(1)
    )
    
    results = list(grok(spectra))
    
    
    # export the grok file
    from astra.models.grok import Grok
    from astra.models.mwm import ApogeeCombinedSpectrum
    from astra.products.pipeline_summary import create_astra_all_star_product, ignore_field_name_callable
    
    def ignore_field(name):
        return ignore_field_name_callable(name) or (name in ("input_spectrum_pks", ))
    
    create_astra_all_star_product(
        Grok,
        apogee_spectrum_model=ApogeeCombinedSpectrum,
        ignore_field_name_callable=ignore_field
    )
