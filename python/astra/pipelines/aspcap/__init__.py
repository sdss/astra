from astra import task
from astra.models import Spectrum
from astra.pipelines.aspcap.coarse import coarse_stellar_parameters
from astra.pipelines.aspcap.stellar_parameters import stellar_parameters

from typing import Optional, Iterable, List, Tuple, Callable, Union

@task
def aspcap(spectra: Iterable[Spectrum], parent_dir: str, **kwargs):
    """
    Run the ASPCAP pipeline on the given spectra.
    
    This is task for convenience. If you want efficiency, you should use the `pre_` and `post_` tasks
    for each stage in the pipeline.
    """
    
    coarse = list(coarse_stellar_parameters(spectra, parent_dir, **kwargs))
    params = list(stellar_parameters(spectra, parent_dir, **kwargs))
    raise a
