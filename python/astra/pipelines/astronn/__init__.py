import scipy.optimize as op # build problems
from typing import Iterable, Optional
from astra import task
from astra.models.astronn import AstroNN
#from astra.pipelines.astronn.utils import read_model # for TensorFlow version
from astra.pipelines.astronn.network import read_model # for PyTorch version
from astra.pipelines.astronn.base import _prepare_data, _worker, parallel_batch_read, _inference
from peewee import ModelSelect

@task
def astronn(
    spectra: Iterable,
    #model_path: str = "$MWM_ASTRA/pipelines/astronn/astroNN_retrain_2_shi",     # for TensorFlow version
    model_path: str = "$MWM_ASTRA/pipelines/astronn/astroNN_model_parameter_dr17_shi.pt", # for PyTorch version
    parallel: Optional[bool] = True,
    batch_size: Optional[int] = 100,
    cpu_count: Optional[int] = 1,
    **kwargs
) -> Iterable[AstroNN]:
    """
    Estimate astrophysical parameters for a stellar spectrum given a pre-trained neural network.
    """

    model = read_model(model_path)
    if isinstance(spectra, ModelSelect):
        spectra = spectra.iterator()

    if parallel: # work for pipelines.sdss.org
        for batch in parallel_batch_read(_worker, spectra, batch_size=batch_size, cpu_count=cpu_count):
            yield from _inference(model, batch)
    else:
        try:
            for spectrum in spectra.iterator():
                yield from _inference(model, [_prepare_data(spectrum)])
        except:
            for spectrum in spectra:
                yield from _inference(model, [_prepare_data(spectrum)])