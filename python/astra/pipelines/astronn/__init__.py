from typing import Iterable, Optional
import tensorflow_probability as tfp
from astra import task
from astra.models.astronn import AstroNN
from astra.pipelines.astronn.utils import read_model
from astra.pipelines.astronn.base import _prepare_data, _worker, parallel_batch_read, _inference

@task
def astronn(
    spectra: Iterable,
    model_path: str = "/uufs/chpc.utah.edu/common/home/sdss50/sdsswork/users/u6039136/software/git/astroNN_projects/astroNN_APOGEE_VAC/astroNN_retrain_2_shi/",
    #model_path: str = "/uufs/chpc.utah.edu/common/home/sdss50/sdsswork/users/u6039136/software/git/astroNN_projects/astroNN_APOGEE_VAC/astroNN_0617_run001/",
    parallel: Optional[bool] = False,
) -> Iterable[AstroNN]:
    """
    Estimate astrophysical parameters for a stellar spectrum given a pre-trained neural network.
    """

    model = read_model(model_path)

    if parallel: # work for pipelines.sdss.org
        for batch in parallel_batch_read(_worker, spectra, batch_size=100, cpu_count=4):
            yield from _inference(model, batch)
    else:
        try:
            for spectrum in spectra.iterator():
                yield from _inference(model, [_prepare_data(spectrum)])
        except:
            for spectrum in spectra:
                yield from _inference(model, [_prepare_data(spectrum)])