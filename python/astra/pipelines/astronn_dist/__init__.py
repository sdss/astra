import scipy.optimize as op # build problems
from typing import Iterable, Optional
from peewee import JOIN
#import tensorflow_probability as tfp
from astra import task, __version__
from astra.models import ApogeeCoaddedSpectrumInApStar
from astra.models.astronn_dist import AstroNNdist
#from astra.pipelines.astronn_dist.utils import read_model # for TensorFlow version
from astra.pipelines.astronn_dist.network import read_model # for PyTorch version
from astra.pipelines.astronn_dist.base import _prepare_data, _worker, parallel_batch_read, _inference

@task
def astronn_dist(
    spectra: Optional[Iterable] = (
        ApogeeCoaddedSpectrumInApStar
        .select()
        .join(
            AstroNNdist, 
            JOIN.LEFT_OUTER,
            on=(
                (ApogeeCoaddedSpectrumInApStar.spectrum_pk == AstroNNdist.spectrum_pk)
            &   (AstroNNdist.v_astra == __version__)
            )
        )
        .where(AstroNNdist.spectrum_pk.is_null())
    ),
    #model_path: str = "$MWM_ASTRA/pipelines/astroNN_dist/astroNN_gaia_dr17_model_3", # for TensorFlow version
    model_path: str = "$MWM_ASTRA/pipelines/astroNN_dist/astroNN_dist_model_parameter.pt",  # for PyTorch version
    #model_path: str = "/uufs/chpc.utah.edu/common/home/sdss50/sdsswork/users/u6039136/software/git/astroNN_projects/astroNN_dist_model_parameter.pt",  # for PyTorch version
    parallel: Optional[bool] = True,
    batch_size: Optional[int] = 100,
    cpu_count: Optional[int] = 1,
) -> Iterable[AstroNNdist]:
    """
    Estimate astrophysical parameters for a stellar spectrum given a pre-trained neural network.
    """

    model = read_model(model_path)

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