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
    spectra: Iterable[ApogeeCoaddedSpectrumInApStar],
    #model_path: str = "$MWM_ASTRA/pipelines/astroNN_dist/astroNN_gaia_dr17_model_3", # for TensorFlow version
    model_path: str = "$MWM_ASTRA/pipelines/astroNN_dist/astroNN_dist_model_parameter.pt",  # for PyTorch version
    #model_path: str = "/uufs/chpc.utah.edu/common/home/sdss50/sdsswork/users/u6039136/software/git/astroNN_projects/astroNN_dist_model_parameter.pt",  # for PyTorch version
    parallel: Optional[bool] = False,
    batch_size: Optional[int] = 100,
    cpu_count: Optional[int] = 1,
    **kwargs
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





def test_astronn():
    import os
    import numpy as np
    from astropy.io import fits

    path_astronn = '/uufs/chpc.utah.edu/common/home/sdss50/dr17/apogee/vac/apogee-astronn/apogee_astroNN-DR17.fits'

    data_astronn = fits.open(path_astronn)[1].data
    print(len(data_astronn))

    mask_good = np.logical_and(data_astronn['fakemag_error']/data_astronn['fakemag'] < 0.1, data_astronn['telescope']=='apo25m')
    mask_good = np.logical_and(mask_good, data_astronn['dist'] < 8e3)
    data_good = data_astronn[mask_good]
    print(len(data_good))

    q = (
        ApogeeCoaddedSpectrumInApStar
        .select()
        .distinct(Source.sdss4_apogee_id)
        .join(Source, on=(Source.pk == ApogeeCoaddedSpectrumInApStar.source_pk))
        .where(
            Source.sdss4_apogee_id.in_(list(data_good['apogee_id']))
        )
        .limit(1000)
    )

    results = list(astronn_dist(q))
    results_serial = list(astronn_dist(list(q)[:10], parallel=False))

    import matplotlib.pyplot as plt
    sdss4_apogee_ids = [Source.get(s.source_pk).sdss4_apogee_id for s in results_serial]

    x = [data_good['dist'][data_good['apogee_id'] == i][0] for i in sdss4_apogee_ids]
    y = [s.dist for s in results]
    y_serial = [s.dist for s in results_serial]
    z = [s.ebv for s in results]

    fig, ax = plt.subplots()
    ax.scatter(x, y_serial)
    
