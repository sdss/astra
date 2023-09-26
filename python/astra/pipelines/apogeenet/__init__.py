from typing import Iterable, Optional
from astra import task
from astra.models.apogeenet import ApogeeNet
from astra.pipelines.apogeenet.network import read_network
from astra.pipelines.apogeenet.base import _prepare_data, _worker, parallel_batch_read, _inference

__all__ = ["apogeenet"]

@task
def apogeenet(
    spectra: Iterable,
    network_path: str = "$MWM_ASTRA/pipelines/APOGEENet/model.pt",
    large_error: Optional[float] = 1e10,
    num_uncertainty_draws: Optional[int] = 100,
    parallel: Optional[bool] = False,
) -> Iterable[ApogeeNet]:
    """
    Estimate astrophysical parameters for a stellar spectrum given a pre-trained neural network.
    """

    network = read_network(network_path)

    if parallel:
        # This used to work when we were writing to operations.sdss.org, but no longer works now that
        # we are using pipelines.sdss.org. I don't know why.
        for batch in parallel_batch_read(_worker, spectra, (large_error, ), batch_size=100, cpu_count=4):
            yield from _inference(network, batch, num_uncertainty_draws)
    else:
        try:
            for spectrum in spectra.iterator():
                yield from _inference(network, [_prepare_data(spectrum, large_error)], num_uncertainty_draws)
        except:
            for spectrum in spectra:
                yield from _inference(network, [_prepare_data(spectrum, large_error)], num_uncertainty_draws)
            
