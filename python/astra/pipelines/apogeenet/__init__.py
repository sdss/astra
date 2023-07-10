from typing import Iterable, Optional
from astra import task
from astra.models import Spectrum
from astra.models.pipelines import ApogeeNet
from astra.pipelines.apogeenet.network import read_network
from astra.pipelines.apogeenet.base import _worker, parallel_batch_read, _inference

__all__ = ["apogee_net"]

@task
def apogee_net(
    spectra: Iterable[Spectrum],
    network_path: str = "$MWM_ASTRA/pipelines/APOGEENet/model.pt",
    large_error: Optional[float] = 1e10,
    num_uncertainty_draws: Optional[int] = 100,
) -> Iterable[ApogeeNet]:
    """
    Estimate astrophysical parameters for a stellar spectrum given a pre-trained neural network.
    """

    network = read_network(network_path)

    for batch in parallel_batch_read(_worker, spectra, (large_error, ), gpu_batch_size=100, cpu_count=4):
        yield from _inference(network, batch, num_uncertainty_draws)

