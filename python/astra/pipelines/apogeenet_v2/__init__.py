from typing import Iterable, Optional, Union
from astra import task
from astra.models import ApogeeNetV2, ApogeeVisitSpectrumInApStar, ApogeeCoaddedSpectrumInApStar
from astra.pipelines.apogeenet_v2.network import read_network
from astra.pipelines.apogeenet_v2.base import _prepare_data, _worker, parallel_batch_read, _inference
from peewee import JOIN, ModelSelect

__all__ = ["apogeenet"]

@task
def apogeenet_v2(
    spectra: Optional[Iterable[Union[ApogeeVisitSpectrumInApStar, ApogeeCoaddedSpectrumInApStar]]] = (
        ApogeeCoaddedSpectrumInApStar
        .select()
        .join(ApogeeNetV2, JOIN.LEFT_OUTER, on=(ApogeeCoaddedSpectrumInApStar.spectrum_pk == ApogeeNetV2.spectrum_pk))
        .where(ApogeeNetV2.spectrum_pk.is_null())
    ),
    network_path: str = "$MWM_ASTRA/pipelines/APOGEENet/model.pt",
    large_error: Optional[float] = 1e10,
    num_uncertainty_draws: Optional[int] = 100,
    parallel: Optional[bool] = False,
    limit: Optional[int] = None,
    **kwargs
) -> Iterable[ApogeeNetV2]:
    """
    Estimate astrophysical parameters for a stellar spectrum given a pre-trained neural network.
    """

    if isinstance(spectra, ModelSelect):
        if limit is not None:
            spectra = spectra.limit(limit)

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
            
