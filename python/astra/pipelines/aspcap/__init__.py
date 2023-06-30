from astra import task
from astra.utils import log
from astra.models.spectrum import Spectrum
from astra.pipelines.aspcap.coarse import coarse_stellar_parameters
from astra.pipelines.aspcap.stellar_parameters import stellar_parameters, FerreStellarParameters
from astra.pipelines.aspcap.abundances import abundances

from typing import Optional, Iterable, List, Tuple, Callable, Union


@task
def aspcap(
    spectra: Iterable[Spectrum], 
    parent_dir: str, 
    initial_guess_callable: Optional[Callable] = None,
    header_paths: Optional[Union[List[str], Tuple[str], str]] = "$MWM_ASTRA/pipelines/aspcap/synspec_dr17_marcs_header_paths.list",
    weight_path: Optional[str] = "$MWM_ASTRA/pipelines/aspcap/masks/global.mask",
    element_weight_paths: str = "$MWM_ASTRA/pipelines/aspcap/masks/elements.list",
    operator_kwds: Optional[dict] = None,
    **kwargs
):
    """
    Run the ASPCAP pipeline on some spectra.
    
    .. warning:: 
        This is task for convenience. 
        
        If you want efficiency, you should use the `pre_` and `post_` tasks for each stage in the pipeline.
    
    :param spectra:
        The spectra to be processed.
    
    :param parent_dir:
        The parent directory where these FERRE executions will be planned.    
    
    :param initial_guess_callable: [optional]
        A callable that returns an initial guess for the stellar parameters. 
    
    :param header_paths: [optional]
        The path to a file containing the paths to the FERRE header files. This file should contain one path per line.
    
    :param weight_path: [optional]
        The path to the FERRE weight file to use during the coarse and main stellar parameter stage.

    :param element_weight_paths: [optional]
        A path containing FERRE weight files for different elements, which will be used in the chemical abundances stage.

    :param operator_kwds: [optional]
        A dictionary of keywords to supply to the `astra.pipelines.ferre.operator.FerreOperator` class.
    
    Keyword arguments
    -----------------
    All additional keyword arguments will be passed through to `astra.pipelines.ferre.pre_process.pre_process.ferre`. 
    Some handy keywords include:
    continuum_order: int = 4,
    continuum_reject: float = 0.3,
    continuum_observations_flag: int = 1,
    """

    # Set up the tasks first, but they won't be executed until we start iterating over those task generators.
    coarse = coarse_stellar_parameters(
        spectra,
        parent_dir=parent_dir,
        initial_guess_callable=initial_guess_callable,
        header_paths=header_paths,
        weight_path=weight_path,
        operator_kwds=operator_kwds,
        **kwargs
    )

    param = stellar_parameters(
        spectra,
        parent_dir=parent_dir,
        weight_path=weight_path,
        operator_kwds=operator_kwds,
        **kwargs
    )

    abund = abundances(
        spectra,
        parent_dir=parent_dir,
        element_weight_paths=element_weight_paths,
        operator_kwds=operator_kwds,
        **kwargs
    )
    
    log.info(f"Running ASPCAP coarse stage")
    coarse = list(coarse)

    log.info(f"Running ASPCAP stellar parameter stage")
    param_results = list(param)
    
    log.info(f"Running ASPCAP chemical abundances stage")
    #abund_results = list(abund)
    
    yield from param_results
    
    