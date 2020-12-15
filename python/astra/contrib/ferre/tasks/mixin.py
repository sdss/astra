
import astra
import datetime
from astra.tasks.base import BaseTask
from astra.tasks.slurm import slurm_mixin_factory


SlurmMixin = slurm_mixin_factory("FERRE")


class FerreMixin(SlurmMixin, BaseTask):

    """ A mixin class for FERRE tasks. """

    task_namespace = "FERRE"

    # Instead of providing the grid header path here, we will pass it through other parameters
    # that can be used to reconstruct the grid header path.
    interpolation_order = astra.IntParameter(
        default=3,
        config_path=dict(section=task_namespace, name="interpolation_order")
    )
    init_algorithm_flag = astra.IntParameter(
        default=1,
        config_path=dict(section=task_namespace, name="init_algorithm_flag")
    )
    error_algorithm_flag = astra.IntParameter(
        default=1,
        config_path=dict(section=task_namespace, name="error_algorithm_flag")
    )
    full_covariance = astra.BoolParameter(
        default=False,
        config_path=dict(section=task_namespace, name="full_covariance")
    )

    continuum_flag = astra.IntParameter(
        default=1,
        config_path=dict(section=task_namespace, name="continuum_flag")
    )
    continuum_order = astra.IntParameter(
        default=4,
        config_path=dict(section=task_namespace, name="continuum_order")
    )
    continuum_reject = astra.FloatParameter(
        default=0.1,
        config_path=dict(section=task_namespace, name="continuum_reject")
    )
    continuum_observations_flag = astra.IntParameter(
        default=1,
        config_path=dict(section=task_namespace, name="continuum_observations_flag")
    )
    
    optimization_algorithm_flag = astra.IntParameter(
        default=3,
        config_path=dict(section=task_namespace, name="optimization_algorithm_flag")
    )
    wavelength_interpolation_flag = astra.IntParameter(
        default=0,
        config_path=dict(section=task_namespace, name="wavelength_interpolation_flag")
    )

    pca_project = astra.BoolParameter(
        default=False,
        config_path=dict(section=task_namespace, name="pca_project")
    )
    pca_chi = astra.BoolParameter(
        default=False,
        config_path=dict(section=task_namespace, name="pca_chi")
    )

    lsf_shape_flag = astra.IntParameter(
        default=0,
        config_path=dict(section=task_namespace, name="lsf_shape_flag")
    )
    use_direct_access = astra.BoolParameter(
        default=False, significant=False,
        config_path=dict(section=task_namespace, name="use_direct_access")
    )
    n_threads = astra.IntParameter(
        default=1, significant=False,
        config_path=dict(section=task_namespace, name="n_threads")
    )

    input_weights_path = astra.OptionalParameter(
        default="",
        config_path=dict(section=task_namespace, name="input_weights_path")
    )
    input_wavelength_mask_path = astra.OptionalParameter(
        default="",
        config_path=dict(section=task_namespace, name="input_wavelength_mask_path")
    )
    input_lsf_path = astra.OptionalParameter(
        default="",
        config_path=dict(section=task_namespace, name="input_lsf_path")
    )
    debug = astra.BoolParameter(
        default=False, significant=False,
        config_path=dict(section=task_namespace, name="debug")
    )

    directory_kwds = astra.DictParameter(
        default=None, significant=False,
        config_path=dict(section=task_namespace, name="directory_kwds")
    )


    # TODO: Put elsewhere?
    speclib_dir = astra.Parameter(
        config_path=dict(section="FERRE", name="speclib_dir")
    )

    ferre_executable = astra.Parameter(
        default="ferre.x",
        config_path=dict(section=task_namespace, name="ferre_executable")
    )

    ferre_kwds = astra.DictParameter(default=None)

    # Optionally disable generating AstraSource objects.
    write_source_output = astra.BoolParameter(default=True, significant=False)

    '''
    use_slurm = astra.BoolParameter(
        default=True, significant=False,
        config_path=dict(section=task_namespace, name="use_slurm")
    )
    slurm_nodes = astra.IntParameter(
        default=1, significant=False,
        config_path=dict(section=task_namespace, name="slurm_nodes")
    )
    slurm_ppn = astra.IntParameter(
        default=64, significant=False,
        config_path=dict(section=task_namespace, name="slurm_ppn")
    )
    slurm_walltime = astra.Parameter(
        default="24:00:00", significant=False,
        config_path=dict(section=task_namespace, name="slurm_walltime")        
    )
    slurm_alloc = astra.Parameter(
        significant=False, default="sdss-np", # The SDSS-V cluster.
        config_path=dict(section=task_namespace, name="slurm_alloc")
    )
    '''

class SourceMixin(BaseTask):

    """ Mixin class for dealing with multiple objects in FERRE. """

    initial_parameters = astra.DictParameter(default=None, batch_method=tuple)
    frozen_parameters = astra.DictParameter(default=None)

