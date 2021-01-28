
import astra
import datetime
import numpy as np
from astra.tasks import BaseTask


class FerreMixin(BaseTask):

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
        default=128, significant=False,
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
    ferre_kwds = astra.DictParameter(
        default=None,
        config_path=dict(section=task_namespace, name="ferre_kwds")
    )

    # Optionally disable generating AstraSource objects.
    write_source_output = astra.BoolParameter(default=True, significant=False)
    
    debug = astra.BoolParameter(
        default=False, significant=False,
        config_path=dict(section=task_namespace, name="debug")
    )
    use_slurm = astra.BoolParameter(
        default=True, significant=False,
        config_path=dict(section=task_namespace, name="use_slurm")
    )
    slurm_nodes = astra.IntParameter(
        default=1, significant=False,
        config_path=dict(section=task_namespace, name="slurm_nodes")
    )
    slurm_ppn = astra.IntParameter(
        default=8, significant=False,
        config_path=dict(section=task_namespace, name="slurm_ppn")
    )
    slurm_walltime = astra.Parameter(
        default="01:00:00", significant=False,
        config_path=dict(section=task_namespace, name="slurm_walltime")        
    )
    slurm_alloc = astra.Parameter(
        significant=False, default="sdss-np", # The SDSS-V cluster.
        config_path=dict(section=task_namespace, name="slurm_alloc")
    )
    slurm_partition = astra.Parameter(
        significant=False, default="sdss-np",
        config_path=dict(section=task_namespace, name="slurm_partition")
    )
    slurm_mem = astra.IntParameter(
        significant=False, default=64000,
        config_path=dict(section=task_namespace, name="slurm_mem")
    )
    slurm_gres = astra.Parameter(
        significant=False, default="",
        config_path=dict(section=task_namespace, name="slurm_gres")
    )


class SourceMixin(BaseTask):

    """ Mixin class for dealing with multiple objects in FERRE. """

    # It may not be a great idea hard-coding these parameter names because FERRE
    # can -- in principle -- be run on grids with any set of parameter names.
    # But in practice these parameter names have not changed in 10 years, and there
    # are no forseeable plans to change them.

    # Encoding them like this (instead of a dictionary) means we can better see performance
    # with respect to frozen and initial parameters.

    initial_teff = astra.FloatParameter(batch_method=tuple)
    initial_logg = astra.FloatParameter(batch_method=tuple)
    initial_metals = astra.FloatParameter(batch_method=tuple)
    initial_log10vdop = astra.FloatParameter(batch_method=tuple)
    initial_o_mg_si_s_ca_ti = astra.FloatParameter(batch_method=tuple)
    initial_lgvsini = astra.FloatParameter(batch_method=tuple)
    initial_c = astra.FloatParameter(batch_method=tuple)
    initial_n = astra.FloatParameter(batch_method=tuple)

    # If a parameter is frozen then it is set through the INDINI/INDIV flags, and we
    # cannot change this behaviour on a per-object basis.
    frozen_teff = astra.BoolParameter(default=False)
    frozen_logg = astra.BoolParameter(default=False)
    frozen_metals = astra.BoolParameter(default=False)
    frozen_log10vdop = astra.BoolParameter(default=False)
    frozen_o_mg_si_s_ca_ti = astra.BoolParameter(default=False)
    frozen_lgvsini = astra.BoolParameter(default=False)
    frozen_c = astra.BoolParameter(default=False)
    frozen_n = astra.BoolParameter(default=False)
