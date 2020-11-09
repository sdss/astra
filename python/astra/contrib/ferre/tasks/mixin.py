
import astra
import datetime
from astra.tasks.base import BaseTask

# TODO: Is this in SDSS access? Or if it's just an environment variable?!
SPECLIB_DIR = "/home/andy/data/sdss/apogeework/apogee/spectro/speclib/"


class BaseDispatcherMixin(BaseTask):

    """ A mixin class for dispatching FERRE tasks. """

    # Grid path parameters.
    radiative_transfer_code = astra.Parameter(default="*")
    model_photospheres = astra.Parameter(default="*")
    isotopes = astra.Parameter(default="*")
    grid_creation_date = astra.DateParameter(default=datetime.date(2018, 9, 1))

    gd = astra.Parameter(default="*")
    spectral_type = astra.Parameter(default="*")
    lsf = astra.Parameter(default="*")
    aspcap = astra.Parameter(default="*")
    
    # Task Factory.
    task_factory = astra.TaskParameter()


class BaseFerreMixin(BaseTask):

    """ A mixin class for FERRE tasks. """

    task_namespace = "FERRE"

    # Current FERRE version at time of writing is 4.8.5
    ferre_version_major = astra.IntParameter(
        default=4,
        config_path=dict(section=task_namespace, name="ferre_version_major")
    )
    ferre_version_minor = astra.IntParameter(
        default=8,
        config_path=dict(section=task_namespace, name="ferre_version_minor")
    )
    ferre_version_patch = astra.IntParameter(
        default=5,
        config_path=dict(section=task_namespace, name="ferre_version_patch")
    )

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
        default=True, significant=False,
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

    use_queue = astra.BoolParameter(
        default=False, significant=False,
        config_path=dict(section=task_namespace, name="use_queue")
    )
    queue_kwds = astra.DictParameter(
        default=None, significant=False,
        config_path=dict(section=task_namespace, name="queue_kwds")
    )
    

class GridHeaderFileMixin(BaseTask):

    """ A mixin class for grid header files. """

    # Use parameters to find the grid path, not provide the grid path itself.
    radiative_transfer_code = astra.Parameter()
    model_photospheres = astra.Parameter()
    isotopes = astra.Parameter()
    gd = astra.Parameter()
    spectral_type = astra.Parameter()
    grid_creation_date = astra.DateParameter(default=datetime.date(2018, 9, 1)) # TODO
    lsf = astra.Parameter()
    aspcap = astra.Parameter()



class ApStarMixin(BaseTask):

    """ Mixin class for dealing with ApStarFile objects in FERRE. """

    # Initial parameters can vary between objects, but frozen parameters cannot.
    #initial_parameters = astra.ListParameter(default=None, batch_method=lambda _: tuple((_, ))[0])
    initial_parameters = astra.DictParameter(default=None, batch_method=tuple)
    frozen_parameters = astra.DictParameter(default=None)


class FerreMixin(BaseFerreMixin, ApStarMixin, GridHeaderFileMixin):
    """ Mixin class for running FERRE on ApStar spectra. """
    pass

class DispatcherMixin(BaseDispatcherMixin, ApStarMixin, BaseFerreMixin):
    pass