
import astra
import datetime
from astra.tasks.base import BaseTask
from astra.contrib.ferre.tasks import ApStarFile

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
    ferre_version_major = astra.IntParameter(default=4)
    ferre_version_minor = astra.IntParameter(default=8)
    ferre_version_patch = astra.IntParameter(default=5)

    # Instead of providing the grid header path here, we will pass it through other parameters
    # that can be used to reconstruct the grid header path.
    interpolation_order = astra.IntParameter(default=3)
    init_algorithm_flag = astra.IntParameter(default=1)
    error_algorithm_flag = astra.IntParameter(default=1)
    full_covariance = astra.BoolParameter(default=False)

    continuum_flag = astra.IntParameter(default=1)
    continuum_order = astra.IntParameter(default=4)
    continuum_reject = astra.FloatParameter(default=0.1)
    continuum_observations_flag = astra.IntParameter(default=1)
    
    optimization_algorithm_flag = astra.IntParameter(default=3)
    wavelength_interpolation_flag = astra.IntParameter(default=0)

    pca_project = astra.BoolParameter(default=False)
    pca_chi = astra.BoolParameter(default=False)

    lsf_shape_flag = astra.IntParameter(default=0)
    use_direct_access = astra.BoolParameter(default=True, significant=False)
    n_threads = astra.IntParameter(default=1, significant=False)

    input_weights_path = astra.OptionalParameter(default="")
    input_lsf_path = astra.OptionalParameter(default="")
    debug = astra.BoolParameter(default=False, significant=False)

    directory_kwds = astra.DictParameter(default=None, significant=False)


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



class ApStarMixin(ApStarFile):

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