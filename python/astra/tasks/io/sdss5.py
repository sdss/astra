
import astra
from astra.tasks.io.base import SDSSDataModelTask


class SDSS5DataModelTask(SDSSDataModelTask):

    """ A task to represent a SDSS-V data product. """

    release = astra.Parameter(default="sdss5")

    # By default, SDSS-V data that we are processing is not public!
    public = astra.BoolParameter(
        default=False,
        significant=False,
        parsing=astra.BoolParameter.IMPLICIT_PARSING
    )
    use_remote = astra.BoolParameter(
        default=True,
        significant=False,
        parsing=astra.BoolParameter.IMPLICIT_PARSING
    )
    

class ApVisitFile(SDSS5DataModelTask):

    """
    A task to represent a SDSS-V ApVisit data product.

    :param fiber:
        The fiber number that the object was observed with.

    :param plate:
        The plate identifier.

    :param telescope:
        The name of the telescope used to observe the object (e.g., apo25m).

    :param field:
        The field the object was observed in.

    :param mjd:
        The Modified Julian Date of the observation.

    :param apred:
        The data reduction version number (e.g., daily).
        
    :param release:
        The name of the SDSS data release (e.g., sdss5).
    """    

    sdss_data_model_name = "apVisit"

    fiber = astra.IntParameter(batch_method=tuple)
    plate = astra.Parameter(batch_method=tuple)
    field = astra.Parameter(batch_method=tuple)
    mjd = astra.IntParameter(
        batch_method=tuple,
        description="The Modified Julian Date of the observation."
    )
    apred = astra.Parameter(batch_method=tuple)
    telescope = astra.Parameter(batch_method=tuple)

    
class ApStarFile(SDSS5DataModelTask):

    """
    A task to represent a SDSS-V ApStar data product.

    :param obj:
        The name of the object.

    :param healpix:
        The healpix identifier based on the object location.

    :param telescope:
        The name of the telescope used to observe the  object (e.g., apo25m).

    :param apstar:
        A string indicating the kind of object (usually 'star').
    
    :param apred:
        The data reduction version number (e.g., daily).

    :param release:
        The name of the SDSS data release (e.g., sdss5).
    """

    sdss_data_model_name = "apStar"
    
    obj = astra.Parameter(batch_method=tuple)
    healpix = astra.IntParameter(batch_method=tuple)
    # TODO: Consider whether apstar is needed, or if it should be a batch parameter
    apstar = astra.Parameter(default="star", batch_method=tuple)
    apred = astra.Parameter(batch_method=tuple) 
    telescope = astra.Parameter(batch_method=tuple)


    def writer(self, spectrum, path, **kwargs):
        from astra.tools.spectrum.loaders import write_sdss_apstar
        return write_sdss_apstar(spectrum, path, **kwargs)


class SpecFile(SDSS5DataModelTask):

    """
    A task to represent a SDSS-V BHM Spec data product.

    :param mjd:
        The modified Julian date of observations.

    :param fiberid:
        The fiber number used for observations.

    :param plateid:
        The identifier of the plate used for observations.

    :param run2d:
        The version of the data reduction pipeline used.

    :param release:
        The name of the SDSS data release (e.g., sdss5).
    """

    sdss_data_model_name = "spec"

    mjd = astra.IntParameter(batch_method=tuple)
    fiberid = astra.IntParameter(batch_method=tuple)
    plateid = astra.IntParameter(batch_method=tuple)
    run2d = astra.Parameter(batch_method=tuple)



"""
# Add requisite parameters and make them batchable.
# (Better to explicitly state these above.)
for klass in (ApVisitFile, ApStarFile):
    for lookup_key in klass(release="sdss5").tree.lookup_keys(klass.sdss_data_model_name):
        setattr(klass, lookup_key, astra.Parameter(batch_method=tuple))
"""


