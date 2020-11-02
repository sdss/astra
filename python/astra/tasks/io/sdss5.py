
import astra
from astra.tasks.io.base import SDSSDataModelTask


class SDSS5DataModelTask(SDSSDataModelTask):

    """ A task to represent a SDSS-V data product. """

    release = astra.Parameter(default="sdss5")
    telescope = astra.Parameter()


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

    fiber = astra.IntParameter()
    plate = astra.Parameter()
    field = astra.Parameter()
    mjd = astra.IntParameter()
    apred = astra.Parameter()

    
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
        The ASPCAP reduction version number (e.g., r12).

    :param release:
        The name of the SDSS data release (e.g., sdss5).
    """

    sdss_data_model_name = "apStar"
    
    obj = astra.Parameter()
    healpix = astra.IntParameter()
    apstar = astra.Parameter(default="star")
    apred = astra.Parameter()    

    def writer(self, spectrum, path, **kwargs):
        from astra.tools.spectrum.loaders import write_sdss_apstar
        return write_sdss_apstar(spectrum, path, **kwargs)


"""
# Add requisite parameters and make them batchable.
# (Better to explicitly state these above.)
for klass in (ApVisitFile, ApStarFile):
    for lookup_key in klass(release="sdss5").tree.lookup_keys(klass.sdss_data_model_name):
        setattr(klass, lookup_key, astra.Parameter(batch_method=tuple))
"""