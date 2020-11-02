import astra
from astra.tasks.io.base import SDSSDataModelTask


class SDSS4DataModelTask(SDSSDataModelTask):

    """ A task to represent a SDSS-IV data product. """

    release = astra.Parameter(default="DR16")


class ApVisitFile(SDSS4DataModelTask):
    """
    A task to represent an ApVisit SDSS/APOGEE data product.

    :param fiber:
        The fiber number that the object was observed with.

    :param plate:
        The plate identifier.

    :param field:
        The field the object was observed in.

    :param mjd:
        The modified Julian date of the observation.

    :param apred:
        The ASPCAP reduction version number (e.g., r12).
        
    :param prefix:
        The prefix of the filename (e.g., ap or as).    

    :param release:
        The name of the SDSS data release (e.g., DR16).
    """    

    sdss_data_model_name = "apVisit"

    
class ApStarFile(SDSS4DataModelTask):

    """
    A task to represent an ApStar SDSS/APOGEE data product.

    :param obj:
        The name of the object.

    :param field:
        The field the object was observed in.

    :param telescope:
        The name of the telescope used to observe the  object (e.g., apo25m).

    :param apred:
        The ASPCAP reduction version number (e.g., r12).

    :param apstar:
        A string indicating the kind of object (usually 'star').

    :param release:
        The name of the SDSS data release (e.g., DR16).
    """

    sdss_data_model_name = "apStar"
    
    def writer(self, spectrum, path, **kwargs):
        from astra.tools.spectrum.loaders import write_sdss_apstar
        return write_sdss_apstar(spectrum, path, **kwargs)


class SpecFile(SDSS4DataModelTask):

    """
    A task to represent a Spec SDSS/BOSS data product.

    :param fiberid:
        The fiber number that the object was observed with.

    :param plateid:
        The plate identifier.
    
    :param mjd:
        The modified Julian date of the observation.

    :param run2d:
        The version of the BOSS reduction pipeline used.

    :param release:
        The name of the SDSS data release (e.g., DR16).
    """
    
    sdss_data_model_name = "spec"


class AllStarFile(SDSS4DataModelTask):
    
    """
    A task to represent an AllStar SDSS/APOGEE data product.

    :param aspcap:
        The version of the ASPCAP analysis pipeline used.

    :param apred:
        The version of the APOGEE reduction pipeline used.

    :param release:
        The name of the SDSS data release (e.g., DR16).
    """

    sdss_data_model_name = "allStar"


class AllVisitSum(SDSS4DataModelTask):

    """
    A task to represent an AllVisitSum SDSS/APOGEE data product.

    :param apred:
        The version of the APOGEE reduction pipeline used.
    
    :param release:
        The name of the SDSS data release (e.g., DR16).
    """

    sdss_data_model_name = "allVisitSum"


# Add requisite parameters and make them batchable.
for klass in (ApVisitFile, ApStarFile, AllStarFile, AllVisitSum, SpecFile):#, ApPlanFile)
    for lookup_key in klass().tree.lookup_keys(klass.sdss_data_model_name):
        setattr(klass, lookup_key, astra.Parameter(batch_method=tuple))
