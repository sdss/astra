import astra
from astra.tasks.io import SDSSDataModelTask
from astra.database import session, astradb


class SDSS4DataModelTask(SDSSDataModelTask):

    """ A task to represent a SDSS-IV data product. """

    release = astra.Parameter(default="DR16")


class SDSS4ApVisitFile(SDSS4DataModelTask):
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

    
class SDSS4ApStarFile(SDSS4DataModelTask):

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


    def get_or_create_data_model_relationships(self):
        """ Return the keywords that reference the input data model for this task. """
        
        keys = ("release", "apstar", "apred", "telescope", "field", "obj")
        kwds = { k: getattr(self, k) for k in keys }

        q = session.query(astradb.SDSS4ApogeeStar).filter_by(**kwds)
        instance = q.one_or_none()
        if instance is None:
            instance = astradb.SDSS4ApogeeStar(**kwds)
            with session.begin():
                session.add(instance)
        
        return { "sdss4_apogee_star_pk": (instance.pk, ) }


    @classmethod
    def get_data_model_keywords(self, task_state):
        keys = ("release", "apstar", "apred", "telescope", "field", "obj")

        pk = task_state.sdss4_apogee_star_pk

        q = session.query(astradb.SDSS4ApogeeStar).filter_by(pk=pk[0])
        instance = q.one_or_none()
        if instance is None:
            raise ValueError(f"no astradb.SDSS4ApogeeStar row found with pk = {pk}")
        
        return { k: getattr(instance, k) for k in keys }
        



    def writer(self, spectrum, path, **kwargs):
        from astra.tools.spectrum.loaders import write_sdss_apstar
        return write_sdss_apstar(spectrum, path, **kwargs)



class SDSS4SpecFile(SDSS4DataModelTask):

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


class SDSS4AllStarFile(SDSS4DataModelTask):
    
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


class SDSS4AllVisitSum(SDSS4DataModelTask):

    """
    A task to represent an AllVisitSum SDSS/APOGEE data product.

    :param apred:
        The version of the APOGEE reduction pipeline used.
    
    :param release:
        The name of the SDSS data release (e.g., DR16).
    """

    sdss_data_model_name = "allVisitSum"


# Add requisite parameters and make them batchable.
for klass in (SDSS4ApVisitFile, SDSS4ApStarFile, SDSS4AllStarFile, SDSS4AllVisitSum, SDSS4SpecFile):#, ApPlanFile)
    for lookup_key in klass().tree.lookup_keys(klass.sdss_data_model_name):
        setattr(klass, lookup_key, astra.Parameter(batch_method=tuple))
