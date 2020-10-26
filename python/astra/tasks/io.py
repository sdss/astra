import os
import luigi
import types
from pathlib import Path

from sdss_access import SDSSPath, RsyncAccess, HttpAccess

from astra.tasks.base import BaseTask, SDSSDataProduct




class LocalTargetTask(BaseTask):
    path = luigi.Parameter()
    def output(self):
        return luigi.LocalTarget(self.path)
    


# $APOGEE_REDUX/{apred}/visit/{telescope}/{field}/{plate}/{mjd}/{prefix}Visit-{apred}-{plate}-{mjd}-{fiber:0>3}.fits
# mastargoodspec: $MANGA_SPECTRO_MASTAR/{drpver}/{mastarver}/mastar-goodspec-{drpver}.fits.gz

class SDSSDataModelTask(BaseTask):

    """ A task to represent a SDSS data product. """
     
    release = luigi.Parameter()
    
    # Parameters for retrieving remote paths.
    use_remote = luigi.BoolParameter(
        default=False,
        significant=False,
        parsing=luigi.BoolParameter.IMPLICIT_PARSING
    )
    remote_access_method = luigi.Parameter(
        default="http",
        significant=False
    )
    public = luigi.BoolParameter(
        default=True,
        significant=False,
        parsing=luigi.BoolParameter.IMPLICIT_PARSING
    )
    mirror = luigi.BoolParameter(
        default=False,
        significant=False,
        parsing=luigi.BoolParameter.IMPLICIT_PARSING
    )
    verbose = luigi.BoolParameter(
        default=True,
        significant=False,
        parsing=luigi.BoolParameter.IMPLICIT_PARSING
    )

    def __init__(self, *args, **kwargs):
        super(SDSSDataModelTask, self).__init__(*args, **kwargs)
        return None


    @property
    def tree(self):
        try:
            return self._tree

        except AttributeError:
            self._tree = SDSSPath(
                release=self.release,
                public=self.public,
                mirror=self.mirror,
                verbose=self.verbose
            )

        return self._tree
        

    @property
    def local_path(self):
        """ The local path of the file. """
        try:
            return self._local_path
            
        except AttributeError:
            self._local_path = self.tree.full(self.sdss_data_model_name, **self.param_kwargs)
            
        return self._local_path
        

    @classmethod
    def get_local_path(cls, release, public=True, mirror=False, verbose=True, **kwargs):
        tree = SDSSPath(
            release=release,
            public=public,
            mirror=mirror,
            verbose=verbose
        )
        return tree.full(cls.sdss_data_model_name, **kwargs)


    def output(self):
        if self.is_batch_mode:
            return [task.output() for task in self.get_batch_tasks()]
        else: 
            if self.use_remote:
                if (os.path.exists(self.local_path) and Path(self.local_path).stat().st_size < 1):
                    # Corrupted. Zero file.
                    os.unlink(self.local_path)

                if not os.path.exists(self.local_path):
                    self.get_remote()
        
            return SDSSDataProduct(self.local_path)


    def get_remote_http(self):
        """ Download the remote file using HTTP. """

        http = HttpAccess(
            verbose=self.verbose,
            public=self.public,
            release=self.release
        )
        http.remote()
        return http.get(self.sdss_data_model_name, **{
                k: getattr(self, k) for k in self.tree.lookup_keys(self.sdss_data_model_name)
            }
        )


    def get_remote_rsync(self):
        """ Download the remote file using rsync. """

        rsync = RsyncAccess(
            mirror=mirror,
            public=public,
            release=release,
            verbose=verbose
        )
        rsync.remote()
        rsync.add(
            self.sdss_data_model_name, **{
                k: getattr(self, k) for k in self.tree.lookup_keys(self.sdss_data_model_name)
            }
        )
        rsync.set_stream()
        return rsync.commit()
        

    def get_remote(self):
        """ Download the remote file. """

        funcs = {
            "http": self.get_remote_http,
            "rsync": self.get_remote_rsync
        }
        try:
            f = funcs[self.remote_access_method.lower()]

        except KeyError:
            raise ValueError(
                f"Unrecognized remote access method '{self.remote_access_method}'. "
                f"Available: {', '.join(list(funcs.keys()))}"
            )
        
        else:
            return f()
        

class SDSS5DataModelTask(SDSSDataModelTask):

    """ A task to represent a SDSS-V data product. """

    release = luigi.Parameter(default="sdss5")


class SDSS4DataModelTask(SDSSDataModelTask):

    """ A task to represent a SDSS-IV data product. """

    release = luigi.Parameter(default="DR16")


#class DSS4DataModelTask):#, ApPlanFile)
#    sdss_data_model_name = "apPlan"


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


# Add requisite parameters and make them batchable.
for klass in (ApVisitFile, ApStarFile, AllStarFile, SpecFile):#, ApPlanFile)
    for lookup_key in klass().tree.lookup_keys(klass.sdss_data_model_name):
        setattr(klass, lookup_key, luigi.Parameter(batch_method=tuple))