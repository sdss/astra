import os
import luigi
import types

from sdss_access import SDSSPath

from astra.tasks.base import BaseTask


# $APOGEE_REDUX/{apred}/visit/{telescope}/{field}/{plate}/{mjd}/{prefix}Visit-{apred}-{plate}-{mjd}-{fiber:0>3}.fits
# mastargoodspec: $MANGA_SPECTRO_MASTAR/{drpver}/{mastarver}/mastar-goodspec-{drpver}.fits.gz

path = SDSSPath()


class BaseDataModelTask(BaseTask):

    def output(self):
        return luigi.LocalTarget(
            path.full(self.sdss_datamodel, **self.param_kwargs)
        )    



class ApVisitFile(BaseDataModelTask):
    sdss_datamodel = "apVisit"        

    
class ApStarFile(BaseDataModelTask):
    sdss_datamodel = "apStar"
    

    def writer(self, spectrum, path, **kwargs):
        from astra.tools.spectrum.loaders import write_sdss_apstar
        return write_sdss_apstar(spectrum, path, **kwargs)



for klass in (ApVisitFile, ApStarFile):
    for lookup_key in path.lookup_keys(klass.sdss_datamodel):
        setattr(klass, lookup_key, luigi.Parameter())




def create_external_task(
        task_name,
        path_name,
        release=None, 
        public=None, 
        verbose=False, 
        force_modules=None
    ):

    path = SDSSPath(
        release=release,
        public=public,
        verbose=verbose,
        force_modules=force_modules
    )

    # Get lookup keys as parameters.
    parameter_names = path.lookup_keys(path_name)
    '''
    types.new_class(
        f"Base{task_name}Task",
        luigi.Task,
        
    )
    '''
    # See https://stackoverflow.com/questions/39217180/how-to-dynamically-create-a-luigi-task
    raise NotImplementedError("dynamic classes not implemented yet")




