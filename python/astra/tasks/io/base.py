import os
import luigi

from pathlib import Path

from sdss_access import SDSSPath, RsyncAccess, HttpAccess

from astra.tasks.base import BaseTask, SDSSDataProduct


class LocalTargetTask(BaseTask):
    path = luigi.Parameter()
    def output(self):
        return luigi.LocalTarget(self.path)

    
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


    @property
    def remote_path(self):
        """ 
        The remote path of the file. Useful for debugging path problems.

        This is relatively expensive to return, so don't use this to download sources.
        Instead use one instance of sdss_access.HttpAccess to get the remote paths of
        many sources.
        """
        http = HttpAccess(
            verbose=self.verbose,
            public=self.public,
            release=self.release
        )
        http.remote()
        return http.url(self.sdss_data_model_name, **{
                k: getattr(self, k) for k in self.tree.lookup_keys(self.sdss_data_model_name)
            }
        )


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
            mirror=self.mirror,
            public=self.public,
            release=self.release,
            verbose=self.verbose
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
        