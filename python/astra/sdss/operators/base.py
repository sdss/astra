from airflow.models.baseoperator import BaseOperator
from sdss_access import SDSSPath
from functools import cached_property


class SDSSOperator(BaseOperator):

    """A base operator to retrieve and ingest SDSS data products."""

    @cached_property
    def lookup_keys(self):
        return self.path_instance.lookup_keys(self.filetype)

    @cached_property
    def path_instance(self):
        return SDSSPath(release=self.release)

    