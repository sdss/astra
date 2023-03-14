from airflow.models.baseoperator import BaseOperator
from functools import cached_property


class SDSSOperator(BaseOperator):

    """A base operator to retrieve and ingest SDSS data products."""

    pass