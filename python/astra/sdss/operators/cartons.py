from airflow.models.baseoperator import BaseOperator
from airflow.exceptions import AirflowSkipException
from astra.database.astradb import DataProduct, Source, SourceDataProduct
from astra import log
from astra.utils import flatten


class CartonOperator(BaseOperator):
    """
    Filter upstream data model products and only keep those that are matched to a specific SDSS-V carton.

    This operator requires a data model operator directly preceeding it in a DAG (e.g., an ApStarOperator).
    """

    ui_color = "#FEA83A"

    def __init__(
        self,
        *,
        cartons=None,
        programs=None,
        mappers=None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.cartons = cartons
        self.programs = programs
        self.mappers = mappers

    def execute(
        self,
        context,
    ):
        # It's bad practice to import here, but we don't want to depend on catalogdb to be accessible from
        # the outer scope of this operator.
        from astra.sdss.catalog import filter_sources

        ti, task = (context["ti"], context["task"])
        data_product_ids = tuple(
            set(flatten(ti.xcom_pull(task_ids=task.upstream_task_ids)))
        )

        log.info(f"Data product IDs ({len(data_product_ids)}): {data_product_ids}")

        log.info(f"Matching on:")
        log.info(f"     Cartons: {self.cartons}")
        log.info(f"     Programs: {self.programs}")
        log.info(f"     Mappers: {self.mappers}")

        # Retrieve the source catalog identifiers for these data products.
        q = (
            Source.select(Source.catalogid, DataProduct.id)
            .join(SourceDataProduct)
            .join(DataProduct)
            .where(DataProduct.id.in_(data_product_ids))
            .tuples()
        )
        lookup = {c_id: dp_id for c_id, dp_id in q}
        log.info(f"Lookup table contains {len(lookup)} matched entries.")

        # Only return the data product ids that match.
        keep = filter_sources(
            tuple(lookup.keys()),
            cartons=self.cartons,
            programs=self.programs,
            mappers=self.mappers,
        )
        log.info(f"Keeping {len(keep)} matched.")

        keep_data_product_ids = [lookup[c_id] for c_id in keep]
        if len(keep_data_product_ids) == 0:
            raise AirflowSkipException(
                f"None of the sources of upstream data products matched."
            )

        log.info(f"{keep_data_product_ids}")
        # Return the associated data product identifiers from the lookup.
        return keep_data_product_ids
