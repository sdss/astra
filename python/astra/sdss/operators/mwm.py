from airflow.models.baseoperator import BaseOperator
from astra.database.astradb import (
    database,
    Task,
    Bundle,
    TaskBundle,
    TaskInputDataProducts,
    Source,
    DataProduct,
)
from astra.utils import flatten
from astra import log, __version__ as astra_version

from typing import Optional


class MWMVisitStarFactory(BaseOperator):

    ui_color = "#A0B9D9"

    def __init__(
        self,
        *,
        product_release: Optional[str] = None,
        apred_release: Optional[str] = None,
        apred: Optional[str] = None,
        run2d_release: Optional[str] = None,
        run2d: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.product_release = product_release
        self.apred = apred
        self.apred_release = apred_release
        self.run2d = run2d
        self.run2d_release = run2d_release
        return None

    def execute(self, context):

        ti, task = (context["ti"], context["task"])

        bundle = Bundle.create()
        log.info(f"Created task bundle {bundle}")

        catalogids = map(
            int, tuple(set(flatten(ti.xcom_pull(task_ids=task.upstream_task_ids))))
        )
        for catalogid in catalogids:

            log.info(
                f"Creating mwmVisit and mwmStar products for catalogid={catalogid}"
            )

            expression = DataProduct.filetype.in_(("apVisit", "specFull"))

            if self.apred_release is not None and self.apred is not None:
                sub = DataProduct.filetype == "apVisit"
                if self.apred_release is not None:
                    sub &= DataProduct.release == self.apred_release
                if self.apred is not None:
                    sub &= DataProduct.kwargs["apred"] == self.apred

                expression |= sub

            if self.run2d_release is not None and self.run2d is not None:
                sub = DataProduct.filetype == "specFull"
                if self.run2d_release is not None:
                    sub &= DataProduct.release == self.run2d_release
                if self.run2d is not None:
                    sub &= DataProduct.kwargs["run2d"] == self.run2d

                expression |= sub

            input_data_products = tuple(
                Source.get(catalogid).data_products.where(expression)
            )

            # Create a task
            with database.atomic():
                created_task = Task.create(
                    name="astra.sdss.datamodels.mwm.CreateMWMVisitStarProducts",
                    parameters=dict(
                        release=self.product_release,
                        catalogid=catalogid,
                        run2d=self.run2d,
                        apred=self.apred,
                    ),
                    version=astra_version,
                )

                for data_product in input_data_products:
                    TaskInputDataProducts.create(
                        task=created_task, data_product=data_product
                    )

                TaskBundle.create(task=created_task, bundle=bundle)
            log.info(f"Created task {created_task}")

        log.info(f"Final task bundle {bundle} now ready.")

        return bundle.id
