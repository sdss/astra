import os
from airflow.models.baseoperator import BaseOperator
from sdss_access import SDSSPath
from astra.database.astradb import (
    database,
    Task,
    Bundle,
    TaskBundle,
    TaskInputDataProducts,
    Source,
    Output,
    MWMSourceStatus,
    DataProduct,
    SourceDataProduct,
    TaskOutput,
    TaskOutputDataProducts,
)
from astra.base import TaskInstance, Parameter
from astra.utils import flatten, expand_path
from astra import log, __version__ as astra_version

from typing import List, Tuple, Dict, Union, Optional

from astra.sdss.datamodels.mwm import create_mwm_data_products


class CreateMWMVisitStarProducts(TaskInstance):

    catalogid = Parameter()
    release = Parameter()
    run2d = Parameter()
    apred = Parameter()

    def execute(self):

        for task, data_products, parameters in self.iterable():

            catalogid = parameters["catalogid"]
            hdu_visit_list, hdu_star_list, meta = create_mwm_data_products(
                catalogid, input_data_products=data_products
            )
            # Get healpix.
            healpix = int(hdu_visit_list[0].header["HEALPIX"])

            kwds = dict(
                catalogid=catalogid,
                astra_version=astra_version,
                run2d=self.run2d,
                apred=self.apred,
            )
            # Write to disk.
            p = SDSSPath(self.release)
            mwmVisit_path = p.full("mwmVisit", **kwds)
            mwmStar_path = p.full("mwmStar", **kwds)
            # Create necessary folders
            for path in (mwmVisit_path, mwmStar_path):
                os.makedirs(os.path.dirname(path), exist_ok=True)

            # Ensure mwmVisit and mwmStar files are always synchronised.
            try:
                hdu_visit_list.writeto(mwmVisit_path, overwrite=True)
                hdu_star_list.writeto(mwmStar_path, overwrite=True)
            except:
                log.exception(
                    f"Exception when trying to write to either:\n{mwmVisit_path}\n{mwmStar_path}"
                )
                # Delete both.
                for path in (mwmVisit_path, mwmStar_path):
                    if os.path.exists(path):
                        os.unlink(path)
            else:
                log.info(f"Wrote mwmVisits product to {mwmVisit_path}")
                log.info(f"Wrote mwmStar product to {mwmStar_path}")

                # Create output data product records that link to this task.
                dp_visit, visit_created = DataProduct.get_or_create(
                    release=self.release, filetype="mwmVisit", kwargs=kwds
                )
                TaskOutputDataProducts.get_or_create(task=task, data_product=dp_visit)
                SourceDataProduct.get_or_create(
                    data_product=dp_visit, source_id=catalogid
                )

                dp_star, star_created = DataProduct.get_or_create(
                    release=self.release, filetype="mwmStar", kwargs=kwds
                )
                TaskOutputDataProducts.get_or_create(task=task, data_product=dp_star)
                SourceDataProduct.get_or_create(
                    data_product=dp_star, source_id=catalogid
                )

            # Get or create an output record.
            task.create_or_update_outputs(MWMSourceStatus, [meta])

            log.info(
                f"Created data products {dp_visit} and {dp_star} for catalogid {catalogid}"
            )

        return None

    def post_execute(self):
        """Generate JSON files for the spectra inspecta."""
        # TODO
        return None


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
                    name="astra.sdss.datamodels.operators.CreateMWMVisitStarProducts",
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
