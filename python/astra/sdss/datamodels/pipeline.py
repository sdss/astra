from ast import Name
import os
import numpy as np
from typing import Dict, List, Union
from sdss_access import SDSSPath
from astra import log, __version__ as astra_version
from astra.database.astradb import Task, DataProduct, Source, TaskOutputDataProducts
from astropy.io import fits
from astra.sdss.datamodels.base import (
    create_empty_hdu,
    fits_column_kwargs,
    create_primary_hdu_cards,
    wavelength_cards,
    metadata_cards,
    FILLER_CARD,
    add_glossary_comments,
    _get_extname,
)

from astra.sdss.datamodels.mwm import (
    HDU_DESCRIPTIONS,
    get_data_hdu_observatory_and_instrument,
)
from astra.utils import list_to_dict, expand_path


def create_pipeline_product(
    task: Task,
    data_product: DataProduct,
    results: Dict,
    release: str = "sdss5",
    source: Source = None,
):
    """
    Create a data product containing the best-fitting model parameters and model spectra from an
    analysis pipeline, given some data product.

    :param task:
        The task that analysed the spectra.

    :param data_product:
        The input data product.

    :param results:
        A dictionary of results, where each key should be a string describing the HDU for those
        results (e.g., 'APOGEE/APO', 'APOGEE/LCO', 'BOSS/APO'). The values should be a dictionary
        that will be converted into a binary FITS table.

    :param release: [optional]
        The name of the release to create the data product in. Defaults to 'sdss5'.

    :param source: [optional]
        A database source for the object that the ``data_product`` is associated with. This is
        only used if the data product is not a mwmVisit or mwmStar product, and if the data
        product has no single source associated to it.
    """

    if data_product.filetype in ("mwmVisit", "mwmStar"):
        # Copy primary HDU.
        with fits.open(data_product.path) as image:
            primary_hdu = image[0].copy()
    else:
        # If it's a `full` file then we will have a bad time.
        if source is None:
            try:
                (source,) = data_product.sources
            except:
                log.exception(
                    f"Could not find unique source associated with data product {data_product}"
                )
        cards = create_primary_hdu_cards(source, HDU_DESCRIPTIONS)
        primary_hdu = fits.PrimaryHDU(header=fits.Header(cards))

    # Update comments and checksum.
    for i, comment in enumerate(primary_hdu.header["COMMENT"]):
        if i == 0:
            continue
        prefix, desc = comment.split(": ")
        primary_hdu.header["COMMENT"][i] = f"{prefix}: Model fits to {desc}"
    primary_hdu.add_checksum()

    hdus = [primary_hdu]
    for i, (observatory, instrument) in enumerate(
        get_data_hdu_observatory_and_instrument()
    ):
        key = _get_extname(instrument, observatory)
        if key not in results or len(results[key]) == 0:
            hdus.append(create_empty_hdu(observatory, instrument))
        else:
            hdus.append(
                create_pipeline_hdu(
                    task,
                    data_product,
                    results[key],
                    observatory=observatory,
                    instrument=instrument,
                )
            )
    # TODO: Warn about result keys that don't match any HDU.

    # Parse pipeline name from the task name
    pipeline = task.name.split(".")[2]
    task_id = task.id

    image = fits.HDUList(hdus)
    kwds = dict(
        pipeline=pipeline,
        astra_version=astra_version,
        apred=data_product.kwargs.get("apred", ""),
        run2d=data_product.kwargs.get("run2d", ""),
        # Get catalog identifier from primary HDU. Don't rely on it being in the data product kwargs.
        catalogid=data_product.kwargs.get("catalogid", image[0].header["SDSS_ID"]),
    )

    filetype = "astraVisit" if data_product.filetype == "mwmVisit" else "astraStar"

    try:
        path = SDSSPath(release).full(filetype, **kwds)
    except:
        log.exception(f"Could not create path for {filetype} {kwds}")

        apred, run2d, catalogid = (kwds["apred"], kwds["run2d"], kwds["catalogid"])
        if filetype == "astraVisit":
            path = expand_path(
                f"$MWM_ASTRA/{astra_version}/{run2d}-{apred}/results/visit/{catalogid % 10000:.0f}/{catalogid % 100:.0f}/astraVisit-{astra_version}-{pipeline}-{catalogid}-{task_id}.fits"
            )
        elif filetype == "astraStar":
            path = expand_path(
                f"$MWM_ASTRA/{astra_version}/{run2d}-{apred}/results/star/{catalogid % 10000:.0f}/{catalogid % 100:.0f}/astraStar-{astra_version}-{pipeline}-{catalogid}-{task_id}.fits"
            )

    os.makedirs(os.path.dirname(path), exist_ok=True)
    image.writeto(path, overwrite=True)

    # Create Data product
    output_data_product, created = DataProduct.get_or_create(
        release=release, filetype=filetype, kwargs=kwds
    )
    # Link to this task as an output
    TaskOutputDataProducts.get_or_create(task=task, data_product=output_data_product)
    log.info(f"Created data product {output_data_product} and wrote to {path}")

    return output_data_product


def create_pipeline_hdu(
    task: Task,
    data_product: DataProduct,
    results: Dict,
    observatory: str,
    instrument: str,
):
    """
    Create a header data unit (HDU) containing the best-fitting model parameters and model spectra
    from an analysis pipeline, given some data product.

    :param task:
        The task that analysed the spectra.

    :param data_product:
        The input data product.
    """

    # Only special keyword is the wavelength array
    spectral_axis = results.pop("spectral_axis", None)
    if spectral_axis is not None:
        # Put in header as CDELT, etc.
        None

    columns = []
    for name, values in results.items():
        values = np.atleast_1d(np.array(values))
        columns.append(
            fits.Column(
                name=name, array=values, unit=None, **fits_column_kwargs(values)
            )
        )

    header = fits.Header(
        cards=[
            *metadata_cards(observatory, instrument),
            # *wavelength_cards(),
            FILLER_CARD,
        ]
    )

    hdu = fits.BinTableHDU.from_columns(
        columns, name=f"{instrument}/{observatory}", header=header
    )
    add_glossary_comments(hdu)

    return hdu
