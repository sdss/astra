import os
import numpy as np
from typing import Dict, List, Union
from sdss_access import SDSSPath
from astra import log, __version__ as astra_version
from astra.database.astradb import Task, DataProduct, TaskOutputDataProducts
from astropy.io import fits
from astra.sdss.datamodels.base import (
    create_empty_hdu,
    fits_column_kwargs,
    create_primary_hdu_cards,
    wavelength_cards,
    metadata_cards,
    FILLER_CARD,
    add_glossary_comments,
)

from astra.sdss.datamodels.mwm import (
    HDU_DESCRIPTIONS,
    get_data_hdu_observatory_and_instrument,
)
from astra.utils import list_to_dict, expand_path


def create_pipeline_product(
    task: Task,
    data_product: DataProduct,
    results: List[List[Dict]],
    pipeline,
    release="sdss5",
):
    """
    Create a data product containing the best-fitting model parameters and model spectra from an
    analysis pipeline, given some data product.

    :param task:
        The task that analysed the spectra.

    :param data_product:
        The input data product.

    :param results:
        A list of list of dicts, where `len(results)` should equal the number of HDUs in the given
        data product. Each element of `results` should be a list with length equal to the number
        of spectra in that HDU, and each element (sub-element of `results`) should be a dictionary
        that contains ..... #TODO
    """

    if data_product.filetype in ("mwmVisit", "mwmStar"):
        # Copy primary HDU.
        with fits.open(data_product.path) as image:
            primary_hdu = image[0].copy()
    else:
        # If it's a `full` file then we will have a bad time.
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

    observatory_instrument = get_data_hdu_observatory_and_instrument()

    hdus = [primary_hdu]
    for i, hdu_results in enumerate(results):
        observatory, instrument = observatory_instrument[i]
        if hdu_results is None or len(hdu_results) == 0:
            # blank frame
            hdus.append(create_empty_hdu(observatory, instrument))
        else:
            hdus.append(
                create_pipeline_hdu(
                    task,
                    data_product,
                    hdu_results,
                    observatory=observatory,
                    instrument=instrument,
                )
            )

    image = fits.HDUList(hdus)
    kwds = dict(
        pipeline=pipeline,
        astra_version=astra_version,
        apred=data_product.kwargs.get("apred", ""),
        run2d=data_product.kwargs.get("run2d", ""),
        # Get catalog identifier from primary HDU. Don't rely on it being in the data product kwargs.
        catalogid=data_product.kwargs.get("catalogid", image[0].header["SDSS_ID"]),
    )

    if data_product.filetype in ("apStar", "apStar-1m", "mwmStar"):
        filetype = "astraStar"
    elif data_product.filetype in ("mwmVisit",):
        filetype = "astraVisit"
    else:
        raise ValueError(
            f"Unknown output file type for {data_product} {data_product.filetype}"
        )

    try:
        path = SDSSPath(release).full(filetype, **kwds)
    except:
        log.exception(f"Could not create path for {filetype} {kwds}")

        apred, run2d, catalogid = (kwds["apred"], kwds["run2d"], kwds["catalogid"])
        if filetype == "astraVisit":
            path = expand_path(
                f"$MWM_ASTRA/{astra_version}/{run2d}-{apred}/results/visit/{catalogid % 10000:.0f}/{catalogid % 100:.0f}/astraVisit-{astra_version}-{pipeline}-{catalogid}.fits"
            )
        elif filetype == "astraStar":
            path = expand_path(
                f"$MWM_ASTRA/{astra_version}/{run2d}-{apred}/results/star/{catalogid % 10000:.0f}/{catalogid % 100:.0f}/astraStar-{astra_version}-{pipeline}-{catalogid}.fits"
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
    results: List[Dict],
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
    label_columns = list_to_dict([ea[0] for ea in results])
    model_spectra = [ea[1] for ea in results]
    meta_columns = list_to_dict([ea[2] for ea in results])

    N = len(label_columns[list(label_columns.keys())[0]])
    task_ids = [task.id] * N
    columns = [
        fits.Column(
            name="TASK_ID", array=task_ids, unit=None, **fits_column_kwargs(task_ids)
        )
    ]
    for name, values in label_columns.items():
        columns.append(
            fits.Column(
                name=name.upper(), array=values, unit=None, **fits_column_kwargs(values)
            )
        )

    # TODO: put in breaks to break up labels / model / meta
    for name, values in meta_columns.items():
        values = np.array(values)
        columns.append(
            fits.Column(
                name=name.upper(), array=values, unit=None, **fits_column_kwargs(values)
            )
        )
    # TODO: Put wavelength information in headers

    model_flux = np.atleast_2d([ea.flux.value for ea in model_spectra])
    columns.append(
        fits.Column(
            name="MODEL_FLUX",
            array=model_flux,
            unit=None,
            **fits_column_kwargs(model_flux),
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
