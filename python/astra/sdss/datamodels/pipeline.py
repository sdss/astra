"""Create pipeline products (e.g., ``astraVisit-*``/``astraStar-*``)."""

from ast import Name
import os
import numpy as np
from typing import Dict, List, Union, Optional, Tuple
from sdss_access import SDSSPath
from astra import log, __version__ as astra_version
from astra.database.astradb import Task, DataProduct, Source, TaskOutputDataProducts
from astropy.io import fits
from functools import partial

from astra.sdss.datamodels.base import (
    add_check_sums,
    create_empty_hdu,
    fits_column_kwargs,
    create_primary_hdu_cards,
    wavelength_cards,
    metadata_cards,
    BLANK_CARD,
    add_glossary_comments,
    add_table_category_headers,
    _get_extname,
    FILLER_CARD
)

from astra.sdss.datamodels.mwm import (
    HDU_DESCRIPTIONS,
    get_data_hdu_observatory_and_instrument,
)
from astra.utils import list_to_dict, expand_path


def create_pipeline_product(
    task: Task,
    data_product: DataProduct,
    results: dict,
    release: str = "sdss5",
    source: Optional[Source] = None,
    header_groups: Optional[dict] = None
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
    
    :header_groups: [optional]
        A dictionary of header groups to add to the data product. The keys should be the name of
        the HDU extension (e.g., 'APOGEE/APO') and the values should be a list of two-length
        tuples where each tuple contains: the header key where the group starts, and the name of
        the group.
    """

    if data_product.filetype in ("mwmVisit", "mwmStar"):
        # Copy primary HDU.
        with fits.open(data_product.path) as image:
            primary_hdu_cards = [[k, v, c] for k, v, c in image[0].header.cards]
    else:
        # If it's a `full` file then we will have a bad time.
        if source is None:
            try:
                (source,) = data_product.sources
            except:
                log.exception(
                    f"Could not find unique source associated with data product {data_product}"
                )
        # list(map(list, ...)) so we can edit the entries.
        primary_hdu_cards = list(map(list, create_primary_hdu_cards(source, HDU_DESCRIPTIONS)))

    # Note: inserting a header into an existing HDU causes all the existing comments to be deleted.
    #       That's why we're doing this using the cards instead

    # Update comments.
    for i, (key, value, comment) in enumerate(primary_hdu_cards):
        if key == "COMMENT" and value.startswith("HDU "):
            prefix, desc = value.split(": ")
            if prefix == "HDU 0": continue 
            primary_hdu_cards[i][1] = f"{prefix}: Model fits to {desc}"

    # Insert cards after MAPPERS
    data_model_cards = [
        BLANK_CARD,
        (" ", "INPUT DATA MODEL KEYWORDS", None),
        ("RELEASE", data_product.release, None),
        ("FILETYPE", data_product.filetype, None),
    ]
    for k in data_product.kwargs.keys():
        data_model_cards.append((k.upper(), data_product.kwargs[k], None))

    # Add a comment with the URL
    try:
        url = SDSSPath(data_product.release).url(data_product.filetype, **data_product.kwargs)
    except:
        url = "NO URL COULD BE GENERATED"
    # it's a long URL, split it by the version

    data_model_cards.extend([
        BLANK_CARD,
        (" ", "INPUT DATA PRODUCT", None),
        ("IDP_URL", url, None)
    ])
    #"https://data.sdss5.org/sas/sdsswork/mwm/spectro/astra/0.2.4dev/v6_0_9-daily/spectra/star/1959/59/mwmStar-0.2.4dev-4346151959.fits"
    
    get_index = lambda k: [i for i, (key, value, comment) in enumerate(primary_hdu_cards) if key == k][0]

    index = get_index("MAPPERS") + 1
    for card in data_model_cards[::-1]:
        primary_hdu_cards.insert(index, card)

    # Add task and data product identifiers
    index = get_index("GAIA_ID") + 1
    primary_hdu_cards.insert(index, ("IDP_ID", data_product.id, None))
    primary_hdu_cards.insert(index, ("TASK_ID", task.id, None))
    
    primary_hdu = fits.PrimaryHDU(header=fits.Header(list(map(tuple, primary_hdu_cards))))
    
    add_glossary_comments(primary_hdu)
    primary_hdu.add_checksum()

    hdus = [primary_hdu]
    for i, (observatory, instrument) in enumerate(
        get_data_hdu_observatory_and_instrument()
    ):
        key = _get_extname(instrument, observatory)
        hdu_header_groups = (header_groups or {}).get(key, [])
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
                    header_groups=hdu_header_groups
                )
            )
    # TODO: Warn about result keys that don't match any HDU.
    
    # Parse pipeline name from the task name
    pipeline = task.name.split(".")[2]
    task_id = task.id

    # Add check sums to everything except the primary, since we already did it and doing it again screws it up.
    add_check_sums(hdus[1:])

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
    capitalize_column_names: bool = True,
    header_groups: Optional[List[Tuple[str]]] = None,
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
        _name = (name.upper() if capitalize_column_names else name).strip()
        _values = np.atleast_1d(np.array(values))
        columns.append(
            fits.Column(
                name=_name, 
                array=_values, 
                unit=None, 
                **fits_column_kwargs(_values)
            )
        )

    # Copy header cards from input data product.
    keep, cards = (False, [])
    extname = _get_extname(instrument, observatory)
    for key, value, comment in fits.getheader(data_product.path, extname).cards:
        if value == "METADATA":
            keep = True
        if key == "TTYPE1":
            cards.pop(-1)
            cards.pop(-1)
            break

        if keep:
            cards.append((key, value, comment))

    # Add pipeline name.
    pipeline = task.name.split(".")[2]
    cards.extend([
        BLANK_CARD,
        (" ", "PIPELINE INFORMATION", ""),
        ("PIPELINE", pipeline, ""),
        BLANK_CARD,
    ])
    cards.append(FILLER_CARD)

    hdu = fits.BinTableHDU.from_columns(
        columns, 
        name=f"{instrument}/{observatory}", 
        header=fits.Header(cards=cards)
    )
    add_glossary_comments(hdu)
    if header_groups is not None:
        add_table_category_headers(hdu, header_groups)
    return hdu
