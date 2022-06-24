import numpy as np
import datetime
import json
import os
from astropy.io import fits
from functools import cached_property
from astropy.coordinates import SkyCoord
from astropy import units as u
from healpy import ang2pix

from astra import (log, __version__ as astra_version)
from astra.database.astradb import (database, DataProduct, TaskOutputDataProducts)
from astra.utils import flatten, expand_path

from .catalog import (get_sky_position, get_gaia_dr2_photometry)


def get_log_lambda_dispersion_kwds(wavelength, decimals=6):
    log_lambda = np.log10(wavelength)
    unique_diffs = np.unique(np.round(np.diff(log_lambda), decimals=decimals))
    if unique_diffs.size > 1:
        raise ValueError(f"Wavelength array is not uniformly sampled in log wavelength: deltas={unique_diffs}")
    return (log_lambda[0], unique_diffs[0], 1)





def get_source_metadata(task):
    """
    Return a dictionary of metadata for the source associated with a task.
    """
    input_data_products = tuple(task.input_data_products)

    # Get source and catalogid.
    sources = tuple(set(flatten([list(dp.sources) for dp in input_data_products])))
    source, *other_sources = sources
    if len(other_sources) > 0:
        raise ValueError(f"More than one source associated with the input data products: {sources} -> {input_data_products}")
    catalog_id = source.catalogid    

    # Sky position.
    ra, dec = get_sky_position(catalog_id)

    # Object identifier.
    for data_product in input_data_products:
        if "obj" in data_product.kwargs:
            object_id = data_product.kwargs["obj"]
            break
    else:
        log.warning(f"No object identifier found for {source} among {input_data_products}, creating one instead")
        coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
        ra_str = coord.to_string("hmsdms", sep="", precision=2).split()[0]
        dec_str = coord.to_string("hmsdms", sep="", precision=1).split()[1]
        object_id = (f"2M{ra_str}{dec_str}").replace(".", "")

    # Nside = 128 is fixed in SDSS-V!        
    healpix = ang2pix(128, ra, dec, lonlat=True)

    return dict(
        input_data_products=input_data_products,
        catalog_id=catalog_id,
        ra=ra,
        dec=dec,
        object_id=object_id,
        healpix=healpix
    )

COMMON_OUTPUTS = {
    "output": ("OUTPUTID", "Astra unique output identifier"),
    "task": ("TASKID", "Astra unique task identifier"),
    "meta": ("META", "Spectrum metadata"),
    "snr": ("SNR", "Signal-to-noise ratio"),
    "teff": ("TEFF", "Effective temperature [K]"),
    "logg": ("LOGG", "Log10 surface gravity [dex]"),
    "u_teff": ("U_TEFF", "Uncertainty in effective temperature [K]"),
    "u_logg": ("U_LOGG", "Uncertainty in surface gravity [dex]"),
}



def create_AstraStar_product(
    task,
    *related_tasks,
    wavelength=None,
    model_flux=None,
    model_ivar=None,
    rectified_flux=None,
    crval=None,
    cdelt=None,
    crpix=None,
    overwrite=False,
    release="sdss5",
    **kwargs
):
    """
    Create an AstraStar data product that contains output parameters and best-fit model(s) from one
    pipeline for a single star.
    
    :param task:
        The primary task responsible for creating this data product.
    
    :param related_tasks: [optional]
        Any related tasks whose parameters should be stored with this file.
    
    :param model_flux: [optional]
        The best-fitting model flux.
    """

    meta = get_source_metadata(task)
    catalog_id, healpix = (meta["catalog_id"], meta["healpix"]) # need these later for path

    hdu_primary = fits.PrimaryHDU(
        header=fits.Header(
            [
                (
                    "DATE", 
                    datetime.datetime.utcnow().strftime("%Y-%m-%d"),
                    "File creation date (UTC)"
                ),
                (   
                    "ASTRAVER", 
                    astra_version,
                    "Software version of Astra"
                ),
                (
                    "CATID", 
                    catalog_id,
                    "SDSS-V catalog identifier"
                ),
                (
                    "OBJID",
                    meta["object_id"],
                    "Object identifier"
                ),
                (
                    "RA",
                    meta["ra"],
                    "RA (J2000)"
                ),
                (
                    "DEC",
                    meta["dec"],
                    "DEC (J2000)" 
                ),
                (
                    "HEALPIX",
                    healpix,
                    "HEALPix location"
                ),
                (
                    "INPUTS",
                    ",".join([dp.path for dp in meta["input_data_products"]]),
                    "Input data products"
                )
            ]
        )
    )

    # astra.contrib.XXXX.
    component_name = task.name.split(".")[2]

    if wavelength is not None:
        if crval is None and cdelt is None and crpix is None:
            # Figure out these values ourselves.
            crval, cdelt, crpix = get_log_lambda_dispersion_kwds(wavelength)
        else:
            raise ValueError("Wavelength given AND crval, cdelt, crpix")
    

    cards = [
        ("PIPELINE", f"{component_name}", "Analysis component name"),
        ("CRVAL1", crval),
        ("CDELT1", cdelt),
        ("CRPIX1", crpix),
        ("CTYPE1", "LOG-LINEAR"),
        ("DC-FLAG", 1),            
    ]
    # Add results from the task's *first* output only.
    # TODO: What if there are many outputs? I guess we just don't allow that with this data model.
    try:
        output, *_ = task.outputs 
    except:
        log.warning(f"No summary outputs found for task {task}")
    else:
        for k, value in output.__data__.items():
            header_key, description = COMMON_OUTPUTS.get(k, (k.upper(), None))
            cards.append(
                (header_key, value, description)
            )

    header = fits.Header(cards)

    flux_col = fits.Column(name="model_flux", format="E", array=model_flux)
    ivar_col = fits.Column(name="model_ivar", format="E", array=model_ivar)
    rectified_flux_col = fits.Column(name="rectified_flux", format="E", array=rectified_flux)

    hdu_spectrum = fits.BinTableHDU.from_columns(
        [flux_col, ivar_col, rectified_flux_col],
        header=header
    )

    # Task parameters
    all_tasks = [task] + list(related_tasks)
    task_ids = [task.id for task in all_tasks]
    task_names = [task.name for task in all_tasks]
    task_columns = [
        fits.Column(name="task_id", format=get_fits_format_code(task_ids), array=task_ids),
        fits.Column(name="task_name", format=get_fits_format_code(task_names), array=task_names)
    ]

    task_parameters = {}
    for task in all_tasks:
        for parameter_name, parameter_value in task.parameters.items():
            task_parameters.setdefault(parameter_name, [])
            # TODO: Is there a better way to handle Nones? It would be nice if we could 
            #       take the parameters in the fits file and immediately reconstruct a
            #       task, but that's not possible if "" and None mean different things.

            if parameter_value is None:
                parameter_value = ""
            elif isinstance(parameter_value, (dict, list)):
                parameter_value = json.dumps(parameter_value)
            task_parameters[parameter_name].append(parameter_value)
        
    
    for parameter_name, values in task_parameters.items():
        task_columns.append(
            fits.Column(
                name=parameter_name,
                format=get_fits_format_code(values),
                array=values
            )
        )

    # Store the outputs from each task.
    hdu_tasks = fits.BinTableHDU.from_columns(task_columns)

    # Results in the hdu_tasks?
    # We can use Table(.., descriptions=(..., )) to give comments for each column type
    hdu_list = fits.HDUList([
        hdu_primary,
        hdu_spectrum,
        hdu_tasks
    ])
    # Add checksums to each
    for hdu in hdu_list:
        hdu.add_datasum()
        hdu.add_checksum()

    # TODO: Make this use sdss_access when the path is defined!
    path = expand_path(
        f"$MWM_ASTRA/{astra_version}/healpix/{healpix // 1000}/{healpix}/"
        f"astraStar-{component_name}-{catalog_id}.fits"
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    raise a
    hdu_list.writeto(path, overwrite=overwrite)

    # Create the data product record.
    # TODO: Make this use "AstraStar" filetype when it is defined!
    with database.atomic():
        output_dp, was_created = DataProduct.get_or_create(
            release=release,
            filetype="full",
            kwargs=dict(
                full=path
            )
        )
        TaskOutputDataProducts.create(
            task=task,
            data_product=output_dp
        )
    
    log.info(f"{'Created' if was_created else 'Retrieved'} data product {output_dp} for {task}: {path}")


def get_fits_format_code(values):
    fits_format_code = {
        bool: "L",
        int: "K",
        str: "A",
        float: "E",
        type(None): "A"
    }.get(type(values[0]))
    assert fits_format_code is not None
    if fits_format_code == "A":
        max_len = max(1, max(map(len, values)))
        return f"{max_len}A"
    return fits_format_code

