import os
import numpy as np
from astropy.io import fits
from astropy.table import Table
from time import gmtime, strftime

from astra.database import session, astradb


def write_astra_source_data_product(
        output_path, 
        spectrum,
        normalized_flux,
        normalized_ivar,
        continuum,
        model_flux,
        model_ivar,
        results_table,
        instance,
        **kwargs
    ):
    """
    Write an AstraSource object to disk.
    
    :param output_path:
        The location on disk where to write the AstraSource object to.
        
    :param spectrum:
        The spectrum that was used for analysis. Headers and relevant quality flags from this
        spectrum will be propagated.
        
    :param normalized_flux:
        A (N, P) shape array of the pseudo-continuum normalized observed flux, where N is
        the number of spectra and P is the number of pixels.
    
    :param normalized_ivar:
        A (N, P) shape array of the inverse variance of the pseudo-continuum normalized observed
        flux, where N is the number of spectra and P is the number of pixels. For example, 
        if the $1\sigma$ Gaussian uncertainty in pseudo-continuum normalized flux in a pixel is 
        $\sigma_n$, then the inverse variance of the pseudo-continuum normalized flux is $1/(\sigma_n^2)$. 
        
        Similarly if the $1\sigma$ Gaussian uncertainty in *un-normalized flux* $y$ is 
        $\sigma$ and the continuum is $C$, then the pseudo-continuum normalized flux is
        $y/C$ and the inverse variance of the pseudo-continuum normalized flux is 
        $(C/\sigma)^2$.
    
    :param continuum:
        A (N, P) shape array of the continuum value used to pseudo-normalize the observations,
        where N is the number of spectra and P is the number of pixels.

    :param model_flux:
        A (N, P) shape array of the pseudo-continuum normalized model flux, where N is the number
        of spectra and P is the number of pixels.
    
    :param model_ivar:
        A (N, P) shape array of the inverse variance of the pseudo-continuum normalized model flux,
        where N is the number of spectra and P is the number of pixels. This gives a measure of the
        inverse variance in the model predictions, which is often not known. If no model uncertainty
        is known then this array should be set to a constant positive value (e.g., 1).
    
    :param results_table:
        A :py:mod:`astropy.table.Table` of outputs from the analysis of the input spectra. This can
        contain any number of columns, but the naming convention of these columns should (as closely
        as possible) follow the conventions of other analysis tasks so that there is not great
        variation in names among identical columns (e.g., `R_CHI_SQ` and `RCHISQ` and `REDUCED_CHI_SQ`). 
        
        This table will be supplemented with the parameters and descriptions of the analysis task.        
    
    :param instance:
        The Task Instance in the database referencing this analysis.
    """

    # Check array shapes etc.
    normalized_flux, normalized_ivar, model_flux, model_ivar, continuum, results_table = _check_shapes(
        normalized_flux, normalized_ivar, model_flux, model_ivar, continuum, results_table
    )
    
    # Build some HDUs.
    dtype = kwargs.pop("dtype", np.float32)

    # Copy the wavelength header cards we need.
    dispersion_map_keys = ("CRVAL1", "CDELT1", "CRPIX1", "CTYPE1", "DC-FLAG")
    header = fits.Header(cards=[(key, spectrum.meta["header"][key]) for key in dispersion_map_keys])
    
    flux_hdu = create_image_hdu(
        data=normalized_flux, 
        header=header, 
        name="NORMALIZED_FLUX", 
        bunit="Pseudo-continuum normalized flux (-)",
        dtype=dtype
    )
    ivar_hdu = create_image_hdu(
        data=normalized_ivar,
        header=header,
        name="NORMALIZED_IVAR",
        bunit="Inverse variance of pseudo-continuum normalized flux (-)",
        dtype=dtype
    )
    bitmask_hdu = create_image_hdu(
        data=get_bitmask(spectrum),
        header=header,
        name="BITMASK",
        bunit="Pixel bitmask",
    )
    continuum_hdu = create_image_hdu(
        data=continuum,
        header=header,
        name="CONTINUUM",
        bunit="Pseudo-continuum flux (10^-17 erg/s/cm^2/Ang)",
        dtype=dtype
    )        
    model_flux_hdu = create_image_hdu(
        data=model_flux,
        header=header,
        name="MODEL_FLUX",
        bunit="Model flux (-)",
        dtype=dtype
    )
    model_ivar_hdu = create_image_hdu(
        data=model_ivar,
        header=header,
        name="MODEL_IVAR",
        bunit="Inverse variance of model flux (-)",
        dtype=dtype
    )
    # TODO: Duplicate the visit information from headers to the results table?
    #       (E.g., fiber, mjd, field, RV information?)
    results_table_hdu = fits.BinTableHDU(
        data=results_table,
        name="RESULTS"
    )

    # Create a task parameter table.
    # TODO: Fill this in.
    parameter_table_hdu = fits.BinTableHDU(
        data=Table(data={"KEY": [""], "VALUE": [""]}),
        name="TASK_PARAMETERS"
    )
    
    # Create a Primary HDU with the headers from the observation.
    primary_hdu = fits.PrimaryHDU(header=spectrum.meta["header"])
    from astra import __version__ as astra_version
    cards = [
        ("ASTRA", astra_version, "Astra version"), 
        ("TI_PK", instance.pk, "Astra task instance primary key"),
        ("CREATED", strftime("%Y-%m-%d %H:%M:%S", gmtime()), "GMT when this file was created"),
    ]
    # Delete reduction pipeline history comments and add what we need.
    del primary_hdu.header["HISTORY"]
    primary_hdu.header.extend(cards, end=True)

    # Add our own history comments.
    hdus = [
        (primary_hdu, "Primary header"),
        (flux_hdu, "Pseudo-continuum normalized flux"),
        (ivar_hdu, "Inverse variance of pseudo-continuum normalized flux"),
        (bitmask_hdu, "Pixel bitmask"),
        (continuum_hdu, "Pseudo-continuum used"),
        (model_flux_hdu, "Pseudo-continuum normalized model flux"),
        (model_ivar_hdu, "Inverse variance of pseudo-continuum normalized model flux"),
        (parameter_table_hdu, "Astra task parameters"),
        (results_table_hdu, "Results")
    ]

    hdu_list = []
    for i, (hdu, comment) in enumerate(hdus):
        hdu_list.append(hdu)
        hdu_list[0].header["HISTORY"] = f"HDU {i}: {comment}"
            
    image = fits.HDUList(hdu_list)

    kwds = dict(
        checksum=True, 
        overwrite=True,
        output_verify="silentfix"
    )
    kwds.update(kwargs)

    # Check that the parent directory exists.
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    return image.writeto(output_path, **kwds)


def create_image_hdu(data, header, name=None, dtype=None, bunit=None, **kwargs):
    kwds = dict(do_not_scale_image_data=True)
    kwds.update(**kwargs)
    hdu = fits.ImageHDU(
        data=data.astype(dtype or data.dtype),
        header=header,
        name=name,
        **kwds
    )
    if bunit is not None:
        hdu.header["BUNIT1"] = bunit
    return hdu



def _check_shapes(normalized_flux, normalized_ivar, model_flux, model_ivar, continuum, results_table):
    """
    Check that all input array shapes are consistent.
    """

    if model_flux is None:
        model_flux = np.nan * np.ones_like(normalized_flux)
    if model_ivar is None:
        model_ivar = np.nan * np.ones_like(normalized_flux)
    if continuum is None:
        continuum = np.nan * np.ones_like(normalized_flux)

    normalized_flux = np.atleast_2d(normalized_flux)
    normalized_ivar = np.atleast_2d(normalized_ivar)
    model_flux = np.atleast_2d(model_flux)
    model_ivar = np.atleast_2d(model_ivar)
    continuum = np.atleast_2d(continuum)

    N, P = shape = normalized_flux.shape

    if N > P:
        raise ValueError(f"I do not believe that you have more visits than pixels!")

    if shape != normalized_ivar.shape:
        raise ValueError(f"normalized_flux and normalized_ivar have different shapes ({shape} != {normalized_ivar.shape})")
    if shape != model_flux.shape:
        raise ValueError(f"normalized_flux and model_flux have different shapes ({shape} != {model_flux.shape})")
    if shape != model_ivar.shape:
        raise ValueError(f"normalized_flux and model_ivar have different shapes ({shape} != {model_ivar.shape}")
    if shape != continuum.shape:
        raise ValueError(f"normalized_flux and continuum have different shapes ({shape} != {continuum.shape})")
    if N != len(results_table):
        raise ValueError(f"results table should have the same number of rows as there are spectra ({N} != {len(data_table)})")

    results_table = results_table.copy()

    return (normalized_flux, normalized_ivar, model_flux, model_ivar, continuum, results_table)



def get_bitmask(spectrum):
    """
    Return a bitmask array, given the spectrum. 
    The reason for this is because the BHM pipeline produces AND/OR bitmasks and
    the MWM pipeline produces a single bitmask.
    """
    try:
        # BHM/BOSS bitmask
        return spectrum.meta["bitmask"]["or_mask"]

    except:
        # MWM
        return spectrum.meta["bitmask"]

