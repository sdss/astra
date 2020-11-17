
import numpy as np
from astropy.io import fits
from astropy.table import Table



def create_image_hdu(data, header, name=None, dtype=None, **kwargs):
    kwds = dict(do_not_scale_image_data=True)
    kwds.update(**kwargs)
    return fits.ImageHDU(
        data=data.astype(dtype or data.dtype),
        header=header,
        name=name,
        **kwds
    )


def get_log_linear_wavelength_keywords(wavelength_array):
    """
    Return the headers needed to construct the given log linear wavelength
    array.

    :param wavelength_array:
        The array of wavelength values.
    
    :returns:
        A dictionary of header keys and values that can be used to reconstruct
        a log linear wavelength array.

    """

    raise NotImplementedError()


def create_astra_source(
        catalog_id, # Should this be catalogid? Use this to populate additional information.
        obj,
        telescope,
        healpix,
        normalized_flux,
        normalized_ivar,
        model_flux,
        #model_ivar,
        crval,
        cdelt,
        crpix,
        ctype,
        reference_task,
        header=None,
        data_table=None,
    ):
    """
    Return a `astropy.io.fits.HDUList` object for an astraSource data product.

    :catalog_id:
        The identifier of this source in the SDSS-V database.

    :param obj:
        The name of the object (e.g., 2M000000+000000).

    :param telescope:
        A short-hand string describing the telescope that took these observations.

    :param healpix:
        The identifier of the HEAL pixel that this object falls in to.
    
    :param normalized_flux:
        A (N, P) shape array that stores the pseudo-continuum normalized fluxes values,
        where N is the number of spectra for this source (`N_SPECTRA` header) and P is
        the number of pixels per spectra.
    
    :param normalized_ivar:
        A (N, P) shape array that stores the inverse variances of the pseudo-continuum 
        normalized fluxes. For example, if the $1\sigma$ Gaussian uncertainty in pseudo-
        continuum normalized flux in a pixel is $\sigma_n$, then the inverse variance of
        the pseudo-continuum normalized flux is $1/(\sigma_n^2)$. 
        
        Similarly if the $1\sigma$ Gaussian uncertainty in *un-normalized flux* $y$ is 
        $\sigma$ and the continuum is $C$, then the pseudo-continuum normalized flux is
        $y/C$ and the inverse variance of the pseudo-continuum normalized flux is 
        $(C/\sigma)^2$.
        
    :param model_flux:
        A (N, P) shape array that stores the best-fitting model flux for each observation.
    
    :param model_ivar:
        A (N, P) shape array that stores the inverse variances of the model fluxes for
        each observation. If no inverse variances are available then this array can be set
        to a single value.
    
    :param crval:
        The coordinate reference value for the wavelength array.
    
    :param cdelt:
        The difference in wavelength (or log wavelength) between ajacent pixels.
    
    :param crpix:
        The coordinate reference pixel for the wavelength array (starts at 1).
    
    :param ctype:
        A string description for the wavelength format (e.g., LOG-LINEAR).
    
    :param reference_task:
        A reference task that performed this analysis. This task is used to tabulate
        the parameters used for this task, and a description of those parameters.
    """

    flux = np.atleast_2d(normalized_flux)
    ivar = np.atleast_2d(normalized_ivar)
    model_flux = np.atleast_2d(model_flux)
    #cont = np.atleast_2d(continuum)

    N, P = shape = normalized_flux.shape

    # Check shapes.
    if shape != ivar.shape:
        raise ValueError(f"flux and ivar have different shapes ({shape} != {ivar.shape})")
    #if shape != cont.shape:
    #    raise ValueError(f"flux and continuum have different shapes ({shape} != {cont.shape})")
    if shape != model_flux.shape:
        raise ValueError(f"flux and model flux have different shapes ({shape} != {model_flux.shape})")

    # Build data HDUs.
    dtype = np.float32
    data_header = fits.Header(cards=[
        ("CRVAL1", crval),
        ("CDELT1", cdelt),
        ("CRPIX1", crpix),
        ("CTYPE1", ctype),
        # TODO: Should we be passing this from keywords/
        ("DC-FLAG", 1)
    ])

    flux_hdu = create_image_hdu(data=flux, header=data_header, name="NORMALIZED_FLUX", dtype=dtype)
    ivar_hdu = create_image_hdu(data=ivar, header=data_header, name="NORMALIZED_IVAR", dtype=dtype)
    model_flux_hdu = create_image_hdu(data=model_flux, header=data_header, name="MODEL_FLUX", dtype=dtype)
    
    #cont_hdu = create_image_hdu(data=cont, header=data_header, name="CONTINUUM", dtype=dtype)

    # Create header for primary HDU.
    #
    from time import gmtime, strftime
    
    astra_version = ".".join(map(str, [
        reference_task.astra_version_major,
        reference_task.astra_version_minor,
        reference_task.astra_version_micro
    ]))

    apred = getattr(reference_task, "apred", "")
    run2d = getattr(reference_task, "run2d", "")

    cards = [
        ("CREATED", strftime("%Y-%m-%d %H:%M:%S", gmtime()), "GMT when this file was created"),
        ("ASTRA", astra_version, "Astra version"), 
        # Data reduction versions.
        ("APRED", apred, "MWM reduction version (if APOGEE instrument used)"),
        ("RUN2D", run2d, "BHM reduction version (if BOSS instrument used)"),
        ("TASKNAME", reference_task.task_namespace, "Namespace of the primary Astra analysis task"),
        #("CATID", catalog_id, "Catalog ID in the SDSS-V catalog database."),
        #("OBJ", obj, "Name of the object (e.g., 2M000000+000000)."),
        #("HEALPIX", healpix, "The identifier of the HEL pixel this object is in."),
        #("TELESCOPE", telescope, "A string describing the telescope that took these observations."),
    ]
    # Update with additional information from catalog database:

    #primary_hdu = fits.Header(cards=cards)
    primary_hdu = fits.PrimaryHDU(header=header)
    del primary_hdu.header["HISTORY"]

    primary_hdu.header.extend(cards, end=True)

    history = [
        "HDU 0: Primary header",
        "HDU 1: Pseudo-continuum normalized flux",
        "HDU 2: Inverse variance of pseudo-continuum normalized flux",
        "HDU 3: Pixel bitmask",
        "HDU 4: Model pseudo-continuum normalized flux",
        "HDU 5: Data table"
    ]
    for row in history:
        primary_hdu.header["HISTORY"] = row

    # Data table HDU
    
    table_hdu = fits.BinTableHDU(
        data=data_table,
        name="DATA"
    )

    hdu_list = [
        primary_hdu,
        flux_hdu,
        ivar_hdu,
        model_flux_hdu,
        table_hdu
    ]

    return fits.HDUList(hdu_list)