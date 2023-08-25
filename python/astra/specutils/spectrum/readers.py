"""Register reader functions for various spectral formats."""
from collections import OrderedDict
from functools import wraps
import numpy as np
import os
from astropy.io import fits
from astropy import units as u
from astropy.nddata import StdDevUncertainty, InverseVariance
from specutils import SpectralAxis, Spectrum1D, SpectrumList
from specutils.io.registers import get_loaders_by_extension, io_registry


from astra.utils import log

"""
From the specutils documentation:
- For spectra that have different shapes, use SpectrumList.
- For spectra that have the same shape but different spectral axes, see SpectrumCollection. 
- For a spectrum or spectra that all share the same spectral axis, use Spectrum1D. 
"""


def clear_fits_registry_for_spectrum_objects(extension, dtypes):
    for data_format in get_loaders_by_extension(extension):
        for dtype in dtypes:
            try:
                io_registry.unregister_identifier(data_format, dtype)
            except:
                continue
    return None


# The registry is full of junk, and many of them don't even work out of the box.
clear_fits_registry_for_spectrum_objects(
    extension="fits", dtypes=(Spectrum1D, SpectrumList)
)

# The `specutils.io.registers.data_loader` doesn't respect the `priority` keyword,
# and does some funky incompatible shit by double-registering things as SpectrumList objects


def data_loader(label, identifier, dtype, extensions=None, priority=0, force=False):
    def identifier_wrapper(ident):
        def wrapper(*args, **kwargs):
            try:
                return ident(*args, **kwargs)
            except Exception as e:
                return False

        return wrapper

    def decorator(func):
        io_registry.register_reader(label, dtype, func, priority=priority, force=force)
        io_registry.register_identifier(
            label, dtype, identifier_wrapper(identifier), force=force
        )
        func.extensions = extensions

        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    return decorator


is_filetype = lambda filetype: lambda f, o, *args, **kwargs: o.split("/")[
    -1
].startswith(filetype)


@data_loader(
    "mwmVisit",
    identifier=is_filetype("mwmVisit"),
    dtype=Spectrum1D,
    priority=1,
    extensions=["fits"],
)
def load_sdss_mwmVisit_1d(path, hdu, **kwargs):
    with fits.open(path) as image:
        return _load_mwmVisit_or_mwmStar_hdu(image, hdu)


@data_loader(
    "mwmVisit",
    identifier=is_filetype("mwmVisit"),
    dtype=SpectrumList,
    priority=1,
    extensions=["fits"],
)
def load_sdss_mwmVisit_list(path, hdu=None, **kwargs):
    return _load_mwmVisit_or_mwmStar_spectrum_list(path, hdu, **kwargs)


@data_loader(
    "mwmStar",
    identifier=is_filetype("mwmStar"),
    dtype=Spectrum1D,
    priority=20,
    extensions=["fits"],
)
def load_sdss_mwmStar_1d(path, hdu, **kwargs):
    with fits.open(path) as image:
        return _load_mwmVisit_or_mwmStar_hdu(image, hdu, **kwargs)


@data_loader(
    "mwmStar",
    identifier=is_filetype("mwmStar"),
    dtype=SpectrumList,
    priority=20,
    extensions=["fits"],
)
def load_sdss_mwmStar_list(path, hdu=None, **kwargs):
    return _load_mwmVisit_or_mwmStar_spectrum_list(path, hdu, **kwargs)


@data_loader(
    "apStar",
    identifier=is_filetype(("apStar", "asStar")),
    dtype=Spectrum1D,
    priority=10,
    extensions=["fits"],
)
def load_sdss_apStar(path, data_slice=None, **kwargs):
    flux_unit = u.Unit("1e-17 erg / (Angstrom cm2 s)")  # TODO

    with fits.open(path) as image:
        wavelength = _wcs_log_linear(
            image[1].header["NAXIS1"],
            image[1].header["CDELT1"],
            image[1].header["CRVAL1"],
        )
        spectral_axis = u.Quantity(wavelength, unit=u.Angstrom)

        slicer = slice(*data_slice) if data_slice is not None else slice(None)

        try:                
            flux = u.Quantity(image[1].data[slicer], unit=flux_unit)
            e_flux = StdDevUncertainty(image[2].data[slicer])
        except:
            if data_slice is not None:
                # HACK to do what we want without updating the pparameters for so many tasks
                slicer = slice(*data_slice[0])
                flux = u.Quantity(image[1].data[slicer], unit=flux_unit)
                e_flux = StdDevUncertainty(image[2].data[slicer])
            else:
                raise

        snr = [image[0].header["SNR"]]
        n_visits = image[0].header["NVISITS"]
        if n_visits > 1:
            snr.append(snr[0])  # duplicate S/N value for second stacking method
            snr.extend([image[0].header[f"SNRVIS{i}"] for i in range(1, 1 + n_visits)])

        # TODO: Consider more explicit key retrieval? Or make this common functionality somewhere
        meta = OrderedDict([])
        for key in image[0].header.keys():
            if key.startswith(("TTYPE", "TFORM", "TDIM")) or key in (
                "",
                "COMMENT",
                "CHECKSUM",
                "DATASUM",
                "NAXIS",
                "NAXIS1",
                "NAXIS2",
                "XTENSION",
                "BITPIX",
                "PCOUNT",
                "GCOUNT",
                "TFIELDS",
            ):
                continue
            meta[key.lower()] = image[0].header[key]

        meta["SNR"] = np.array(snr)[slicer]
        meta["BITMASK"] = image[3].data[slicer]
        meta["J_MAG"] = image[0].header["J"]
        meta["H_MAG"] = image[0].header["H"]
        meta["K_MAG"] = image[0].header["K"]
        print(f"Warning: not yet loading data product entries")

    return Spectrum1D(
        spectral_axis=spectral_axis, flux=flux, uncertainty=e_flux, meta=meta
    )


@data_loader(
    "apStar",
    identifier=is_filetype(("apStar", "asStar")),
    dtype=SpectrumList,
    priority=10,
    extensions=["fits"],
)
def load_sdss_apStar_list(path, **kwargs):
    flux_unit = u.Unit("1e-17 erg / (Angstrom cm2 s)")  # TODO

    spectra = []
    with fits.open(path) as image:
        wavelength = _wcs_log_linear(
            image[1].header["NAXIS1"],
            image[1].header["CDELT1"],
            image[1].header["CRVAL1"],
        )
        spectral_axis = u.Quantity(wavelength, unit=u.Angstrom)
        flux = u.Quantity(np.atleast_2d(image[1].data), unit=flux_unit)
        e_flux = StdDevUncertainty(np.atleast_2d(image[2].data))

        N, P = flux.shape

        snr = [image[0].header["SNR"]]
        n_visits = image[0].header["NVISITS"]
        mjd = [image[0].header[f"JD{i}"] - 2400000.5 for i in range(1, 1 + n_visits)]
        fiber = [image[0].header[f"FIBER{i}"] for i in range(1, 1 + n_visits)]
        telescope = "lco25m" if image[0].header["SFILE1"].startswith("as") else "apo25m"
        date_obs = [image[0].header[f"DATE{i}"] for i in range(1, 1 + n_visits)]

        if n_visits > 1:
            snr.append(snr[0])  # duplicate S/N value for second stacking method
            snr.extend([image[0].header[f"SNRVIS{i}"] for i in range(1, 1 + n_visits)])

            mjd.insert(0, None)
            mjd.insert(0, None)
            fiber.insert(0, None)
            fiber.insert(0, None)
            date_obs.insert(0, None)
            date_obs.insert(0, None)



        meta = OrderedDict([])
        for key in image[0].header.keys():
            if key.startswith(("TTYPE", "TFORM", "TDIM")) or key in (
                "",
                "COMMENT",
                "CHECKSUM",
                "DATASUM",
                "NAXIS",
                "NAXIS1",
                "NAXIS2",
                "XTENSION",
                "BITPIX",
                "PCOUNT",
                "GCOUNT",
                "TFIELDS",
            ):
                continue
            meta[key.lower()] = image[0].header[key]
        
        meta["J_MAG"] = image[0].header["J"]
        meta["H_MAG"] = image[0].header["H"]
        meta["K_MAG"] = image[0].header["K"]

        # Some strange situations where N_visits = 1 and flux has shape (2, 8575).
        for i in range(n_visits if n_visits == 1 else N):
            meta_i = meta.copy()
            meta_i.update(
                SNR=snr[i],
                TELESCOPE=telescope,
                FIBER=fiber[i],
                MJD=mjd[i],
                BITMASK=image[3].data[i],
                DATE=date_obs[i],
            )
            spectra.append(
                Spectrum1D(
                    spectral_axis=spectral_axis,
                    flux=flux[i],
                    uncertainty=e_flux[i],
                    meta=meta_i
                )
            )

        
    return SpectrumList(spectra)


@data_loader(
    "apVisit",
    identifier=is_filetype("apVisit"),
    dtype=Spectrum1D,
    priority=10,
    extensions=["fits"],
)
def load_sdss_apVisit(path, **kwargs):
    flux_unit = u.Unit("1e-17 erg / (Angstrom cm2 s)")  # TODO

    ordered = lambda d: d[::-1].flatten()

    with fits.open(path) as image:
        spectral_axis = u.Quantity(ordered(image[4].data), unit=u.Angstrom)

        # Handle chips
        flux = u.Quantity(ordered(image[1].data), unit=flux_unit)
        e_flux = StdDevUncertainty(ordered(image[2].data))

        # TODO: Consider more explicit key retrieval? Or make this common functionality somewhere
        meta = OrderedDict([])
        for key in image[0].header.keys():
            if key.startswith(("TTYPE", "TFORM", "TDIM")) or key in (
                "",
                "COMMENT",
                "CHECKSUM",
                "DATASUM",
                "NAXIS",
                "NAXIS1",
                "NAXIS2",
                "XTENSION",
                "BITPIX",
                "PCOUNT",
                "GCOUNT",
                "TFIELDS",
            ):
                continue
            meta[key.lower()] = image[0].header[key]

        meta["bitmask"] = ordered(image[3].data)
        # TODO: Include things like sky flux, sky error, telluric flux, telluric error?
        #       wavelength coefficients? lsf coefficients?

    return Spectrum1D(
        spectral_axis=spectral_axis, flux=flux, uncertainty=e_flux, meta=meta
    )


@data_loader(
    "apVisit",
    identifier=is_filetype("apVisit"),
    dtype=SpectrumList,
    priority=10,
    extensions=["fits"],
)
def load_sdss_apVisit_multi(path, **kwargs):
    flux_unit = u.Unit("1e-17 erg / (Angstrom cm2 s)")  # TODO
    spectra = SpectrumList()
    with fits.open(path) as image:
        # TODO: Consider more explicit key retrieval? Or make this common functionality somewhere
        common_meta = OrderedDict([])
        for key in image[0].header.keys():
            if key.startswith(("TTYPE", "TFORM", "TDIM")) or key in (
                "",
                "COMMENT",
                "CHECKSUM",
                "DATASUM",
                "NAXIS",
                "NAXIS1",
                "NAXIS2",
                "XTENSION",
                "BITPIX",
                "PCOUNT",
                "GCOUNT",
                "TFIELDS",
            ):
                continue
            common_meta[key.lower()] = image[0].header[key]

        for chip in range(image[1].data.shape[0]):
            spectral_axis = u.Quantity(image[4].data[chip], unit=u.Angstrom)
            flux = u.Quantity(image[1].data[chip], unit=flux_unit)
            e_flux = StdDevUncertainty(image[2].data[chip])

            meta = common_meta.copy()
            meta["BITMASK"] = image[3].data[chip]

            # TODO: Include things like sky flux, sky error, telluric flux, telluric error?
            #       wavelength coefficients? lsf coefficients?

            spectra.append(
                Spectrum1D(
                    spectral_axis=spectral_axis,
                    flux=flux,
                    uncertainty=e_flux,
                    meta=meta,
                )
            )

    return spectra


@data_loader(
    "specFull",
    # Note the path definition here is not the same as other SDSS-V data models.
    identifier=is_filetype("spec"),
    dtype=Spectrum1D,
    priority=10,
    extensions=["fits"],
)
def load_sdss_specFull(path, **kwargs):
    with fits.open(path) as image:
        # The flux unit is stored in the `BUNIT` keyword, but not in a form that astropy
        # will accept.
        flux_unit = u.Unit("1e-17 erg / (Angstrom cm2 s)")  # TODO
        spectral_axis = u.Quantity(10 ** image[1].data["LOGLAM"], unit=u.Angstrom)

        flux = u.Quantity(image[1].data["FLUX"], unit=flux_unit)
        ivar = InverseVariance(image[1].data["IVAR"])

        meta = OrderedDict([])
        for key in image[0].header.keys():
            if key.startswith(("TTYPE", "TFORM", "TDIM")) or key in (
                "",
                "COMMENT",
                "CHECKSUM",
                "DATASUM",
                "NAXIS",
                "NAXIS1",
                "NAXIS2",
                "XTENSION",
                "BITPIX",
                "PCOUNT",
                "GCOUNT",
                "TFIELDS",
            ):
                continue
            meta[key] = image[0].header[key]

        # Note: specFull file does not include S/N value, but this gets calculated
        #       for mwmVisit/mwmStar files when they are created

    return Spectrum1D(
        spectral_axis=spectral_axis, flux=flux, uncertainty=ivar, meta=meta
    )


@data_loader(
    "specFull",
    identifier=is_filetype("spec"),
    dtype=SpectrumList,
    priority=1,
    extensions=["fits"],
)
def load_sdss_specFull_multi(path, **kwargs):
    return SpectrumList([load_sdss_specFull(path, **kwargs)])


def _wcs_log_linear(naxis, cdelt, crval):
    return 10 ** (np.arange(naxis) * cdelt + crval)


def _load_mwmVisit_or_mwmStar_spectrum_list(path, hdu=None, **kwargs):
    spectra = SpectrumList()
    with fits.open(path) as image:
        hdu = range(1, len(image)) if hdu is None else [hdu]
        for hdu in hdu:
            if image[hdu].header["DATASUM"] == "0":
                continue
            spectra.extend(_load_mwmVisit_or_mwmStar_hdu_as_spectrum_list(image, hdu))
    return spectra

def _load_mwmVisit_or_mwmStar(path, **kwargs):
    spectra = SpectrumList()
    with fits.open(path) as image:
        for hdu in range(1, len(image)):
            if image[hdu].header["DATASUM"] == "0":
                spectra.append(None)
                continue
            spectra.append(_load_mwmVisit_or_mwmStar_hdu(image, hdu))
    return spectra

def _load_mwmVisit_or_mwmStar_hdu_as_spectrum_list(image, hdu, **kwargs):
    flux_unit = u.Unit("1e-17 erg / (Angstrom cm2 s)")  # TODO
    try:
        wavelength = np.array(image[hdu].data["LAMBDA"])[0]
    except:
        wavelength = _wcs_log_linear(
            image[hdu].header["NPIXELS"],
            image[hdu].header["CDELT"],
            image[hdu].header["CRVAL"],
        )
    finally:
        spectral_axis = u.Quantity(wavelength, unit=u.Angstrom)

    pixel_bitmask = np.atleast_2d(image[hdu].data["BITMASK"])
    flux_value = np.atleast_2d(image[hdu].data["FLUX"])
    N, P = flux_value.shape

    common_meta = OrderedDict([])
    for hdu_idx in (0, hdu):
        for key in image[hdu_idx].header.keys():
            if key.startswith(("TTYPE", "TFORM", "TDIM")) or key in (
                "",
                "COMMENT",
                "CHECKSUM",
                "DATASUM",
                "NAXIS",
                "NAXIS1",
                "NAXIS2",
                "XTENSION",
                "BITPIX",
                "PCOUNT",
                "GCOUNT",
                "TFIELDS",
            ):
                continue
            common_meta[key] = image[hdu_idx].header[key]    

    keys = (
        "MJD", 
        "FIBER", 
        ("FIELD", "FIELDID"), "PLATE", "TELESCOPE", "DATE-OBS", "DITHERED", "NPAIRS",
        "CONTINUUM_THETA", "V_RAD", "E_V_RAD", "V_REL", "V_BC", "VISIT_PK", "RV_VISIT_PK", "DATA_PRODUCT_ID",
        "SNR",
    )
    spectra = []
    for i in range(N):
        flux = u.Quantity(flux_value[i], unit=flux_unit)
        e_flux = StdDevUncertainty(array=image[hdu].data["E_FLUX"][i])

        meta = common_meta.copy()
        use_keys = []
        values = []
        for key in keys:
            if isinstance(key, tuple):
                use_keys.append(key[0])
                for k in key:
                    if k in image[hdu].data.dtype.names:
                        values.append(image[hdu].data[k][i])
                        break
                else:
                    values.append(None)
            else:
                use_keys.append(key)
                values.append(image[hdu].data[key][i] if key in image[hdu].data.dtype.names else None)


        meta.update(dict(zip(use_keys, values)))
        meta["BITMASK"] = pixel_bitmask[i]
        spectra.append(
            Spectrum1D(
                spectral_axis=spectral_axis, 
                flux=flux, 
                uncertainty=e_flux, 
                meta=meta
            )
        )
    return spectra


def _load_mwmVisit_or_mwmStar_hdu(image, hdu, **kwargs):
    if image[hdu].header["DATASUM"] == "0":
        # TODO: What should we return?
        return None

    flux_unit = u.Unit("1e-17 erg / (Angstrom cm2 s)")  # TODO
    try:
        wavelength = np.array(image[hdu].data["LAMBDA"])[0]
    except:
        wavelength = _wcs_log_linear(
            image[hdu].header["NPIXELS"],
            image[hdu].header["CDELT"],
            image[hdu].header["CRVAL"],
        )
    finally:
        spectral_axis = u.Quantity(wavelength, unit=u.Angstrom)

    flux = u.Quantity(image[hdu].data["FLUX"], unit=flux_unit)
    e_flux = StdDevUncertainty(array=image[hdu].data["E_FLUX"])

    # TODO: Read in other quantities from the binary table....
    meta = OrderedDict([])
    for hdu_idx in (0, hdu):
        for key in image[hdu_idx].header.keys():
            if key.startswith(("TTYPE", "TFORM", "TDIM")) or key in (
                "",
                "COMMENT",
                "CHECKSUM",
                "DATASUM",
                "NAXIS",
                "NAXIS1",
                "NAXIS2",
                "XTENSION",
                "BITPIX",
                "PCOUNT",
                "GCOUNT",
                "TFIELDS",
            ):
                continue
            meta[key] = image[hdu_idx].header[key]

    # Add bitmask
    meta["BITMASK"] = np.array(image[hdu].data["BITMASK"])
    try:
        meta["SNR"] = np.array(image[hdu].data["SNR"])
    except KeyError:
        # Early versions of mwmStar had this differently.
        # TODO: Remove this later on.
        meta["SNR"] = np.array([image[hdu].header["SNR"]])

    # If it's a mwmVisit, then try to load in parent data product identifiers
    try:
        meta["DATA_PRODUCT_ID"] = image[hdu].data["DATA_PRODUCT_ID"]
    except:
        meta["DATA_PRODUCT_ID"] = []

    return Spectrum1D(
        spectral_axis=spectral_axis, flux=flux, uncertainty=e_flux, meta=meta
    )
