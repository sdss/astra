

import astropy.units as u
from collections import OrderedDict
import numpy as np
from astropy.io import fits
from astropy.nddata import InverseVariance

from specutils import Spectrum1D, SpectrumList
from specutils.io.registers import data_loader, get_loaders_by_extension, io_registry

from astra.utils.data_models import parse_data_model

# De-register some default readers (and identifiers) that cause ambiguity,
# and would actually fail if they were used.
ignore_loaders = ("tabular-fits", "APOGEE apVisit", "APOGEE apStar", 
                  "APOGEE aspcapStar", "SDSS-III/IV spec", "SDSS-I/II spSpec")
for data_format in set(ignore_loaders).intersection(get_loaders_by_extension("fits")):
    io_registry.unregister_identifier(data_format, Spectrum1D)
    io_registry.unregister_identifier(data_format, SpectrumList)


def _wcs_log_linear(header):
    return 10**(np.arange(header["NAXIS1"]) * header["CDELT1"] + header["CRVAL1"]) * u.Angstrom


def _is_sdss_data_model(path, data_model_name):
    try:
        return (data_model_name == parse_data_model(path, strict=False))
    except ValueError:
        return False


@data_loader("SDSS APOGEE apStar", 
             identifier=lambda o, *a, **k: _is_sdss_data_model(a[0], "apStar"),
             extensions=["fits"])
def load_sdss_apstar(path, **kwargs):
    r"""
    Read a spectrum from a path that is described by the SDSS apStar data model
    https://data.SDSS.org/datamodel/files/APOGEE_REDUX/APRED_VERS/APSTAR_VERS/TELESCOPE/LOCATION_ID/apStar.html

    :param path:
        The local path of the spectrum.

    :returns:
        A `specutils.Spectrum1D` object.
    """
    units = u.Unit("1e-17 erg / (Angstrom cm2 s)")

    with fits.open(path, **kwargs) as image:
        # Build spectral axis ourselves because specutils does not handle
        # log-linear transformations yet.
        spectral_axis = _wcs_log_linear(image[1].header)

        flux = np.atleast_2d(image[1].data) * units
        uncertainty = InverseVariance(image[2].data.reshape(flux.shape)**-2)

        verbosity = kwargs.get("verbosity", 0)
        meta = OrderedDict([
            ("header", image[0].header),
            ("mask", image[3].data.reshape(flux.shape))
        ])

        if verbosity > 1:
            meta["hdu_headers"] = [hdu.header for hdu in image]

        if verbosity > 2:
            meta.update(OrderedDict([
                ("sky_flux", image[4].data.reshape(flux.shape) * units),
                ("sky_error", image[5].data.reshape(flux.shape) * units),
                ("telluric_flux", image[6].data.reshape(flux.shape) * units),
                ("telluric_error", image[7].data.reshape(flux.shape) * units),
                ("lsf_coefficients", image[8].data),
                ("rv_ccf_structure", image[9].data)
            ]))

    return Spectrum1D(spectral_axis=spectral_axis, flux=flux, uncertainty=uncertainty, meta=meta)


@data_loader("SDSS APOGEE apVisit", 
             identifier=lambda o, *a, **k: _is_sdss_data_model(a[0], "apVisit"),
             extensions=["fits"], dtype=SpectrumList)
def load_sdss_apvisit(path, **kwargs):
    r"""
    Read a spectrum from a path that is described by the SDSS apVisit data model
    https://data.SDSS.org/datamodel/files/APOGEE_REDUX/APRED_VERS/TELESCOPE/PLATE_ID/MJD5/apVisit.html

    :param path:
        The local path of the spectrum.

    :returns:
        A `specutils.Spectrum1D` object.
    """
    units = u.Unit("1e-17 erg / (Angstrom cm2 s)")

    with fits.open(path, **kwargs) as image:
        image = fits.open(path, **kwargs)

        spectral_axis = image[4].data * u.Angstrom
        flux = np.atleast_2d(image[1].data) * units
        uncertainty = InverseVariance(image[2].data.reshape(flux.shape)**-2)

        S, P = spectral_axis.shape

        # Multiple spectra, so need to generate these and send back a SpectrumList
        common_meta = OrderedDict([
            ("header", image[0].header),
        ])
        verbosity = kwargs.get("verbosity", 0)
        if verbosity > 1:
            common_meta["hdu_headers"] = [hdu.header for hdu in image]

        structured_meta = OrderedDict([
            ("mask", image[3].data.reshape(flux.shape)),
        ])

        if verbosity > 2:
            structured_meta.update(OrderedDict([
                ("sky_flux", image[5].data.reshape(flux.shape) * units),
                ("sky_error", image[6].data.reshape(flux.shape) * units),
                ("telluric_flux", image[7].data.reshape(flux.shape) * units),
                ("telluric_error", image[8].data.reshape(flux.shape) * units),
                ("wavelength_coefficients", image[9].data),
                ("lsf_coefficients", image[10].data)
            ]))

        spectra = []
        for s in range(S):
            meta = common_meta.copy()
            for k in structured_meta.keys():
                meta[k] = structured_meta[k][s]                

            spectra.append(Spectrum1D(spectral_axis=spectral_axis[s],
                                      flux=flux[s], uncertainty=uncertainty[s], meta=meta))

    return spectra


@data_loader("SDSS BOSS spec",
             identifier=lambda o, *a, **k: _is_sdss_data_model(a[0], "spec"),
             extensions=["fits"])
def load_sdss_boss(path, hdu=1, **kwargs):
    r"""
    Read a spectrum from a path that is described by the SDSS BOSS 'spec' data model
    https://data.SDSS.org/datamodel/files/BOSS_SPECTRO_REDUX/RUN2D/spectra/PLATE4/spec.html

    :param path:
        The local path of the spectrum.

    :returns:
        A `specutils.Spectrum1D` object.
    """
    units = u.Unit("1e-17 erg / (Angstrom cm2 s)")
    
    with fits.open(path, **kwargs) as image:
    
        spectral_axis = 10**image[hdu].data["loglam"] * u.Angstrom

        flux = np.atleast_2d(image[hdu].data["flux"]) * units
        uncertainty = InverseVariance(image[hdu].data["ivar"].reshape(flux.shape))

        meta = OrderedDict([
            ("header", image[0].header),
            ("hdu_headers", [hdu.header for hdu in image]),
            ("mask", dict(and_mask=image[hdu].data["and_mask"].reshape(flux.shape), 
                           or_mask=image[hdu].data["or_mask"].reshape(flux.shape))),
            ("wavelength", image[hdu].data["wdisp"]),
            ("model", image[hdu].data["model"].reshape(flux.shape)),
        ])

    return Spectrum1D(spectral_axis=spectral_axis, flux=flux, uncertainty=uncertainty, meta=meta)


@data_loader("SDSS MaNGA MaStar",
             identifier=lambda o, *a, **k: _is_sdss_data_model(a[0], "MaStar"),
             dtype=SpectrumList, extensions=["fits"])
def load_sdss_mastar(path, hdu=1, **kwargs):
    r"""
    Read a list of spectrum from a path that is described by the SDSS MaNGA MaStar data model,
    which actually describes a collection of spectra of different sources:
    https://data.sdss.org/datamodel/files/MANGA_SPECTRO_MASTAR/DRPVER/MPROCVER/mastar-goodspec-DRPVER-MPROCVER.html
    
    :param path:
        The local path of the spectrum.

    :returns:
        A `specutils.Spectrum1D` object.
    """
    spectra = []

    units = u.Unit("1e-17 erg / (Angstrom cm2 s)")

    with fits.open(path, **kwargs) as image:

        _hdu = image[hdu]
        for i in range(_hdu.header["NAXIS2"]):

            meta = OrderedDict(zip(_hdu.data.dtype.names, _hdu.data[i]))

            spectral_axis = meta.pop("WAVE") * u.Angstrom
            flux = np.atleast_2d(meta.pop("FLUX") * units)
            uncertainty = InverseVariance(meta.pop("IVAR").reshape(flux.shape))

            spectra.append(Spectrum1D(spectral_axis=spectral_axis, 
                                      flux=flux, uncertainty=uncertainty, meta=meta))

    return SpectrumList(spectra)