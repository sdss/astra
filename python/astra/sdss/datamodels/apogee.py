"""Create HDUs in mwmVisit/mwmStar products with APOGEE spectra."""

import numpy as np
import pickle
from astropy.io import fits
from functools import partial
from typing import Union, List, Callable, Optional, Dict, Tuple
from collections import OrderedDict

from astra.database.astradb import DataProduct
from astra.utils import flatten, expand_path, log, list_to_dict, dict_to_list
from astra.sdss.datamodels import base, combine, util
from astra.tools.spectrum import Spectrum1D
from astra.sdss.bitmasks.apogee_drp import StarBitMask, PixelBitMask
from astropy import units as u
from astropy.nddata import StdDevUncertainty
from sdss_access import SDSSPath

from astra.tools.continuum.nmf import Emulator

# TODO: move to tools?
finite_or_empty = lambda _: _ if np.isfinite(_) else ""


def create_apogee_hdus(
    data_products: List[DataProduct],
    crval: float = 4.179,
    cdelt: float = 6e-6,
    num_pixels: int = 8_575,
    num_pixels_per_resolution_element=(5, 4.25, 3.5),
    median_filter_size: int = 501,
    median_filter_mode: str = "reflect",
    gaussian_filter_size: int = 100,
    scale_by_pseudo_continuum: bool = False,
    **kwargs,
) -> Tuple[fits.BinTableHDU]:        
    """
    Create HDUs for resampled APOGEE visits and HDUs for stacked spectra (both separated by observatory).

    :param data_products:
        A list of SDSS-V ApVisit data products, or SDSS-IV apStar data products.

    :param crval: [optional]
        The log10(lambda) of the wavelength of the first pixel to resample to.

    :param cdelt: [optional]
        The log (base 10) of the wavelength spacing to use when resampling.

    :param num_pixels: [optional]
        The number of pixels to use for the resampled array.

    :param num_pixels_per_resolution_element: [optional]
        The number of pixels per resolution element assumed when performing sinc interpolation.

        If a list-like is given, this is assumed to be a value per chip.

    :param observatory: [optional]
        Short name for the observatory where the data products originated from. If `None` is given, this will
        be inferred from the data model keywords.    
    """

    instrument = "APOGEE"
    default_values = dict(prefix="")

    sdss5_apVisit, sdss4_apStar, ignored = ([], [], [])
    for data_product in data_products:
        if data_product.release == "sdss5" and data_product.filetype == "apVisit":
            sdss5_apVisit.append(data_product)
        elif data_product.release == "dr17" and data_product.filetype == "apStar":
            sdss4_apStar.append(data_product)
        else:
            ignored.append(data_product)

    if ignored:
        log.warning(f"Ignoring {len(ignored)} data products of incorrect type: {ignored}")

    if len(sdss4_apStar) > 2:
        """
        Examples:
        {'obj': '2M16294628+3914104', 'apred': 'dr17', 'field': '062+44_MGA', 'apstar': 'stars', 'prefix': 'ap', 'telescope': 'apo25m'} catalogid=5191805335
        {'obj': '2M16294628+3914104', 'apred': 'dr17', 'field': '064+44_MGA', 'apstar': 'stars', 'prefix': 'ap', 'telescope': 'apo25m'} catalogid=5191805335
        {'obj': '2M16294628+3914104', 'apred': 'dr17', 'field': '063+44_MGA', 'apstar': 'stars', 'prefix': 'ap', 'telescope': 'apo25m'} catalogid=5191805335
        {'obj': '2M16294628+3914104', 'apred': 'dr17', 'field': '063+43_MGA', 'apstar': 'stars', 'prefix': 'ap', 'telescope': 'apo25m'} catalogid=5191805335
        {'obj': '2M16294628+3914104', 'apred': 'dr17', 'field': '062+43_MGA', 'apstar': 'stars', 'prefix': 'ap', 'telescope': 'apo25m'} catalogid=5191805335

        {'obj': '2M16083340+2829509', 'apred': 'dr17', 'field': '046+47_MGA', 'apstar': 'stars', 'prefix': 'ap', 'telescope': 'apo25m'} catalogid=5188086007
        {'obj': '2M16083340+2829509', 'apred': 'dr17', 'field': '046+48', 'apstar': 'stars', 'prefix': 'ap', 'telescope': 'apo25m'} catalogid=5188086007
        {'obj': '2M16083340+2829509', 'apred': 'dr17', 'field': '046+48_MGA', 'apstar': 'stars', 'prefix': 'ap', 'telescope': 'apo25m'} catalogid=5188086007
        """
        log.warning(f"Expected at most two SDSS-IV apStar data products (LCO/APO). Found {len(sdss4_apStar)}: {sdss4_apStar}")
        log.warning(f"Only taking one apStar data product per observatory.")
        sdss4_apStar = list({ dp.kwargs["telescope"]: dp for dp in sdss4_apStar }.values())

    # Do SDSS4 data first.
    sdss4_visit_data, sdss4_star_data, input_data_model_keys = ({}, {}, [])
    for i, data_product in enumerate(sdss4_apStar):
        visit, star, sdss4_keys = _format_apogee_data_from_dr17_apStar(data_product)
        # TODO: This will lump apo-1m data with apo-25m data because we don't have a mwmVisit/mwmStar HDU for apo-1m. 
        #       Should experiment to see what the consequences are.
        observatory = get_observatory(data_product.kwargs["telescope"])
        sdss4_visit_data[observatory] = visit
        sdss4_star_data[observatory] = star
        input_data_model_keys.extend(sdss4_keys)

    # Do SDSS5 data.
    sampling_kwargs = dict(
        num_pixels_per_resolution_element=num_pixels_per_resolution_element,
        median_filter_size=median_filter_size,
        median_filter_mode=median_filter_mode,
        gaussian_filter_size=gaussian_filter_size,
        scale_by_pseudo_continuum=scale_by_pseudo_continuum,
    )

    sdss5_visit_data, sdss5_keys = _format_apogee_data_from_sdssv_apVisits(sdss5_apVisit, **sampling_kwargs)
    if sdss5_keys is not None:
        input_data_model_keys.extend(sdss5_keys)

    # Unique input_data_model_keys but keep order
    unique_input_data_model_keys = list(dict.fromkeys(input_data_model_keys))

    visit_hdu_header_groups = OrderedDict([
        ("Spectral Data", ["snr", "flux", "e_flux", "bitmask", "continuum"]), # TODO: continuum
        ("Data Product Keywords", ["release", "filetype"] + unique_input_data_model_keys),
        ("Observing Conditions", ["date-obs", "exptime", "fluxflam", "npairs", "dithered", "fps"]),
        ("Continuum Fitting", ["continuum_theta"]),
        ("Radial Velocities (Doppler)", ["jd", "v_rad", "e_v_rad", "v_rel", "v_bc", "rchisq"]),
        ("Radial Velocities (Cross-Correlation)", ["v_rad_xcorr", "v_rel_xcorr", "e_v_rad_xcorr", "n_rv_components"]),
        ("Spectrum Sampling and Stacking", ["v_shift", "starflag", "in_stack"]),
        ("Database Primary Keys", ["visit_pk", "rv_visit_pk", "data_product_id"]),
    ])
    star_hdu_header_groups = OrderedDict([
        ("Spectral Data", ["snr", "lambda", "flux", "e_flux", "bitmask"]),
        ("Continuum Fitting", ["model_flux", "continuum", "continuum_theta", "continuum_phi", "continuum_rchisq", "continuum_success"]),
    ])
    visit_hdu_keys = flatten(visit_hdu_header_groups.values())
    star_hdu_keys = flatten(star_hdu_header_groups.values())

    # TODO: Need to do continuum fitting

    # Prepare common cards.    
    wavelength = util.log_lambda_dispersion(crval, cdelt, num_pixels)
    wavelength_cards = base.wavelength_cards(crval=crval, cdelt=cdelt, num_pixels=num_pixels)
    spectrum_sampling_cards = base.spectrum_sampling_cards(**sampling_kwargs)

    # Prepare cards for visits and stars per observatory
    observatories = list(set(flatten([d.keys() for d in (sdss4_visit_data, sdss5_visit_data)])))

    # TODO: I don't like hard-coding in paths, but here we are.
    with open(expand_path("$MWM_ASTRA/component_data/continuum/sgGK_200921nlte_nmf_components.pkl"), "rb") as fp:
        components = pickle.load(fp)

    with open(expand_path("$MWM_ASTRA/component_data/continuum/20230222_sky_mask_ivar_scalar.pkl"), "rb") as f:
        ivar_scalar = pickle.load(f)

    emulator = Emulator(
        components,
        regions=[
            (15_100.0, 15_800.0), 
            (15_840.0, 16_417.0), 
            (16_500.0, 17_000.0)
        ],   
        mask=(ivar_scalar != 1)     
    )
    
    fit_continuum_to_visit = {}
    visit_cards, visit_data = ({}, {})
    star_cards, star_data = ({}, {})
    for observatory in observatories:

        observatory_cards = base.metadata_cards(observatory, instrument)
        # Since the data could come from multiple data reduction pipeline versions, there is no
        # "common" thing here. Instead we will give V_APREDS.
        
        # NOTE: This is only true for the visit HDU. The star HDU will only stack data from a single pipeline version.
        v_apreds = []
        if observatory in sdss5_visit_data:
            v_apreds.extend(sdss5_visit_data[observatory]["v_apred"])
        if observatory in sdss4_visit_data:
            v_apreds.extend(sdss4_visit_data[observatory]["v_apred"])
        v_apreds = list(dict.fromkeys(v_apreds))

        drp_cards_visit = [
            base.BLANK_CARD,
            (" ", f"{instrument} DATA REDUCTION PIPELINE", None),
            ("V_APRED", ",".join(v_apreds), "DRP version(s)")
        ]
        drp_cards_star = [
            base.BLANK_CARD,
            (" ", f"{instrument} DATA REDUCTION PIPELINE", None),
            ("V_APRED", v_apreds[0], "DRP version(s)")
        ]        

        # Doppler stellar parameters do vary by observatory.
        # Where available we will take SDSS-V values.
        # TODO: Should we do this? Or should we make DOPPLER values a per-visit value (same for all SDSS-V and same for all SDSS-IV)
        if observatory in sdss5_visit_data:        
            doppler_data = sdss5_visit_data[observatory]
            doppler_release = "SDSS-V"
        else:
            doppler_data = sdss4_visit_data[observatory]
            doppler_release = "SDSS-IV"

        doppler_cards = [
            base.BLANK_CARD,
            (" ", f"DOPPLER STELLAR PARAMETERS ({doppler_release})", None),
            *[
                (f"{k.upper()}_D", finite_or_empty(np.round(doppler_data[f"{k}_doppler"][0], 1)))
                for k in ("teff", "e_teff")
            ],
            *[
                (f"{k.upper()}_D", finite_or_empty(np.round(doppler_data[f"{k}_doppler"][0], 3)))
                for k in ("logg", "e_logg")
            ],
            *[
                (f"{k.upper()}_D", finite_or_empty(np.round(doppler_data[f"{k}_doppler"][0], 3)))
                for k in ("feh", "e_feh")
            ],
        ]

        # Combine data from SDSS5 and SDSS4.
        has_sdss5_data = (observatory in sdss5_visit_data)
        has_sdss4_data = (observatory in sdss4_visit_data)
        if has_sdss4_data and has_sdss5_data:
            missing_keys = set(sdss5_visit_data[observatory]).symmetric_difference(sdss4_visit_data[observatory])
        elif has_sdss5_data:
            missing_keys = ["prefix"] # TODO: it's a bit hacky doing it this way
        else:
            missing_keys = []
        
        visit_observatory_data = {}
        for data in (sdss5_visit_data, sdss4_visit_data):
            if observatory in data:
                for k, v in data[observatory].items():
                    visit_observatory_data.setdefault(k, [])
                    visit_observatory_data[k].extend(v)
                
                for k in missing_keys:
                    if k not in data[observatory]:
                        N, P = np.atleast_2d(data[observatory]["flux"]).shape
                        visit_observatory_data.setdefault(k, [])
                        visit_observatory_data[k].extend([default_values[k]] * N)
        
        # Order by julian date.
        ordered = np.argsort(visit_observatory_data["jd"])
        visit_observatory_data = { k: np.array(v)[ordered] for k, v in visit_observatory_data.items()}

        # Add placeholder for continuum
        N, P = np.atleast_2d(visit_observatory_data["flux"]).shape
        visit_observatory_data.update(
            continuum=np.ones((N, P), dtype=float),
            continuum_theta=np.zeros((N, emulator.theta_size), dtype=float)
        )

        # Define what goes into the stack.
        starmask = StarBitMask()
        starflag = np.uint64(visit_observatory_data["starflag"])
        in_stack = (
                np.isfinite(visit_observatory_data["snr"]) 
            & (visit_observatory_data["snr"] > 10) 
            & np.isfinite(visit_observatory_data["v_rad"])   
            & ((starflag & starmask.bad_value) == 0)
            & ((starflag & starmask.get_value("RV_REJECT")) == 0)
        )
        # We will fit continuum to anything that has a reasonable spectrum.
        # We can't just use "in-stack" for this, because good SDSS4 visits are excldued from the stack if we have SDSS5 data. 
        fit_continuum_to_visit[observatory] = in_stack.copy()

        # If we don't have any SDSS-V data, then we will use the star data from SDSS-IV.
        # But if we have any useful SDSS-V data, then we will use that for stacking and ignore the SDSS-IV data.
        is_sdss4_data = (visit_observatory_data["release"] == "dr17") # TODO: better way to distinguish this?
        is_sdss5_data = (visit_observatory_data["release"] == "sdss5")
        if any(in_stack * is_sdss5_data) and has_sdss4_data:
            # Mark any sdss4 data as not being used in the stack.
            in_stack[is_sdss4_data] = False
            log.info(f"setting {np.sum(is_sdss4_data)} SDSS-IV visits to not be used in {observatory} stack.")

        visit_observatory_data["in_stack"] = in_stack
        
        visit_cards[observatory] = [
            *observatory_cards,
            *drp_cards_visit,
            *spectrum_sampling_cards,
            *doppler_cards,
            *wavelength_cards,
            base.FILLER_CARD
        ]

        visit_columns = []
        for name in visit_hdu_keys:
            visit_columns.append(
                fits.Column(
                    name=name.upper(),
                    array=visit_observatory_data[name], 
                    unit=None,
                    **base.fits_column_kwargs(visit_observatory_data[name])
                )
            )
            
        visit_data[observatory] = visit_columns

        # If we only have SDSS-IV data, just use the stack.
        if list(set(visit_observatory_data["release"][in_stack])) == ["dr17"]:
            stacked_observatory_data = sdss4_star_data[observatory]

        elif list(set(visit_observatory_data["release"][in_stack])) == ["sdss5"]:
            (
                combined_flux,
                combined_e_flux,
                combined_bitmask,
                pseudo_continuum,
                meta_combine,
            ) = combine.pixel_weighted_spectrum(
                visit_observatory_data["flux"][in_stack], 
                visit_observatory_data["e_flux"][in_stack], 
                visit_observatory_data["bitmask"][in_stack], 
                **kwargs
            )

            stacked_observatory_data = dict(
                flux=combined_flux,
                e_flux=combined_e_flux,
                bitmask=combined_bitmask,
            )

        else:
            # No data wll be stacked for this obervatory
            continue

        dithered = visit_observatory_data["dithered"]
        star_visits = np.sum(in_stack)
        star_dithered = np.round(np.sum(dithered[in_stack])/star_visits, 1)
        star_fps_fraction = np.round(np.sum(visit_observatory_data["fps"][in_stack])/star_visits, 1)
        assert np.isfinite(star_visits)
        assert np.isfinite(star_dithered)
        star_cards[observatory] = [
            *observatory_cards,
            *drp_cards_star,
            *spectrum_sampling_cards,

            # add dithered / nvisits
            ("DITHERED", star_dithered),
            ("NVISITS", star_visits),
            ("FPS", star_fps_fraction),

            *doppler_cards,
            *wavelength_cards,
            base.FILLER_CARD
        ]

        shape = (1, -1)
        stacked_observatory_data.update(
            {
                "snr": util.calculate_snr(
                    stacked_observatory_data["flux"],
                    stacked_observatory_data["e_flux"], 
                    axis=None
                ),
                "lambda": wavelength,
                "model_flux": np.zeros(P, dtype=float),
                "continuum": np.zeros(P, dtype=float),
                "continuum_phi": np.zeros(emulator.phi_size, dtype=float),
                "continuum_theta": np.zeros(emulator.theta_size, dtype=float),
                "continuum_rchisq": np.array([0.0]),
                #"continuum_warnings": np.array([0]),
                "continuum_success": np.array([False])
            }
        )
        star_columns = []
        for name in star_hdu_keys:
            array = stacked_observatory_data[name].reshape(shape)
            star_columns.append(
                fits.Column(
                    name=name.upper(),
                    array=array,
                    unit=None,
                    **base.fits_column_kwargs(array)
                )
            )
        star_data[observatory] = star_columns

    visit_hdus = {}
    star_hdus = {}
    visit_category_headers = [(v[0].upper(), k.upper()) for k, v in visit_hdu_header_groups.items()]
    star_category_headers = [(v[0].upper(), k.upper()) for k, v in star_hdu_header_groups.items()]

    for observatory in ("APO", "LCO"):
        if observatory in visit_data:
            visit_hdu = fits.BinTableHDU.from_columns(
                visit_data[observatory],
                header=fits.Header(visit_cards[observatory])
            )
            base.add_table_category_headers(visit_hdu, visit_category_headers)
            base.add_glossary_comments(visit_hdu)
            visit_hdus[observatory] = visit_hdu

            if observatory in star_data:                
                star_hdu = fits.BinTableHDU.from_columns(
                    star_data[observatory],
                    header=fits.Header(star_cards[observatory])
                )
                base.add_table_category_headers(star_hdu, star_category_headers)
                base.add_glossary_comments(star_hdu)

                star_hdus[observatory] = star_hdu
            else:
                star_hdus[observatory] = base.create_empty_hdu(observatory, instrument)

        else:
            visit_hdus[observatory] = base.create_empty_hdu(observatory, instrument)
            star_hdus[observatory] = base.create_empty_hdu(observatory, instrument)

    # Now that everything is in the right place, do continuum fitting.
    flux, e_flux = ([], [])
    for observatory, mask in fit_continuum_to_visit.items():
        flux.extend(visit_hdus[observatory].data["FLUX"][mask])
        e_flux.extend(visit_hdus[observatory].data["E_FLUX"][mask])

    if len(flux) > 0:
        flux_unit = u.Unit("1e-17 erg / (Angstrom cm2 s)")  # TODO

        spectrum = Spectrum1D(
            spectral_axis=u.Quantity(wavelength, unit=u.Angstrom),
            flux=np.array(flux) * flux_unit,
            uncertainty=StdDevUncertainty(np.array(e_flux) * flux_unit),
        )

        phi, theta, continuum, model_rectified_flux, meta = emulator.fit(spectrum)

        n_warnings = np.sum(np.diff(meta["chi_sqs"]) > 0)

        # Set the continuum_phi values for each star HDU
        for observatory, star_hdu in star_hdus.items():
            if len(star_hdu.data) == 0: continue
            star_hdu.data["MODEL_FLUX"][0] = model_rectified_flux
            star_hdu.data["CONTINUUM_PHI"][0] = phi
            star_hdu.data["CONTINUUM_RCHISQ"][0] = np.min(meta["reduced_chi_sqs"])
            star_hdu.data["CONTINUUM_SUCCESS"][0] = meta["success"]
            #star_hdu.data["CONTINUUM_WARNINGS"][0] = n_warnings

            # Now estimate the continuum for the stacked flux.
            flux = star_hdu.data["FLUX"].copy()
            ivar = star_hdu.data["E_FLUX"].copy()**-2

            bad_pixels = ~np.isfinite(ivar) | ~np.isfinite(flux) | (ivar == 0)
            flux[bad_pixels] = 0
            ivar[bad_pixels] = 0

            star_theta, star_continuum = emulator._maximization(
                flux / model_rectified_flux,
                model_rectified_flux * ivar * model_rectified_flux,
                meta["continuum_args"]
            )
            # star_theta and star_continuum will have shape (N_visits, P),
            # but we only want the first one
            star_hdu.data["CONTINUUM_THETA"][:] = star_theta[0].flatten()
            star_hdu.data["CONTINUUM"][:] = star_continuum[0]

        # Set the continuum_theta values for each HDU
        si, ei = (0, 0)
        for observatory, mask in fit_continuum_to_visit.items():
            K = np.sum(mask)
            if K == 0: continue
            ei += K
            visit_hdus[observatory].data["CONTINUUM_THETA"][mask] = theta[si:ei].reshape((K, -1))
            visit_hdus[observatory].data["CONTINUUM"][mask] = continuum[si:ei]
            si += K

    return (visit_hdus["APO"], visit_hdus["LCO"], star_hdus["APO"], star_hdus["LCO"])



def get_observatory(telescope):
    return telescope.upper()[:3]


def _format_apogee_data_from_dr17_apStar(data_product):
    """
    Extract spectra and metadata from an APOGEE DR17 apStar data product,
    such that it can be collated together into a mwmVisit/mwmStar data product.
    """

    sdss_path = SDSSPath("dr17")
    input_data_model_keys = sdss_path.lookup_keys("apVisit")

    telescope = data_product.kwargs["telescope"]

    #with fits.open(data_product.path) as image:
    image = fits.open(data_product.path)
    if True:
        flux = np.atleast_2d(image[1].data)
        e_flux = np.atleast_2d(image[2].data)
        bitmask = np.atleast_2d(image[3].data)

        N, P = flux.shape
        # What I was told by the APOGEE team is that the first two visits
        # are the stacked spectra (if there are more than 1 visit), and the
        # rest are visits. So that means there should only ever be either 1
        # spectrum (1 visit), or >= 4 spectra (e.g., 2 visit + 2 stack).
        
        # However, there are some cases where there are 2 "spectra" but only 1 visit. 
        # The second spectra are all NaNs.
        if N in (1, 2):
            N_visits = 1
            visit_mask = np.zeros(N, dtype=bool)
            visit_mask[0] = True
            stack_mask = visit_mask
            
        elif N in (0, 3):
            raise ValueError(f"Unexpected number of spectra ({N}) in data product {data_product}")
        
        else:
            N_visits = N - 2
            visit_mask = np.ones(N, dtype=bool)
            visit_mask[:2] = False
            stack_mask = np.zeros(N, dtype=bool)
            stack_mask[0] = True
        
        visit_keys = ["date-obs", "starflag", "fps"] + input_data_model_keys
        visit_meta = { k: [] for k in visit_keys }

        for i in range(1, 1 + N_visits):
            sfile = image[0].header[f"SFILE{i}"]
            _, apred, plate, mjd, fiber = sfile.split("-")
            fiber = fiber.split(".")[0]

            visit_meta["apred"].append(apred)
            visit_meta["plate"].append(plate)
            visit_meta["mjd"].append(int(mjd))
            visit_meta["fps"].append(0) # no fps in SDSS-IV
            visit_meta["prefix"].append(sfile[:2])
            visit_meta["fiber"].append(int(fiber))
            visit_meta["date-obs"].append(image[0].header[f"DATE{i}"])
            visit_meta["starflag"].append(np.uint64(image[0].header[f"FLAG{i}"]))

    visit_meta.update({
        "release": [data_product.release] * N_visits,
        "filetype": ["apVisit"] * N_visits,
        "telescope": [telescope] * N_visits,
        "field": [data_product.kwargs["field"]] * N_visits,
        "data_product_id": [data_product.id] * N_visits,
    })

    # Some information is in the SDSS-V apVisit files that isn't in the SDSS-IV apStar files.
    # Open the SDSS-V apVisit files to get what we need.
    visit_paths = [sdss_path.full(**kwds) for kwds in dict_to_list(visit_meta)]
    missing_keys = ("v_apred", "exptime", "fluxflam", "npairs")
    visit_meta.update({ k: [] for k in missing_keys })
    visit_meta["dithered"] = [] # determined separately.

    for i, visit_path in enumerate(visit_paths):
        with fits.open(visit_path) as image:
            dithered = 1.0 if image[1].data.size == (3 * 4096) else 0.0
            visit_meta["dithered"].append(dithered)
            for key in missing_keys:
                visit_meta[key].append(image[0].header[key.upper()])

    visit_meta.update(get_apogee_visit_radial_velocity_from_apStar(data_product))
    visit_meta.update(
        flux=flux[visit_mask],
        e_flux=e_flux[visit_mask],
        bitmask=bitmask[visit_mask],
        # Ignoring the SDSS-IV SNR value so that we consistently calcualte S/N between SDSS-4 and SDSS-5.
        snr=util.calculate_snr(flux[visit_mask], e_flux[visit_mask], axis=1)
    )
    star_meta = dict(
        flux=flux[stack_mask],
        e_flux=e_flux[stack_mask],
        bitmask=bitmask[stack_mask],
    )    
    return (visit_meta, star_meta, input_data_model_keys)


def _format_apogee_data_from_sdssv_apVisits(
    data_products, 
    crval: float = 4.179,
    cdelt: float = 6e-6,
    num_pixels: int = 8575,
    num_pixels_per_resolution_element=(5, 4.25, 3.5),
    **kwargs
) -> Tuple[dict, list]:

    D = len(data_products)
    keep = np.ones(D, dtype=bool)
    starflag = np.zeros(D, dtype=np.uint64)
    dithered = np.zeros(D, dtype=float)
    fps = np.zeros(D, dtype=int)
    for i, data_product in enumerate(data_products):
        try:    
            with fits.open(data_product.path) as image:
                starflag[i] = image[0].header["STARFLAG"]
                dithered[i] = 1.0 if image[1].data.size == (3 * 4096) else 0.0
                mjd = int(data_product.kwargs["mjd"])
                # Start of FPS ops according to https://wiki.sdss.org/display/IPL/Caveats+for+BHM+IPL-1
                fps[i] = int(int(mjd) >= 59635)
            
                if len(image) < 5:
                    log.exception(f"Data product {data_product} at {data_product.path} has unexpectedly few HDUs ({len(image)})")
                    keep[i] = False
        except:
            log.exception(f"OSError when loading {data_product}: {data_product.path}")
            keep[i] = False
            continue

    # Restrict ourselves to non-corrupted data products
    data_products = [dp for dp, keep_dp in zip(data_products, keep) if keep_dp]
    dithered = dithered[keep]
    starflag = starflag[keep]
    fps = fps[keep]

    if len(data_products) == 0:
        return ({}, [])

    input_data_model_keys = SDSSPath("sdss5").lookup_keys("apVisit")
    keys = ("date-obs", "exptime", "fluxflam", "npairs", "starflag", "v_apred")
    visit_meta = dict(
        release=[data_product.release for data_product in data_products],
        filetype=[data_product.filetype for data_product in data_products],
        dithered=dithered,
        fps=fps,
        data_product_id=[data_product.id for data_product in data_products],
    )
    visit_meta.update({ k: [] for k in keys })
    visit_meta.update({ k: [] for k in input_data_model_keys })

    for data_product in data_products:
        for k in input_data_model_keys:
            visit_meta[k].append(data_product.kwargs[k])

        with fits.open(data_product.path) as image:
            for k in keys:
                visit_meta[k].append(image[0].header[k.upper()])

    velocity_meta = list_to_dict(
        tuple(map(get_apogee_visit_radial_velocity, data_products))
    )

    # Set velocity to use when shifting and resampling to rest-frame.
    velocity_meta["v_shift"] = velocity_meta["v_rel"]

    flux, e_flux, bitmask, sampling_meta = resample_apogee_visit_spectra(
        data_products,
        crval=crval,
        cdelt=cdelt,
        num_pixels=num_pixels,
        num_pixels_per_resolution_element=num_pixels_per_resolution_element,
        radial_velocities=velocity_meta["v_shift"],
        **kwargs,
    )

    # Increase the flux uncertainties at the pixel level due to persistence and significant skylines,
    # in the same way that is done for apStar data products.
    increase_flux_uncertainties_due_to_persistence(e_flux, bitmask)
    increase_flux_uncertainties_due_to_skylines(e_flux, bitmask)

    visit_meta.update(velocity_meta)

    # Need to group by observatory.
    unique_observatories = list(map(get_observatory, set(visit_meta["telescope"])))
    observatory = np.array(list(map(get_observatory, visit_meta["telescope"])))

    visit_data = {}
    for unique_observatory in unique_observatories:
        mask = (observatory == unique_observatory)
        _visit_data = { k: np.array(v)[mask] for k, v in visit_meta.items() }
        _visit_data.update(
            flux=flux[mask],
            e_flux=e_flux[mask],
            bitmask=bitmask[mask],
            snr=util.calculate_snr(flux[mask], e_flux[mask], axis=1)
        )
        visit_data[unique_observatory] = _visit_data
    
    return (visit_data, input_data_model_keys)


def _no_rv_measurement():
    return {
        "v_bc": np.nan,
        "v_rel": 0,
        "v_rad": np.nan,
        "e_v_rel": np.nan,
        "e_v_rad": np.nan,
        "v_type": -1,
        "jd": -1,
        "date-obs": "",
        "teff_doppler": np.nan,
        "e_teff_doppler": np.nan,
        "logg_doppler": np.nan,
        "e_logg_doppler": np.nan,
        "feh_doppler": np.nan,
        "e_feh_doppler": np.nan,
        "visit_pk": -1,
        "rv_visit_pk": -1,
        "rchisq": np.nan,
        "n_rv_components": 0,
        "v_rel_xcorr": np.nan,
        "e_v_rad_xcorr": np.nan,
        "v_rad_xcorr": np.nan,
        "rv_components": np.array([np.nan, np.nan, np.nan]),
    }    


def get_apogee_visit_radial_velocity_from_apStar(data_product):
    with fits.open(data_product.path) as image:
        N_visits = image[0].header["NVISITS"]

        get_visit_values = lambda key: [image[0].header[f"{key}{i}"] for i in range(1, 1 + N_visits)]

        meta = {
            "v_bc": get_visit_values("BC"),
            "v_rel": get_visit_values("VHELIO"),
            "v_shift": get_visit_values("VHELIO"), # TODO: is this right? how do we get what shift value was actually used?
            "v_rad": get_visit_values("VRAD"),
            "e_v_rel": get_visit_values("VERR"),
            "e_v_rad": get_visit_values("VERR"),
            "v_type": [2] * N_visits,
            "jd": get_visit_values("JD"),
            "date-obs": get_visit_values("DATE"),
            "teff_doppler": image[-2].data["teff"],
            "e_teff_doppler": image[-2].data["tefferr"],
            "logg_doppler": image[-2].data["logg"],
            "e_logg_doppler": image[-2].data["loggerr"],
            "feh_doppler": image[-2].data["feh"],
            "e_feh_doppler": image[-2].data["feherr"],
            "visit_pk": [-1] * N_visits,
            "rv_visit_pk": [-1] * N_visits,
            "rchisq": image[-2].data["chisq"],
            "n_rv_components": np.ones(N_visits),
            "v_rel_xcorr": image[-2].data["xcorr_vrel"],
            "e_v_rad_xcorr": image[-2].data["xcorr_vrelerr"],
            "v_rad_xcorr": image[-2].data["xcorr_vhelio"],
            "rv_components": np.zeros((N_visits, 3)),
        }

    return meta


def get_apogee_visit_radial_velocity(data_product: DataProduct) -> dict:
    """
    Return the (current) best-estimate of the radial velocity (in km/s) of this
    visit spectrum that is stored in the APOGEE DRP database.

    :param image:
        The FITS image of the ApVisit data product.

    :param visit:
        The supplied visit.
    """
    from astra.database.apogee_drpdb import RvVisit, Visit

    if data_product.filetype.lower() != "apvisit":
        raise ValueError(f"Data product must have filetype 'apVisit'")

    # Note: The rv_visit table contains *NEARLY* everything to uniquely identify
    #       a visit. It contains mjd, apred, fiber (fibreid), telescope, plate,
    #       but does not contain FIELD. So we must cross-match with the visit table.

    q = (
        RvVisit.select()
        .join(Visit, on=(RvVisit.visit_pk == Visit.pk))
        .where(
            (Visit.telescope == data_product.kwargs["telescope"])
        &   (Visit.fiberid == data_product.kwargs["fiber"])
        &   (Visit.mjd == data_product.kwargs["mjd"])
        &   (Visit.field == data_product.kwargs["field"])
        &   (Visit.plate == data_product.kwargs["plate"])
        &   (Visit.apred == data_product.kwargs["apred"])
        # TODO: Currently the tagged version might not have a RV, but the daily does.
        #       Should we take daily when tagged doesn't have an RV? A bit ugly..
        )
        .order_by(RvVisit.created.desc())
    )
    result = q.first()
    if result is None:
        log.warning(
            f"No entry found for data product {data_product} in apogee_drp.RvVisit table"
        )

        # No RV measurement for this visit.
        return _no_rv_measurement() 
    
    # Sanity check
    if data_product.source.catalogid != result.catalogid:
        # TODO: This could be because of different catalog identifiers between targeting versions.
        #       We should probably cross-match and check for this, but for now let's raise an error in all situations.
        raise ValueError(
            f"Data product {data_product} catalogid does not match record in APOGEE DRP "
            f"table ({data_product.sources[0].catalogid} != {result.catalogid}) "
            f"on APOGEE rv_visit.pk={result.pk} and visit.pk={result.visit_pk}"
        )

    # Return the named metadata we need, using keys from common glossary.
    return {
        "v_bc": result.bc,
        "v_rel": result.vrel,
        "v_rad": result.vrad, # formerly vheliobary
        "e_v_rel": result.vrelerr,
        "e_v_rad": result.vrelerr,
        "v_type": result.vtype,  # 1=chisq, 2=xcorr
        "jd": result.jd,
        "date-obs": result.dateobs,
        "teff_doppler": result.rv_teff,
        "e_teff_doppler": result.rv_tefferr,
        "logg_doppler": result.rv_logg,
        "e_logg_doppler": result.rv_loggerr,
        "feh_doppler": result.rv_feh,
        "e_feh_doppler": result.rv_feherr,
        "visit_pk": result.visit_pk,
        "rv_visit_pk": result.pk,
        "rchisq": result.chisq,
        "n_rv_components": result.n_components,
        "v_rel_xcorr": result.xcorr_vrel,
        "e_v_rad_xcorr": result.xcorr_vrelerr,
        "v_rad_xcorr": result.xcorr_vrad, # formerly vheliobary
        "rv_components": result.rv_components,
    }


def resample_apogee_visit_spectra(
    visits: List[DataProduct],
    crval: float,
    cdelt: float,
    num_pixels: int,
    num_pixels_per_resolution_element=(5, 4.25, 3.5),
    radial_velocities: Optional[Union[Callable, List[float]]] = None,
    use_smooth_filtered_spectrum_for_bad_pixels: bool = True,
    median_filter_size: int = 501,
    median_filter_mode: str = "reflect",
    gaussian_filter_size: float = 100,
    **kwargs,
):
    """
    Resample APOGEE visit spectra onto a common wavelength array.

    :param visits:
        A list of ApVisit data products, or their paths.

    :param crval: [optional]
        The log10(lambda) of the wavelength of the first pixel to resample to.

    :param cdelt: [optional]
        The log (base 10) of the wavelength spacing to use when resampling.

    :param num_pixels: [optional]
        The number of pixels to use for the resampled array.

    :param num_pixels_per_resolution_element: [optional]
        The number of pixels per resolution element assumed when performing sinc interpolation.
        If a tuple is given, then it is assumed the input visits are multi-dimensional (e.g., multiple
        chips) and a different number of pixels per resolution element should be used per chip.

    :param radial_velocities: [optional]
        Either a list of radial velocities (one per visit), or a callable that takes two arguments
        (the FITS image of the data product, and the input visit) and returns a radial velocity
        in units of km/s.

        If `None` is given then we take the most recent radial velocity measurement from the APOGEE DRP
        database.

    :param use_smooth_filtered_spectrum_for_bad_pixels: [optional]
        For any bad pixels (defined by the bad pixel mask) use a smooth filtered spectrum (a median
        filtered spectrum with a gaussian convolution) to fill in bad pixel values (default: True).

    :param median_filter_size: [optional]
        The filter width (in pixels) to use for any median filters (default: 501).

    :param median_filter_mode: [optional]
        The mode to use for any median filters (default: reflect).

    :param gaussian_filter_size: [optional]
        The filter size (in pixels) to use for any gaussian filter applied.

    :returns:
        A 7-length tuple containing:
            - a list of visits that were included (e.g., no problems finding the file)
            - radial velocities [km/s] used for stacking
            - array of shape (P, ) of resampled wavelengths
            - array of shape (V, P) containing the resampled flux values, where V is the number of visits
            - array of shape (V, P) containing the resampled flux error values
            - array of shape (V, P) containing the pseudo-continuum values
            - array of shape (V, P) containing the resampled bitmask values
    """

    pixel_mask = PixelBitMask()

    if radial_velocities is None:
        radial_velocities = get_apogee_visit_radial_velocity

    include_visits, wavelength, v_shift = ([], [], [])
    flux, e_flux = ([], [])
    bitmask, bad_pixel_mask = ([], [])

    for i, visit in enumerate(visits):
        path = visit.path if isinstance(visit, DataProduct) else visit

        with fits.open(path) as image:
            if callable(radial_velocities):
                v = radial_velocities(image, visit)
            else:
                v = radial_velocities[i]

            hdu_header, hdu_flux, hdu_e_flux, hdu_bitmask, hdu_wl, *_ = range(11)

            v_shift.append(v)
            wavelength.append(image[hdu_wl].data)
            flux.append(image[hdu_flux].data)
            e_flux.append(image[hdu_e_flux].data)
            # We resample the bitmask, and we provide a bad pixel mask.
            bitmask.append(image[hdu_bitmask].data)
            bad_pixel_mask.append((bitmask[-1] & pixel_mask.bad_value) > 0)
        
        include_visits.append(visit)

    resampled_wavelength = util.log_lambda_dispersion(crval, cdelt, num_pixels)
    args = (
        resampled_wavelength,
        num_pixels_per_resolution_element,
        v_shift,
        wavelength,
    )

    kwds = dict(
        use_smooth_filtered_spectrum_for_bad_pixels=use_smooth_filtered_spectrum_for_bad_pixels,
        bad_pixel_mask=bad_pixel_mask,
        median_filter_size=median_filter_size,
        median_filter_mode=median_filter_mode,
        gaussian_filter_size=gaussian_filter_size,
    )
    kwds.update(kwargs)

    (
        resampled_flux,
        resampled_e_flux,
        resampled_bitmask,
    ) = combine.resample_visit_spectra(*args, flux, e_flux, bitmask, **kwds)
    # TODO: have combine.resample_visit_spectra return this so we dont repeat ourselves
    meta = dict(
        crval=crval,
        cdelt=cdelt,
        num_pixels=num_pixels,
        wavelength=resampled_wavelength,
        v_shift=v_shift,
        num_pixels_per_resolution_element=num_pixels_per_resolution_element,
        median_filter_size=median_filter_size,
        median_filter_mode=median_filter_mode,
        gaussian_filter_size=gaussian_filter_size,
    )

    return (
        resampled_flux,
        resampled_e_flux,
        resampled_bitmask,
        meta,
    )


def increase_flux_uncertainties_due_to_persistence(
    resampled_e_flux, resampled_bitmask
) -> None:
    """
    Increase the pixel-level resampled flux uncertainties (in-place, no array copying) due to persistence flags in the resampled bitmask.

    This logic follows directly from what is performed when constructing the apStar files. See:
        https://github.com/sdss/apogee_drp/blob/73cfd3f7a7fbb15963ddd2190e24a15261fb07b1/python/apogee_drp/apred/rv.py#L780-L791
    """

    V, P = resampled_bitmask.shape
    pixel_mask = PixelBitMask()
    is_high = (resampled_bitmask & pixel_mask.get_value("PERSIST_HIGH")) > 0
    is_medium = (resampled_bitmask & pixel_mask.get_value("PERSIST_MED")) > 0
    is_low = (resampled_bitmask & pixel_mask.get_value("PERSIST_LOW")) > 0

    resampled_e_flux[is_high] *= np.sqrt(5)
    resampled_e_flux[is_medium & ~is_high] *= np.sqrt(4)
    resampled_e_flux[is_low & ~is_medium & ~is_high] *= np.sqrt(3)
    return None


def increase_flux_uncertainties_due_to_skylines(
    resampled_e_flux, resampled_bitmask
) -> None:
    """
    Increase the pixel-level resampled flux uncertainties (in-place; no array copying) due to significant skylines.

    This logic follows directly from what is performed when constructing the apStar files. See:
        https://github.com/sdss/apogee_drp/blob/73cfd3f7a7fbb15963ddd2190e24a15261fb07b1/python/apogee_drp/apred/rv.py#L780-L791
    """
    is_significant_skyline = (
        resampled_bitmask & PixelBitMask().get_value("SIG_SKYLINE")
    ) > 0
    resampled_e_flux[is_significant_skyline] *= np.sqrt(100)
    return None


if __name__ == "__main__":
    from astra.database.astradb import DataProduct, Source

    # Example source with data in SDSS-V and SDSS-IV
    source = Source.get(329915927)
    from astra.sdss.datamodels.mwm import create_mwm_hdus
    foo = create_mwm_hdus(
        source,
        run2d="v6_0_7",
        apred="1.0",
        apogee_release="sdss5",
        boss_release="sdss5"
    )


    #data_products = list(source.data_products)
    #create_apogee_hdus(data_products)
    