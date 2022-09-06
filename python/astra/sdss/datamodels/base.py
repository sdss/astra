import datetime
import numpy as np
from typing import Union, List, Callable, Optional, Dict
from astropy.io import fits
from astra import log, __version__ as astra_version
from astra.database.astradb import Source


from peewee import Alias, JOIN, fn
from sdssdb.peewee.sdss5db import database as sdss5_database

sdss5_database.set_profile("operations")

from sdssdb.peewee.sdss5db.catalogdb import (
    Catalog,
    CatalogToTIC_v8,
    TIC_v8 as TIC,
    TwoMassPSC,
)
from sdssdb.peewee.sdss5db.targetdb import Target, CartonToTarget, Carton

try:
    from sdssdb.peewee.sdss5db.catalogdb import Gaia_DR3 as Gaia
except ImportError:
    from sdssdb.peewee.sdss5db.catalogdb import Gaia_DR2 as Gaia

    log.warning(
        f"Gaia DR3 not yet available in sdssdb.peewee.sdss5db.catalogdb. Using Gaia DR2."
    )


from healpy import ang2pix

BLANK_CARD = (" ", " ", None)
FILLER_CARD = (FILLER_CARD_KEY, *_) = ("TTYPE0", "Water cuggle", None)

GLOSSARY = {
    #           "*****************************************************"
    "INSTRMNT": "Instrument name",
    "OBSRVTRY": "Observatory name",
    "EXTNAME": "Short extension name",
    # Observing conditions
    "ALT": "Telescope altitude [deg]",
    "AZ": "Telescope azimuth [deg]",
    "EXPTIME": "Total exposure time [s]",
    "NEXP": "Number of exposures taken",
    "AIRMASS": "Mean airmass",
    "AIRTEMP": "Air temperature [C]",
    "DEWPOINT": "Dew point temperature [C]",
    "HUMIDITY": "Humidity [%]",
    "PRESSURE": "Air pressure [inch Hg?]",  # TODO
    "MOON_PHASE_MEAN": "Mean phase of the moon",
    "MOON_DIST_MEAN": "Mean sky distance to the moon [deg]",
    "SEEING": "Median seeing conditions [arcsecond]",
    "GUSTD": "Wind gust direction [deg]",
    "GUSTS": "Wind gust speed [km/s]",
    "WINDD": "Wind direction [deg]",
    "WINDS": "Wind speed [km/s]",
    "TAI-BEG": "MJD (TAI) at start of integrations [s]",
    "TAI-END": "MJD (TAI) at end of integrations [s]",
    "NGUIDE": "Number of guider frames during integration",
    # Stacking
    "V_HELIO": "Heliocentric velocity correction [km/s]",
    "V_SHIFT": "Relative velocity shift used in stack [km/s]",
    "IN_STACK": "Was this spectrum used in the stack?",
    # Metadata related to sinc interpolation and stacking
    "NRES": "Sinc bandlimit [pixel/resolution element]",
    "FILTSIZE": "Median filter size for pseudo-continuum [pixel]",
    "NORMSIZE": "Gaussian width for pseudo-continuum [pixel]",
    "CONSCALE": "Scale by pseudo-continuum when stacking",
    # BOSS data reduction pipeline
    "V_BOSS": "Version of the BOSS ICC",
    "VJAEGER": "Version of Jaeger",
    "VKAIJU": "Version of Kaiju",
    "VCOORDIO": "Version of coordIO",
    "VCALIBS": "Version of FPS calibrations",
    "VERSREAD": "Version of idlspec2d for processing raw data",
    "VERSIDL": "Version of IDL",
    "VERSUTIL": "Version of idlutils",
    "VERS2D": "Version of idlspec2d for 2D reduction",
    "VERSCOMB": "Version of idlspec2d for combining exposures",
    "VERSLOG": "Version of SPECLOG product",
    "VERSFLAT": "Version of SPECFLAT product",
    "DIDFLUSH": "Was CCD flushed before integration",
    "CARTID": "Cartridge identifier",
    "PSFSKY": "Order of PSF sky subtraction",
    "PREJECT": "Profile area rejection threshold",
    "LOWREJ": "Extraction: low rejection",
    "HIGHREJ": "Extraction: high rejection",
    "SCATPOLY": "Extraction: Order of scattered light polynomial",
    "PROFTYPE": "Extraction profile: 1=Gaussian",
    "NFITPOLY": "Extraction: Number of profile parameters",
    "RDNOISE0": "CCD read noise amp 0 [electrons]",
    "SKYCHI2": "Mean \chi^2 of sky subtraction",
    "SCHI2MIN": "Minimum \chi^2 of sky subtraction",
    "SCHI2MAX": "Maximum \chi^2 of sky subtraction",
    "ZWARNING": "See sdss.org/dr14/algorithms/bitmasks/#ZWARNING",
    "FIBER_OFFSET": "Position offset applied during observations",
    "DELTA_RA": "Offset in right ascension [arcsecond]",
    "DELTA_DEC": "Offset in declination [arcsecond]",
    # APOGEE data reduction pipeline
    "DATE-OBS": "Observation date (UTC)",
    "JD-MID": "Julian date at mid-point of visit",
    "UT-MID": "Date at mid-point of visit",
    "FLUXFLAM": "ADU to flux conversion factor [ergs/s/cm^2/A]",
    "NPAIRS": "Number of dither pairs combined",
    "DITHERED": "Fraction of visits that were dithered",
    "NVISITS": "Number of visits included in the stack",
    # XCSAO
    "V_HELIO_XCSAO": "Heliocentric velocity from XCSAO [km/s]",
    "E_V_HELIO_XCSAO": "Error in heliocentric velocity from XCSAO [km/s]",
    "RXC_XCSAO": "Cross-correlation R-value (1979AJ.....84.1511T)",
    "TEFF_XCSAO": "Effective temperature from XCSAO [K]",
    "E_TEFF_XCSAO": "Error in effective temperature from XCSAO [K]",
    "LOGG_XCSAO": "Surface gravity from XCSAO",
    "E_LOGG_XCSAO": "Error in surface gravity from XCSAO",
    "FEH_XCSAO": "Metallicity from XCSAO",
    "E_FEH_XCSAO": "Error in metallicity from XCSAO",
    # Data things
    "LAMBDA": "Source rest frame vacuum wavelength [Angstrom]",
    "FLUX": "Source flux",
    "E_FLUX": "Standard deviation of source flux",
    "SNR": "Mean signal-to-noise ratio",
    # Wavelength solution
    "CRVAL": "Log(10) wavelength of first pixel [Angstrom]",
    "CDELT": "Log(10) delta wavelength per pixel [Angstrom]",
    "CRPIX": "Pixel offset from the first pixel",
    "CTYPE": "Wavelength solution description",
    "CUNIT": "Wavelength solution unit",
    "DC-FLAG": "Wavelength solution flag",
    "RELEASE": "SDSS data release name",
    "FILETYPE": "SDSS data model filetype",
    # BOSS data model keywords
    "RUN2D": "Spectro-2D reduction name",
    "FIELDID": "Field identifier",
    "MJD": "Modified Julian Date of the observations",
    "CATALOGID": "SDSS-V catalog identifier",
    "ISPLATE": "Whether the data were taken with plates",
    # APOGEE data model keywords
    "FIBER": "Fiber number",
    "TELESCOPE": "Telescope name",
    "PLATE": "Plate number",
    "FIELD": "Field number",
    "APRED": "APOGEE reduction tag",
    # Radial velocity keys (common to any code)
    "V_RAD": "Radial velocity in Solar barycentric rest frame [km/s]",
    "E_V_RAD": "Error in radial velocity [km/s]",
    "V_BC": "Solar barycentric velocity applied [km/s]",
    "V_HC": "Heliocentric velocity applied [km/s]",
    "V_REL": "Relative velocity [km/s]",
    "E_V_REL": "Error in relative velocity [km/s]",
    "JD": "Julian date at mid-point of visit",
    "STARFLAG": "APOGEE DRP quality bit mask",
    "BITMASK": "Pixel-level bitmask (see documentation)",
    # Doppler keys
    "TEFF_D": "Effective temperature from DOPPLER [K]",
    "E_TEFF_D": "Error in effective temperature from DOPPLER [K]",
    "LOGG_D": "Surface gravity from DOPPLER",
    "E_LOGG_D": "Error in surface gravity from DOPPLER",
    "FEH_D": "Metallicity from DOPPLER",
    "E_FEH_D": "Error in metallicity from DOPPLER",
    "RCHISQ": "Reduced \chi-squared of model fit",
    "VISIT_PK": "Primary key in `apogee_drp.visit` table",
    "RV_VISIT_PK": "Primary key in `apogee_drp.rv_visit` table",
    "N_RV_COMPONENTS": "Number of detected RV components",
    "RV_COMPONENTS": "Relative velocity of detected components [km/s]",
    "V_REL_XCORR": "Relative velocity from XCORR [km/s]",
    "E_V_RAD_XCORR": "Error in relative velocity from XCORR [km/s]",
    "V_RAD_XCORR": "Radial velocity in Solar barycentric rest frame [km/s]",
}
for key, comment in GLOSSARY.items():
    if len(comment) > 80:
        log.warning(
            f"Glossary term {key} has a comment that is longer than 80 characters. "
            f"It will be truncated from:\n{comment}\nTo:\n{comment[:80]}"
        )


def get_catalog_identifier(source: Union[Source, int]):
    """
    Return a catalog identifer given either a source, or catalog identifier (as string or int).

    :param source:
        The astronomical source, or the SDSS-V catalog identifier.
    """
    return source.catalogid if isinstance(source, Source) else int(source)


def get_cartons_and_programs(source: Union[Source, int]):
    """
    Return the name of cartons and programs that this source is matched to.

    :param source:
        The astronomical source, or the SDSS-V catalog identifier.

    :returns:
        A two-length tuple containing a list of carton names (e.g., `mwm_snc_250pc`)
        and a list of program names (e.g., `mwm_snc`).
    """

    catalogid = get_catalog_identifier(source)

    sq = (
        Carton.select(Target.catalogid, Carton.carton, Carton.program)
        .distinct()
        .join(CartonToTarget)
        .join(Target)
        .where(Target.catalogid == catalogid)
        .alias("distinct_cartons")
    )

    q_cartons = (
        Target.select(
            Target.catalogid,
            fn.STRING_AGG(sq.c.carton, ",").alias("cartons"),
            fn.STRING_AGG(sq.c.program, ",").alias("programs"),
        )
        .join(sq, on=(sq.c.catalogid == Target.catalogid))
        .group_by(Target.catalogid)
        .tuples()
    )
    _, cartons, programs = q_cartons.first()
    cartons, programs = (cartons.split(","), programs.split(","))
    return (cartons, programs)


def get_first_carton(source: Union[Source, int]) -> Carton:
    """
    Return the first carton this source was assigned to.

    This is the carton with the lowest priority value in the targeting database,
    because fibers are assigned in order of priority (lowest numbers first).

    Note that this provides the "first carton" for the observations taken with the
    fiber positioning system (FPS), but in the 'plate era', it may not be 100% true.
    The situation in the plate era is a little more complex, in so much that the
    same source on two plates could have a different first carton assignment because
    of the fiber priority in that plate. Similarly, the first carton for a source in
    the plate era may not match the first carton for the same source in the FPS
    era. The targeting and robostrategy people know what is happening, but it is a
    little complex to put everything together.

    In summary, the first carton will be correct for the FPS-era data, but may be
    incorrect for the plate-era data.
    """

    catalogid = get_catalog_identifier(source)

    sq = (
        CartonToTarget.select(CartonToTarget.carton_pk)
        .join(Target)
        .where(Target.catalogid == catalogid)
        .order_by(CartonToTarget.priority.asc())
        .limit(1)
        .alias("first_carton")
    )
    return Carton.select().join(sq, on=(sq.c.carton_pk == Carton.pk)).first()


def get_auxiliary_source_data(source: Union[Source, int]):
    """
    Return auxiliary data (e.g., photometry) for a given SDSS-V source.

    :param source:
        The astronomical source, or the SDSS-V catalog identifier.
    """

    catalogid = get_catalog_identifier(source)
    tic_dr = TIC.__name__.split("_")[-1]
    gaia_dr = Gaia.__name__.split("_")[-1]

    ignore = lambda c: c is None or isinstance(c, str)

    # Define the columns and associated comments.
    field_descriptors = [
        BLANK_CARD,
        (" ", "IDENTIFIERS", None),
        ("SDSS_ID", Catalog.catalogid, f"SDSS-V catalog identifier"),
        ("TIC_ID", TIC.id.alias("tic_id"), f"TESS Input Catalog ({tic_dr}) identifier"),
        ("GAIA_ID", Gaia.source_id, f"Gaia {gaia_dr} source identifier"),
        BLANK_CARD,
        (" ", "ASTROMETRY", None),
        ("RA", Catalog.ra, "SDSS-V catalog right ascension (J2000) [deg]"),
        ("DEC", Catalog.dec, "SDSS-V catalog declination (J2000) [deg]"),
        ("GAIA_RA", Gaia.ra, f"Gaia {gaia_dr} right ascension [deg]"),
        ("GAIA_DEC", Gaia.dec, f"Gaia {gaia_dr} declination [deg]"),
        ("PLX", Gaia.parallax, f"Gaia {gaia_dr} parallax [mas]"),
        ("E_PLX", Gaia.parallax_error, f"Gaia {gaia_dr} parallax error [mas]"),
        ("PMRA", Gaia.pmra, f"Gaia {gaia_dr} proper motion in RA [mas/yr]"),
        (
            "E_PMRA",
            Gaia.pmra_error,
            f"Gaia {gaia_dr} proper motion in RA error [mas/yr]",
        ),
        ("PMDE", Gaia.pmdec, f"Gaia {gaia_dr} proper motion in DEC [mas/yr]"),
        (
            "E_PMDE",
            Gaia.pmdec_error,
            f"Gaia {gaia_dr} proper motion in DEC error [mas/yr]",
        ),
        ("VRAD", Gaia.radial_velocity, f"Gaia {gaia_dr} radial velocity [km/s]"),
        (
            "E_VRAD",
            Gaia.radial_velocity_error,
            f"Gaia {gaia_dr} radial velocity error [km/s]",
        ),
        BLANK_CARD,
        (" ", "PHOTOMETRY", None),
        (
            "G_MAG",
            Gaia.phot_g_mean_mag,
            f"Gaia {gaia_dr} mean apparent G magnitude [mag]",
        ),
        (
            "BP_MAG",
            Gaia.phot_bp_mean_mag,
            f"Gaia {gaia_dr} mean apparent BP magnitude [mag]",
        ),
        (
            "RP_MAG",
            Gaia.phot_rp_mean_mag,
            f"Gaia {gaia_dr} mean apparent RP magnitude [mag]",
        ),
        ("J_MAG", TwoMassPSC.j_m, f"2MASS mean apparent J magnitude [mag]"),
        ("E_J_MAG", TwoMassPSC.j_cmsig, f"2MASS mean apparent J magnitude error [mag]"),
        ("H_MAG", TwoMassPSC.h_m, f"2MASS mean apparent H magnitude [mag]"),
        ("E_H_MAG", TwoMassPSC.h_cmsig, f"2MASS mean apparent H magnitude error [mag]"),
        ("K_MAG", TwoMassPSC.k_m, f"2MASS mean apparent K magnitude [mag]"),
        ("E_K_MAG", TwoMassPSC.k_cmsig, f"2MASS mean apparent K magnitude error [mag]"),
    ]

    q = (
        Catalog.select(*[c for k, c, comment in field_descriptors if not ignore(c)])
        .distinct(Catalog.catalogid)
        .join(CatalogToTIC_v8, JOIN.LEFT_OUTER)
        .join(TIC)
        .join(Gaia, JOIN.LEFT_OUTER)
        .switch(TIC)
        .join(TwoMassPSC, JOIN.LEFT_OUTER)
        .where(Catalog.catalogid == catalogid)
        .dicts()
    )
    row = q.first()

    # Return as a list of entries suitable for a FITS header card.
    data = []
    for key, field, comment in field_descriptors:
        if ignore(field):
            data.append((key, field, comment))
        else:
            data.append(
                (
                    key,
                    row[field._alias if isinstance(field, Alias) else field.name],
                    comment,
                )
            )

    # Add carton and target information
    cartons, programs = get_cartons_and_programs(source)

    first_carton = get_first_carton(source)

    data.extend(
        [
            BLANK_CARD,
            (" ", "TARGETING", None),
            (
                "CARTON_0",
                first_carton.carton,
                f"First carton for source (see documentation)",
            ),
            ("CARTONS", ",".join(cartons), f"SDSS-V cartons"),
            ("PROGRAMS", ",".join(list(set(programs))), f"SDSS-V programs"),
            (
                "MAPPERS",
                ",".join(list(set([p.split("_")[0] for p in programs]))),
                f"SDSS-V mappers",
            ),
        ]
    )
    return data


def create_empty_hdu(observatory: str, instrument: str) -> fits.BinTableHDU:
    """
    Create an empty HDU to use as a filler.
    """
    cards = metadata_cards(observatory, instrument)
    cards.extend(
        [
            BLANK_CARD,
            (
                "COMMENT",
                f"No {instrument} data available from {observatory} for this source.",
            ),
        ]
    )
    return fits.BinTableHDU(
        header=fits.Header(cards),
    )


def metadata_cards(observatory: str, instrument: str) -> List:
    return [
        BLANK_CARD,
        (" ", "METADATA"),
        ("EXTNAME", f"{instrument}/{observatory}"),
        ("OBSRVTRY", observatory),
        ("INSTRMNT", instrument),
    ]


def spectrum_sampling_cards(
    num_pixels_per_resolution_element: Union[int, float],
    median_filter_size: Union[int, float],
    gaussian_filter_size: Union[int, float],
    scale_by_pseudo_continuum: bool,
    **kwargs,
) -> List:
    if isinstance(num_pixels_per_resolution_element, (float, int)):
        nres = f"{num_pixels_per_resolution_element}"
    else:
        nres = " ".join(list(map(str, num_pixels_per_resolution_element)))
    return [
        BLANK_CARD,
        (" ", "SPECTRUM SAMPLING AND STACKING"),
        ("NRES", nres),
        ("FILTSIZE", median_filter_size),
        ("NORMSIZE", gaussian_filter_size),
        ("CONSCALE", scale_by_pseudo_continuum),
    ]


def wavelength_cards(
    crval: Union[int, float], cdelt: Union[int, float], num_pixels: int, **kwargs
) -> List:
    return [
        BLANK_CARD,
        (" ", "WAVELENGTH INFORMATION (VACUUM)", None),
        ("CRVAL", crval, None),
        ("CDELT", cdelt, None),
        ("CTYPE", "LOG-LINEAR", None),
        ("CUNIT", "Angstrom (Vacuum)", None),
        ("CRPIX", 1, None),
        ("DC-FLAG", 1, None),
        ("NPIXELS", num_pixels, "Number of pixels per spectrum"),
    ]


def remove_filler_card(hdu):
    if FILLER_CARD_KEY is not None:
        try:
            del hdu.header[FILLER_CARD_KEY]
        except:
            None


def hdu_from_data_mappings(data_products, mappings, header):
    category_headers = []
    values = {}
    for j, data_product in enumerate(data_products):
        with fits.open(data_product.path) as image:
            for i, (key, function) in enumerate(mappings):
                if j == 0 and function is None:
                    category_headers.append((mappings[i + 1][0], key))
                else:
                    values.setdefault(key, [])
                    if callable(function):
                        try:
                            value = function(data_product, image)
                        except KeyError:
                            log.warning(f"No {key} found in {data_product.path}")
                            value = np.nan
                        values[key].append(value)
                    else:
                        values[key] = function

    columns = []
    for key, function in mappings:
        if function is None:
            continue
        columns.append(
            fits.Column(
                name=key,
                array=values[key],
                unit=None,
                **fits_column_kwargs(values[key]),
            )
        )
    hdu = fits.BinTableHDU.from_columns(
        columns,
        header=header,
        # name=f"{header['INSTRMNT']}/{header['OBSRVTRY']}"
    )

    add_table_category_headers(hdu, category_headers)
    add_glossary_comments(hdu)
    return hdu


def add_table_category_headers(hdu, category_headers):
    """
    Add comments to the HDU to categorise different headers.

    :param hdu:
        The FITS HDU to add the comments to.

    :param category_headers:
        A list of (`DATA_COLUMN_NAME`, `HEADER`) tuples, where the header
        comment `HEADER` will be added above the `DATA_COLUMN_NAME` column.
    """
    for dtype_name, category_header in category_headers:
        index = 1 + hdu.data.dtype.names.index(dtype_name)
        key = f"TTYPE{index}"
        hdu.header.insert(key, BLANK_CARD)
        hdu.header.insert(key, (" ", category_header))
    return None


def add_glossary_comments(hdu):
    for key in hdu.header.keys():
        hdu.header.comments[key] = GLOSSARY.get(key, None)
    for i, key in enumerate(hdu.data.dtype.names, start=1):
        hdu.header.comments[f"TTYPE{i}"] = GLOSSARY.get(key, None)
    remove_filler_card(hdu)
    return None


def headers_as_cards(data_product, input_header_keys):
    cards = []
    with fits.open(data_product.path) as image:
        for key in input_header_keys:
            if isinstance(key, tuple):
                old_key, new_key = key
                if new_key is None:
                    cards.append((None, None, None))
                    cards.append((" ", old_key, None))
                    continue
            else:
                old_key = new_key = key
            try:
                value = image[0].header[old_key]
                comment = image[0].header.comments[old_key]
            except KeyError:
                log.warning(f"No {old_key} header of HDU 0 in {data_product.path}")
                value = comment = None

            cards.append((new_key, value, GLOSSARY.get(new_key, comment)))
    return cards


def add_check_sums(hdu_list: fits.HDUList):
    """
    Add checksums to the HDU list.
    """
    for hdu in hdu_list:
        hdu.verify("fix")
        hdu.add_checksum()
        hdu.header.insert("CHECKSUM", BLANK_CARD)
        hdu.header.insert("CHECKSUM", (" ", "DATA INTEGRITY"))
        hdu.add_checksum()

    return None


def create_primary_hdu_cards(
    source: Union[Source, int],
    hdu_descriptions: Optional[List[str]] = None,
    nside: Optional[int] = 128,
) -> List:
    """
    Create primary HDU (headers only) for a Milky Way Mapper data product, given some source.

    :param source:
        The astronomical source, or the SDSS-V catalog identifier.

    :param hdu_descriptions: [optional]
        A list of strings describing all HDUs.

    :param nside: [optional]
        Number of sides used in Healpix (lon, lat) mapping (default: 128).
    """
    catalogid = get_catalog_identifier(source)

    # Sky position.
    ra, dec = (
        Catalog.select(Catalog.ra, Catalog.dec)
        .where(Catalog.catalogid == catalogid)
        .tuples()
        .first()
    )

    healpix = ang2pix(nside, ra, dec, lonlat=True)

    # I would like to use .isoformat(), but it is too long and makes headers look disorganised.
    # Even %Y-%m-%d %H:%M:%S is one character too long! ARGH!
    datetime_fmt = "%y-%m-%d %H:%M:%S"
    created = datetime.datetime.utcnow().strftime(datetime_fmt)

    cards = [
        BLANK_CARD,
        (" ", "METADATA", None),
        ("ASTRA", astra_version, f"Astra version"),
        ("CREATED", created, f"File creation time (UTC {datetime_fmt})"),
        ("HEALPIX", healpix, f"Healpix location ({nside} sides)"),
    ]
    # Get photometry and other auxiliary data.
    cards.extend(get_auxiliary_source_data(source))

    if hdu_descriptions is not None:
        cards.extend(
            [
                BLANK_CARD,
                (" ", "HDU DESCRIPTIONS", None),
                *[
                    (f"COMMENT", f"HDU {i}: {desc}", None)
                    for i, desc in enumerate(hdu_descriptions)
                ],
            ]
        )

    return cards


def fits_column_kwargs(values):
    if all(isinstance(v, str) for v in values):
        max_len = max(1, max(map(len, values)))
        return dict(format=f"{max_len}A")

    """
    FITS format code         Description                     8-bit bytes

    L                        logical (Boolean)               1
    X                        bit                             *
    B                        Unsigned byte                   1
    I                        16-bit integer                  2
    J                        32-bit integer                  4
    K                        64-bit integer                  8
    A                        character                       1
    E                        single precision float (32-bit) 4
    D                        double precision float (64-bit) 8
    C                        single precision complex        8
    M                        double precision complex        16
    P                        array descriptor                8
    Q                        array descriptor                16
    """

    mappings = [
        ("E", lambda v: isinstance(v[0], (float, np.floating))),  # all 32-bit
        (
            "K",
            lambda v: isinstance(v[0], (int, np.integer))
            and (isinstance(v[0], np.uint64))
            or (isinstance(v[0], (int, np.integer)) and (int(max(v) >> 32) > 0)),
        ),  # 64-bit integers
        (
            "J",
            lambda v: isinstance(v[0], (int, np.integer)) and (int(max(v) >> 32) == 0),
        ),  # 32-bit integers
        ("L", lambda v: isinstance(v[0], (bool, np.bool_))),  # bools
    ]
    flat_values = np.array(values).flatten()
    for format_code, check in mappings:
        if check(flat_values):
            break
    else:
        return {}

    kwds = {}
    if isinstance(values, np.ndarray):
        # S = values.size
        V, P = np.atleast_2d(values).shape
        if values.ndim == 2:
            kwds["format"] = f"{P:.0f}{format_code}"
            kwds["dim"] = f"({P}, )"
        else:
            kwds["format"] = f"{format_code}"

    else:
        kwds["format"] = format_code
    return kwds
