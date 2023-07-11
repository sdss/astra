import datetime
from typing import Union, Iterable
from astropy.io import fits

from astra import (task, __version__)
from astra.models.apogee import ApogeeVisitSpectrum
from astra.models.boss import BossVisitSpectrum
from astra.models.source import Source


@task
def create_mwm_visit_and_mwm_star_data_products(sources: Iterable[Union[Source, int]]):
    """
    Create Milky Way Mapper data products (mwmVisit and mwmStar) for the given source.

    :param source:
        The SDSS-V source to create data products for.
    """

    for source in sources:

        # Create the primary HDU card

        # In each HDU, put different spectra.

        cards = _create_primary_hdu_cards(source)

        #source.boss_visit_spectra

        q = (
            source.apogee_visit_spectra.where(
                (ApogeeVisitSpectrum.telescope == "apo25m")
            )
        )

        raise a





DATETIME_FMT = "%y-%m-%d %H:%M:%S"


def _create_primary_hdu_cards(source, hdu_descriptions=None, nside=128):

    gaia_dr = 3
    created = datetime.datetime.utcnow().strftime(DATETIME_FMT)
    # TODO: Remove descriptions from here, use glossary instead.
    cards = [
        BLANK_CARD,
        (" ", "METADATA", None),
        ("V_ASTRA", __version__, f"Astra version"),
        ("CREATED", created, f"File creation time (UTC {DATETIME_FMT})"),
        ("HEALPIX", source.healpix, f"Healpix location ({nside} sides)"),
        BLANK_CARD,
        (" ", "IDENTIFIERS", None),
        ("SDSS_ID", source.sdss_id, f"SDSS-V identifier"),
        ("GAIA3_ID", source.gaia_dr3_source_id, "Gaia DR3 source identifier"),
        ("GAIA2_ID", source.gaia_dr2_source_id, "Gaia DR2 source identifier"),
        ("SDSS4_ID", source.sdss4_dr17_apogee_id, "SDSS4 DR17 APOGEE Identifier"),
        ("TIC_ID", source.tic_v8_id, f"TESS Input Catalog (v8) identifier"),

        ("CAT_ID", source.catalogid, f"SDSS-V catalog identifier"),        
        ("CAT_ID05", source.catalogid_v0p5, f"SDSS-V catalog identifier (v0.5)"),
        ("CAT_ID10", source.catalogid_v1, f"SDSS-V catalog identifier (v1)"),
        BLANK_CARD,
        (" ", "ASTROMETRY", None),
        ("RA", source.ra, "SDSS-V catalog right ascension (J2000) [deg]"),
        ("DEC", source.dec, "SDSS-V catalog declination (J2000) [deg]"),
        ("PLX", source.plx, f"Gaia {gaia_dr} parallax [mas]"),
        ("E_PLX", source.e_plx, f"Gaia {gaia_dr} parallax error [mas]"),
        ("PMRA", source.pmra, f"Gaia {gaia_dr} proper motion in RA [mas/yr]"),
        (
            "E_PMRA",
            source.e_pmra,
            f"Gaia {gaia_dr} proper motion in RA error [mas/yr]",
        ),
        ("PMDE", source.pmde, f"Gaia {gaia_dr} proper motion in DEC [mas/yr]"),
        (
            "E_PMDE",
            source.e_pmde,
            f"Gaia {gaia_dr} proper motion in DEC error [mas/yr]",
        ),
        ("V_RAD", source.gaia_v_rad, f"Gaia {gaia_dr} radial velocity [km/s]"),
        (
            "E_V_RAD",
            source.gaia_e_v_rad,
            f"Gaia {gaia_dr} radial velocity error [km/s]",
        ),
        BLANK_CARD,
        (" ", "PHOTOMETRY", None),
        (
            "G_MAG",
            source.g_mag,
            f"Gaia {gaia_dr} mean apparent G magnitude [mag]",
        ),
        (
            "BP_MAG",
            source.bp_mag,
            f"Gaia {gaia_dr} mean apparent BP magnitude [mag]",
        ),
        (
            "RP_MAG",
            source.rp_mag,
            f"Gaia {gaia_dr} mean apparent RP magnitude [mag]",
        ),
        ("J_MAG", source.j_mag, f"2MASS mean apparent J magnitude [mag]"),
        ("E_J_MAG", source.e_j_mag, f"2MASS mean apparent J magnitude error [mag]"),
        ("H_MAG", source.h_mag, f"2MASS mean apparent H magnitude [mag]"),
        ("E_H_MAG", source.e_h_mag, f"2MASS mean apparent H magnitude error [mag]"),
        ("K_MAG", source.k_mag, f"2MASS mean apparent K magnitude [mag]"),
        ("E_K_MAG", source.e_k_mag, f"2MASS mean apparent K magnitude error [mag]"),

        # TODO: unWISE
    ]
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