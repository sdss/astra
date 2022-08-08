from astropy.io import fits
from typing import Union, List, Callable, Optional, Dict, Tuple

from astra.database.astradb import Source, DataProduct

from astra.sdss.dm import (base, apogee, boss)


def create_mwm_data_products(
    source: Union[Source, int],
    boss_kwargs: Optional[Dict] = None,
    apogee_kwargs: Optional[Dict] = None,
):# -> Tuple[DataProduct, DataProduct]:
    """
    Create Milky Way Mapper `Visit` and `Star` data products for the given source.

    :param source:
        The SDSS-V source to create data products for.
    
    :param boss_kwargs: [optional]
        Keyword arguments to pass to the `boss.create_boss_hdus` function.
    
    :param apogee_kwargs: [optional]
        Keyword arguments to pass to the `apogee.create_apogee_hdus` function.
    """

    hdu_descriptions = [
        "Source information only",
        "BOSS spectra from Apache Point Observatory",
        "BOSS spectra from Las Campanas Observatory",
        "APOGEE spectra from Apache Point Observatory",
        "APOGEE spectra from Las Campanas Observatory"
    ]
    primary_hdu = base.create_primary_hdu(source, hdu_descriptions)

    boss_north_visits, boss_north_star = boss.create_boss_hdus(
        [dp for dp in source.data_products if dp.filetype == "specLite"],
        **(boss_kwargs or dict())
    )
    # TODO: Eventually we might have some BOSS spectra from Las Campanas..
    boss_south_visits = boss_south_star = base.create_empty_hdu("LCO", "BOSS")

    apogee_north_visits, apogee_north_star = apogee.create_apogee_hdus(
        [dp for dp in source.data_products \
            if dp.filetype == "apVisit" and dp.kwargs["telescope"] == "apo25m"],
        **(apogee_kwargs or dict())
    )
    apogee_south_visits, apogee_south_star = apogee.create_apogee_hdus(
        [dp for dp in source.data_products \
            if dp.filetype == "apVisit" and dp.kwargs["telescope"] == "lco25m"],
        **(apogee_kwargs or dict())
    )

    hdu_visit_list = fits.HDUList([
        primary_hdu,
        boss_north_visits,
        boss_south_visits,
        apogee_north_visits,
        apogee_south_visits,
    ])
    hdu_star_list = fits.HDUList([
        primary_hdu,
        boss_north_star,
        boss_south_star,
        apogee_north_star,
        apogee_south_star,
    ])

    # Define the paths, return data products
    return (hdu_visit_list, hdu_star_list)