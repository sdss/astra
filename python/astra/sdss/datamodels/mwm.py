import datetime
from astropy.io import fits
from astropy.time import Time
from typing import Union, List, Callable, Optional, Dict, Tuple

from astra.database.astradb import Source, DataProduct

from astra.sdss.datamodels import base, apogee, boss
from astra import log

HDU_DESCRIPTIONS = [
    "Source information only",
    "BOSS spectra from Apache Point Observatory",
    "BOSS spectra from Las Campanas Observatory",
    "APOGEE spectra from Apache Point Observatory",
    "APOGEE spectra from Las Campanas Observatory",
]


def create_mwm_data_products(
    source: Union[Source, int],
    input_data_products: Optional[List[DataProduct]] = None,
    boss_kwargs: Optional[Dict] = None,
    apogee_kwargs: Optional[Dict] = None,
):  # -> Tuple[DataProduct, DataProduct]:
    """
    Create Milky Way Mapper `Visit` and `Star` data products for the given source.

    :param source:
        The SDSS-V source to create data products for.

    :param input_data_products: [optional]
        The input data products to use when creating these products. If `None`
        is given then all possible data products linked to this source will be used.

    :param boss_kwargs: [optional]
        Keyword arguments to pass to the `boss.create_boss_hdus` function.

    :param apogee_kwargs: [optional]
        Keyword arguments to pass to the `apogee.create_apogee_hdus` function.
    """

    input_filetypes = boss_filetype, apogee_filetype = ("specFull", "apVisit")

    if input_data_products is None:
        input_data_products = tuple(
            [dp for dp in source.data_products if dp.filetype in input_filetypes]
        )

    for dp in input_data_products:
        if dp.filetype not in input_filetypes:
            log.warning(
                f"Ignoring file type '{dp.filetype}' ({dp}: {dp.path}). It's not used for creating MWM Visit/Star products."
            )

    cards = base.create_primary_hdu_cards(source, HDU_DESCRIPTIONS)
    primary_visit_hdu = fits.PrimaryHDU(header=fits.Header(cards))
    primary_star_hdu = fits.PrimaryHDU(header=fits.Header(cards))

    boss_north_visits, boss_north_star = boss.create_boss_hdus(
        [dp for dp in input_data_products if dp.filetype == boss_filetype],
        observatory="APO",
        **(boss_kwargs or dict()),
    )

    boss_south_visits = base.create_empty_hdu("LCO", "BOSS")
    boss_south_star = base.create_empty_hdu("LCO", "BOSS")

    apogee_north_visits, apogee_north_star = apogee.create_apogee_hdus(
        [
            dp
            for dp in input_data_products
            if dp.filetype == apogee_filetype and dp.kwargs["telescope"] == "apo25m"
        ],
        observatory="APO",
        **(apogee_kwargs or dict()),
    )
    apogee_south_visits, apogee_south_star = apogee.create_apogee_hdus(
        [
            dp
            for dp in input_data_products
            if dp.filetype == apogee_filetype and dp.kwargs["telescope"] == "lco25m"
        ],
        observatory="LCO",
        **(apogee_kwargs or dict()),
    )

    hdu_visit_list = fits.HDUList(
        [
            primary_visit_hdu,
            boss_north_visits,
            boss_south_visits,
            apogee_north_visits,
            apogee_south_visits,
        ]
    )
    hdu_star_list = fits.HDUList(
        [
            primary_star_hdu,
            boss_north_star,
            boss_south_star,
            apogee_north_star,
            apogee_south_star,
        ]
    )

    # Add checksums and datasums to each HDU.
    base.add_check_sums(hdu_visit_list)
    base.add_check_sums(hdu_star_list)

    # Create a metadata dictionary
    is_empty_hdu = lambda hdu: hdu.data is None or hdu.data.size == 0
    get_num_visits = lambda hdu: 0 if is_empty_hdu(hdu) else len(hdu.data)
    get_num_visits_in_stack = (
        lambda hdu: 0 if is_empty_hdu(hdu) else sum(hdu.data["IN_STACK"])
    )
    get_obs_start = (
        lambda hdu: None
        if is_empty_hdu(hdu)
        else Time(min(hdu.data["MJD"][hdu.data["IN_STACK"]]), format="mjd").datetime
    )
    get_obs_end = (
        lambda hdu: None
        if is_empty_hdu(hdu)
        else Time(max(hdu.data["MJD"][hdu.data["IN_STACK"]]), format="mjd").datetime
    )

    meta = {
        "num_apogee_apo_visits": get_num_visits(apogee_north_visits),
        "num_apogee_lco_visits": get_num_visits(apogee_south_visits),
        "num_boss_apo_visits": get_num_visits(boss_north_visits),
        "num_boss_lco_visits": get_num_visits(boss_south_visits),
        "num_apogee_apo_visits_in_stack": get_num_visits_in_stack(apogee_north_visits),
        "num_apogee_lco_visits_in_stack": get_num_visits_in_stack(apogee_south_visits),
        "num_boss_apo_visits_in_stack": get_num_visits_in_stack(boss_north_visits),
        "num_boss_lco_visits_in_stack": get_num_visits_in_stack(boss_south_visits),
        # These obs start/end just refer to those used in the stack.
        "obs_start_apogee_apo": get_obs_start(apogee_north_visits),
        "obs_end_apogee_apo": get_obs_end(apogee_north_visits),
        "obs_start_apogee_lco": get_obs_start(apogee_south_visits),
        "obs_end_apogee_lco": get_obs_end(apogee_south_visits),
        "obs_start_boss_apo": get_obs_start(boss_north_visits),
        "obs_end_boss_apo": get_obs_end(boss_north_visits),
        "obs_start_boss_lco": get_obs_start(boss_south_visits),
        "obs_end_boss_lco": get_obs_end(boss_south_visits),
        "updated": datetime.datetime.now(),
    }

    # Define the paths, return data products
    return (hdu_visit_list, hdu_star_list, meta)


def get_hdu_index(filetype, telescope):
    if filetype == "specFull":
        return 1  # TODO: revise when we have BOSS spectra from LCO
    if filetype in ("apStar", "apVisit", "apStar-1m"):
        if telescope in ("apo25m", "apo1m"):
            return 3
        elif telescope == "lco25m":
            return 4
    raise ValueError(f"Unknown filetype/telescope combination: {filetype}/{telescope}")


def get_data_hdu_observatory_and_instrument():
    return [
        ("APO", "BOSS"),
        ("LCO", "BOSS"),
        ("APO", "APOGEE"),
        ("LCO", "APOGEE"),
    ]
