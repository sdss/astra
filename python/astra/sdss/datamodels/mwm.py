"""Create Milky Way Mapper data products (mwmVisit/mwmStar)."""

import datetime
import os
import numpy as np
from astropy.io import fits
from astropy.time import Time
from sdss_access import SDSSPath
from typing import Union, List, Callable, Optional, Dict, Tuple, Iterable

from astra import __version__ as v_astra
from astra.base import task

#from astra.base import TaskInstance, Parameter
from astra.database.astradb import Source, DataProduct, BaseTaskOutput
from astra.utils import log, flatten, expand_path
from astra.sdss.datamodels import base, apogee, boss

from peewee import IntegerField, DateTimeField

HDU_DESCRIPTIONS = [
    "Source information only",
    "BOSS spectra from Apache Point Observatory",
    "BOSS spectra from Las Campanas Observatory",
    "APOGEE spectra from Apache Point Observatory",
    "APOGEE spectra from Las Campanas Observatory",
]

class MWMSourceStatus(BaseTaskOutput):

    num_apogee_apo_visits = IntegerField(default=0)
    num_apogee_lco_visits = IntegerField(default=0)
    num_boss_apo_visits = IntegerField(default=0)
    num_boss_lco_visits = IntegerField(default=0)

    num_apogee_apo_visits_in_stack = IntegerField(default=0)
    num_apogee_lco_visits_in_stack = IntegerField(default=0)
    num_boss_apo_visits_in_stack = IntegerField(default=0)
    num_boss_lco_visits_in_stack = IntegerField(default=0)

    obs_start_apogee_apo = DateTimeField(null=True)
    obs_end_apogee_apo = DateTimeField(null=True)
    obs_start_apogee_lco = DateTimeField(null=True)
    obs_end_apogee_lco = DateTimeField(null=True)

    obs_start_boss_apo = DateTimeField(null=True)
    obs_end_boss_apo = DateTimeField(null=True)
    obs_start_boss_lco = DateTimeField(null=True)
    obs_end_boss_lco = DateTimeField(null=True)

    updated = DateTimeField(default=datetime.datetime.now)


@task
def create_mwm_data_products(
    source: Iterable[Union[Source, int]],
    run2d: str,
    apred: str,
    release: str = "sdss5",
    boss_release: str = "sdss5",
    apogee_release: str = "sdss5",
    boss_kwargs: Optional[Dict] = None,
    apogee_kwargs: Optional[Dict] = None,
) -> Iterable[MWMSourceStatus]: 
    """
    Create Milky Way Mapper data products (mwmVisit and mwmStar) for the given source.

    :param source:
        The SDSS-V source to create data products for.

    :param boss_kwargs: [optional]
        Keyword arguments to pass to the `boss.create_boss_hdus` function.

    :param apogee_kwargs: [optional]
        Keyword arguments to pass to the `apogee.create_apogee_hdus` function.
    """

    for source in flatten(source):
        catalogid = source if isinstance(source, int) else source.catalogid
        if catalogid < 0:
            log.warning(f"Skipping negative catalog identifier {source}")
            continue
    
        log.info(f"Creating data products for source {source}")
        meta = _create_mwm_data_products(
            source,
            run2d=run2d,
            apred=apred,
            release=release,
            boss_release=boss_release,
            apogee_release=apogee_release,
            boss_kwargs=boss_kwargs,
            apogee_kwargs=apogee_kwargs,
        )
        yield MWMSourceStatus(
            source=source, 
            data_product=None,
            **meta
        )


def _create_mwm_data_products(
    source: Union[Source, int],
    run2d: str,
    apred: str,
    release: str = "sdss5",
    boss_release: str = "sdss5",
    apogee_release: str = "sdss5",
    boss_kwargs: Optional[Dict] = None,
    apogee_kwargs: Optional[Dict] = None,
):

    if isinstance(source, int):
        source = Source.get(catalogid=source)

    hdu_visit_list, hdu_star_list, meta = create_mwm_hdus(
        source, 
        run2d=run2d,
        apred=apred,
        boss_release=boss_release,
        apogee_release=apogee_release,
        boss_kwargs=boss_kwargs, 
        apogee_kwargs=apogee_kwargs
    )
    any_stacked_spectra = sum([sum(hdu.data["IN_STACK"]) for hdu in hdu_visit_list if hdu.size > 0]) > 0
    kwds = dict(
        cat_id=source.catalogid,
        v_astra=v_astra,
        run2d=run2d,
        apred=apred,
    )
    p = SDSSPath(release)
    mwmStar_path = p.full("mwmStar", **kwds)
    mwmVisit_path = p.full("mwmVisit", **kwds)

    # Create necessary folders
    os.makedirs(os.path.dirname(mwmVisit_path), exist_ok=True)
    if any_stacked_spectra:
        os.makedirs(os.path.dirname(mwmStar_path), exist_ok=True)
    else:
        # Remove an existing mwmStar file if there are no stacked spectra.
        if os.path.exists(mwmStar_path):
            os.unlink(mwmStar_path)
    
    # Ensure mwmVisit and mwmStar files are always synchronised.
    try:
        hdu_visit_list.writeto(mwmVisit_path, overwrite=True)
        if any_stacked_spectra:
            hdu_star_list.writeto(mwmStar_path, overwrite=True)
    except:
        log.exception(
            f"Exception when trying to write to either:\n{mwmVisit_path}\n{mwmStar_path}"
        )
        # Delete both.
        for path in (mwmVisit_path, mwmStar_path):
            if os.path.exists(path):
                os.unlink(path)
    else:
        log.info(f"Wrote mwmVisit product to {mwmVisit_path}")
    
        # Create output data product records that link to this task.
        dp_visit, visit_created = DataProduct.get_or_create(
            source=source, release=release, filetype="mwmVisit", kwargs=kwds
        )
        log.info(f"Created data product {dp_visit} for source {source}")

        if any_stacked_spectra:
            log.info(f"Wrote mwmStar product to {mwmStar_path}")
            dp_star, star_created = DataProduct.get_or_create(
                source=source, release=release, filetype="mwmStar", kwargs=kwds
            )        
        else:
            log.info(f"No stacked spectra to store for {source}")

    return meta


def create_mwm_hdus(
    source: Union[Source, int],
    run2d: str,
    apred: str,
    apogee_release: str,
    boss_release: str,
    boss_kwargs: Optional[Dict] = None,
    apogee_kwargs: Optional[Dict] = None,
):  # -> Tuple[DataProduct, DataProduct]:
    """
    Create Milky Way Mapper `Visit` and `Star` data products for the given source.

    :param source:
        The SDSS-V source to create data products for.

    :param boss_kwargs: [optional]
        Keyword arguments to pass to the `boss.create_boss_hdus` function.

    :param apogee_kwargs: [optional]
        Keyword arguments to pass to the `apogee.create_apogee_hdus` function.
    """

    boss_filetype, apogee_filetype = ("specFull", "apVisit")

    data_products = list(source.data_products)

    cards = base.create_primary_hdu_cards(source, HDU_DESCRIPTIONS)
    primary_visit_hdu = fits.PrimaryHDU(header=fits.Header(cards))
    primary_star_hdu = fits.PrimaryHDU(header=fits.Header(cards))

    is_boss = lambda dp: (
        (dp.filetype == boss_filetype) 
    &   ((run2d is None) or (dp.kwargs.get("run2d", None) == run2d))
    &   ((boss_release is None) or (dp.release == boss_release))
    )
    is_apogee_dr17_apstar = lambda dp: (dp.filetype == "apStar") & (dp.release == "dr17")
    is_apogee_sdss5_apvisit = lambda dp: (
        (dp.filetype == apogee_filetype)
    &   ((apred is None) or (dp.kwargs.get("apred", None) == apred))
    &   ((apogee_release is None) or (dp.release == apogee_release))
    )
    is_apogee = lambda dp: is_apogee_dr17_apstar(dp) or is_apogee_sdss5_apvisit(dp)

    ignored = [dp for dp in data_products if not is_apogee(dp) and not is_boss(dp)]
    for dp in ignored:
        log.warning(
            f"Ignoring file type '{dp.filetype}' ({dp}: {dp.path}). It's not used for creating MWM Visit/Star products."
        )

    boss_north_visits, boss_north_star = boss.create_boss_hdus(
        list(filter(is_boss, data_products)),
        observatory="APO",
        **(boss_kwargs or dict()),
    )

    apogee_north_visits, apogee_south_visits, apogee_north_star, apogee_south_star = apogee.create_apogee_hdus(
        list(filter(is_apogee, data_products)),
        **(apogee_kwargs or dict()),
    )

    boss_south_visits = base.create_empty_hdu("LCO", "BOSS")
    boss_south_star = base.create_empty_hdu("LCO", "BOSS")

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

    def _safe_datetime_from_mjd_in_stack(hdu, statistic):
        # If there are no data in the stack then we are taking a min/max of empty sequence
        try:
            return Time(statistic(hdu.data["MJD"][hdu.data["IN_STACK"]]), format="mjd").datetime
        except ValueError:
            return None

    # Create a metadata dictionary
    is_empty_hdu = lambda hdu: hdu.data is None or hdu.data.size == 0
    get_num_visits = lambda hdu: 0 if is_empty_hdu(hdu) else len(hdu.data)
    get_num_visits_in_stack = (
        lambda hdu: 0 if is_empty_hdu(hdu) else np.sum(hdu.data["IN_STACK"])
    )
    get_obs_start = (
        lambda hdu: None
        if is_empty_hdu(hdu)
        else _safe_datetime_from_mjd_in_stack(hdu, min)
    )
    get_obs_end = (
        lambda hdu: None
        if is_empty_hdu(hdu)
        else _safe_datetime_from_mjd_in_stack(hdu, max)
    )

    source_id = source.catalogid if isinstance(source, Source) else source

    meta = {
        "source_id": source_id,
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

