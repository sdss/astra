import pandas as pd
from util import Lookup, mask_unique_id, drop_duplicates
from typing import Optional
import copy


def clean_APOGEE(data: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the APOGEENet data files for use.

    :param data: APOGEENet data.

    :returns: cleaned DataFrame
    """
    return mask_unique_id(data)


def clean_ASPCAP(
    data_visits: pd.DataFrame,
    data_stars: Optional[pd.DataFrame] = None,
    zero_lower_bound: bool = False,
):
    """
    Clean the ASPCAP data files for use, removing invalid values, etc.

    :param data_visits: visit dataframe
    :param data_stars: coadd dataframe
    :param zero_lower_bound:
        whether lower bound for uncertainty validity is zero or above zero. Defaults to False.

    :returns: tuple of cleaned DataFrames, or just a DataFrame if 1 is parsed
    """
    vars = Lookup.vars
    uvars = Lookup.uvars
    infos = Lookup.infos + ["TELESCOPE"]
    flags = Lookup.flags
    uvardict = copy.copy(Lookup.uvardict)  # need shallow copy

    # filter out crap values to NaN mask
    # (coadd does not depend on uvar so it is ignored)
    concat_mask_visits = pd.DataFrame()
    concat_mask_stars = pd.DataFrame()

    # Patch to fix
    # TODO: make this handling more elegant
    if zero_lower_bound:
        for uvar in uvars:
            uvardict[uvar][0] = 0

    # Mask to keep all info data
    for col in infos:
        concat_mask_visits = pd.concat(
            (concat_mask_visits, pd.Series([False] * len(data_visits),
                                           name=col)),
            axis=1,
        )

    # Get valids for variables
    for var in vars:
        concat_mask_visits = pd.concat(
            (
                concat_mask_visits,
                ~((data_visits[var] >= Lookup.vardict[var][0])
                  & (data_visits[var] <= Lookup.vardict[var][1])),
            ),
            axis=1,
        )

    # Get valids for uncertainties
    for uvar in uvars:
        concat_mask_visits = pd.concat(
            (
                concat_mask_visits,
                ~((data_visits[uvar] >= uvardict[uvar][0])
                  & (data_visits[uvar] <= uvardict[uvar][1])),
            ),
            axis=1,
        )

    # Get valids for flags
    for flag in flags:
        mask = (data_visits[flag] > 0).rename(flag[:-6])
        mask = mask.values | concat_mask_visits[flag[:-6]].values
        cols = list(concat_mask_visits.columns.values)
        cols.remove(flag[:-6])
        concat_mask_visits = pd.concat(
            (concat_mask_visits[cols], pd.Series(mask, name=flag[:-6])),
            axis=1)

    data_visits = data_visits.mask(concat_mask_visits)

    # mask invalid ID's with backup ID
    data_visits = mask_unique_id(data_visits)

    # drop duplicate spectra
    data_visits = drop_duplicates(data_visits)

    # nan will only occur in ID's now, so we drop them
    data_visits = data_visits.dropna(subset=["GAIA_DR3_SOURCE_ID"])

    # Process coadd file if given
    if data_stars is not None:
        # Masking
        # Keeping all info (IDs, SNR)
        for col in infos:
            concat_mask_stars = pd.concat(
                (concat_mask_stars,
                 pd.Series([False] * len(data_stars), name=col)),
                axis=1,
            )
        # Taking valid values
        for var in vars:
            concat_mask_stars = pd.concat(
                (
                    concat_mask_stars,
                    ~((data_stars[var] >= Lookup.vardict[var][0])
                      & (data_stars[var] <= Lookup.vardict[var][1])),
                ),
                axis=1,
            )
        # Taking valid uncertainties
        for uvar in uvars:
            concat_mask_stars = pd.concat(
                (
                    concat_mask_stars,
                    ~((data_stars[uvar] >= uvardict[uvar][0])
                      & (data_stars[uvar] <= uvardict[uvar][1])),
                ),
                axis=1,
            )

        data_stars = data_stars.mask(concat_mask_stars)

        # mask invalid ID's with backup ID
        data_stars = mask_unique_id(data_stars)

        # drop duplicate spectra
        data_stars = drop_duplicates(data_stars)

        # drop duplicate ids in coadd file
        data_stars = drop_duplicates(data_stars, key="GAIA_DR3_SOURCE_ID")

        # drop NaN ID's if any
        data_stars = data_stars.dropna(subset=["GAIA_DR3_SOURCE_ID"])

        # return both as tuple
        return data_visits, data_stars

    return data_visits
