import numpy as np
from astropy.time import Time
import astropy.coordinates as coord
import astropy.units as u
from scipy.signal import argrelmin
from tqdm import tqdm
from peewee import chunked
import concurrent.futures

from astra.utils import log
from astra.models.source import Source
from astra.models.apogee import ApogeeVisitSpectrum
from astra.models.boss import BossVisitSpectrum

def set_missing_gaia_source_ids_to_null():
    (
        Source
        .update(gaia_dr3_source_id=None)
        .where(Source.gaia_dr3_source_id <= 0)
        .execute()
    )
    (
        Source
        .update(gaia_dr2_source_id=None)
        .where(Source.gaia_dr2_source_id <= 0)
        .execute()
    )

def compute_f_night_time_for_boss_visits(
        where=(
            BossVisitSpectrum.f_night_time.is_null()
        &   BossVisitSpectrum.tai_end.is_null(False)
        &   BossVisitSpectrum.tai_beg.is_null(False) # sometimes we don't have tai_beg or tai_end
        ),
        limit=None, batch_size=1000, n_time=256, max_workers=128):
    """
    Compute `f_night_time`, which is the observation mid-point expressed as a fraction of time between local sunset and sunrise.
    
    :param where:
        A peewee expression to filter the visits to compute `f_night_time` for.
    
    :param limit:
        The maximum number of visits to compute `f_night_time` for.
    
    :param batch_size:
        The number of visits to update at a time.
    
    :param n_time:
        The number of points to use when computing the sun's position.

    :param max_workers:
        The maximum number of workers to use when computing `f_night_time`.
    """
        
    q = (
        BossVisitSpectrum
        .select()
        .where(where)
        .limit(limit)
    )

    get_obs_time = lambda v: Time((v.tai_beg + 0.5 * (v.tai_end - v.tai_beg))/(24*3600), format="mjd").datetime

    return _compute_f_night_time_for_visits(q, BossVisitSpectrum, get_obs_time, batch_size, n_time, max_workers)

    
def compute_f_night_time_for_apogee_visits(where=ApogeeVisitSpectrum.f_night_time.is_null(), limit=None, batch_size=1000, n_time=256, max_workers=128):
    """
    Compute `f_night_time`, which is the observation mid-point expressed as a fraction of time between local sunset and sunrise.
    
    :param where:
        A peewee expression to filter the visits to compute `f_night_time` for.
    
    :param limit:
        The maximum number of visits to compute `f_night_time` for.
    
    :param batch_size:
        The number of visits to update at a time.
    
    :param n_time:
        The number of points to use (per 24 hour period) when computing the sun's position.
    
    :param max_workers:
        The maximum number of workers to use when computing `f_night_time`.        
    """
        
    q = (
        ApogeeVisitSpectrum
        .select()
        .where(where)
        .limit(limit)
    )
    return _compute_f_night_time_for_visits(q, ApogeeVisitSpectrum, lambda v: v.date_obs, batch_size, n_time, max_workers)


def _compute_f_night_time(pk, observatory, time, n_time):
    # Thanks to Adrian Price-Whelan for blogging about this. This code from his post: https://adrian.pw/blog/sunset-times/
    time_grid = time + np.linspace(-24, 24, 2 * n_time) * u.hour

    sun = coord.get_sun(time_grid[:, None])
    altaz_frame = coord.AltAz(location=observatory, obstime=time_grid[:, None])
    sun_altaz = sun.transform_to(altaz_frame)

    min_idx = np.array(
        [argrelmin(a**2, axis=0, mode="wrap")[0] for a in sun_altaz.alt.degree.T]
    ).flatten()

    # Take the two closest to the middle (e.g., the observing time)
    sunset_idx, sunrise_idx = min_idx[min_idx.searchsorted(n_time) - 1:][:2]

    f_night_time = ((time - time_grid[sunset_idx]) / (time_grid[sunrise_idx] - time_grid[sunset_idx])).value
    return (pk, f_night_time)



def _compute_f_night_time_for_visits(q, model, get_obs_time, batch_size, n_time, max_workers):
        
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)

    futures, visit_by_pk, observatories = ([], {}, {})
    for visit in tqdm(q.iterator(), desc="Submitting jobs", total=1):
        time = Time(get_obs_time(visit))
        observatory_name = visit.telescope[:3].upper()

        try:
            observatory = observatories[observatory_name]
        except:
            observatory = observatories[observatory_name] = coord.EarthLocation.of_site(observatory_name)
        
        futures.append(executor.submit(_compute_f_night_time, visit.pk, observatory, time, n_time))
        visit_by_pk[visit.pk] = visit

    updated = []
    n_updated = 0

    with tqdm(total=len(futures), desc="Computing") as pb:
        for future in concurrent.futures.as_completed(futures):
            pk, f_night_time = future.result()
            visit = visit_by_pk[pk]
            visit.f_night_time = f_night_time
            if not (0 <= f_night_time <= 1):
                log.warning(f"Bad f_night_time for {visit} (f_night_time={f_night_time})")
            updated.append(visit)
            pb.update()                        

            if len(updated) >= batch_size:
                n_updated += (
                    model
                    .bulk_update(
                        updated,
                        fields=[model.f_night_time],
                    )
                )
                updated = []

    if len(updated) > 0:
        n_updated += (
            model
            .bulk_update(
                updated,
                fields=[model.f_night_time],
            )
        )    
        
    return n_updated    
